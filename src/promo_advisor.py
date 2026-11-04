"""Promo Advisor — Phase 1 & 2

Operator บอก "กำลังจะทำโปรนี้" → ระบบวิเคราะห์:
  1. โปรของ operator ถ้าส่งให้ใครได้ผลยังไง (CFO guardrail)
  2. เปรียบกับ ML recommendation ต่อคน — อะไรกำไรกว่า
  3. Conflict: ถ้าเป็นประเภทเดียวกัน เลือกอันดีกว่า
  4. Scenario A/B/C พร้อมตัวเลขและเหตุผล

Usage:
    from promo_advisor import PromoSpec, promo_advisor, print_promo_report

    spec = PromoSpec(
        name="B1G1 ครัวซองต์",
        discount_rate=0.50,
        scope="category",
        category="ครัวซองต์และเบเกอรี่",
        target="all_customers",
    )
    result_df, report = promo_advisor(ml_output_df, cust_agg_df, spec)
    print_promo_report(report)

Phase 2 additions:
    from promo_advisor import (analyze_promo_history, multi_promo_advisor,
                                add_cannibalization_risk, print_multi_promo_report)

    # Historical analysis
    history = analyze_promo_history(pos_df, history_path="data/processed/promo_history.jsonl")
    print_history_report(history)

    # Multi-promo (operator has 2 promos this month)
    specs = [PromoSpec(...), PromoSpec(...)]
    resolved_df, report = multi_promo_advisor(ml_output, cust_agg, specs)
    print_multi_promo_report(report)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

# ── Promo type grouping for dedup ─────────────────────────────────────────────
# If operator promo and ML recommendation share the same group → conflict
_PROMO_TYPE_GROUP: dict[str, str] = {
    # ML action names → group
    "push promotion: bundle":   "discount",
    "off-peak special":         "discount",
    "off-peak traffic driver":  "discount",
    "off-peak vip points":      "points_vip",
    "vip reward":               "points_vip",
    "vip early access":         "points_vip",
    "win-back offer":           "retention",
    "churn rescue":             "retention",
    "welcome offer":            "welcome",
    "milestone reward":         "points_vip",
    "zero-cost: เราคิดถึงคุณ": "notification",
    # Operator promo types → group
    "blanket_discount":         "discount",
    "bundle_discount":          "discount",
    "category_discount":        "discount",
    "bogo":                     "discount",
    "points":                   "points_vip",
    "notification":             "notification",
    "winback":                  "retention",
    "welcome":                  "welcome",
}


def _promo_group(action_or_type: str) -> str:
    """Map action name or promo type to a conflict-detection group."""
    key = action_or_type.lower().strip()
    for k, v in _PROMO_TYPE_GROUP.items():
        if k in key:
            return v
    return "other"


# ── PromoSpec ─────────────────────────────────────────────────────────────────

@dataclass
class PromoSpec:
    """Describes an operator-planned promotion.

    Parameters
    ----------
    name         : Human-readable label shown in reports
    discount_rate: Effective discount as fraction of avg_spent (e.g. 0.50 for B1G1)
    promo_type   : broad category — "discount" | "points_vip" | "retention" |
                   "notification" | "welcome" | "other"
    scope        : "all_items" | "category" | "specific_items"
    category     : product category name (required when scope="category")
    items        : list of product names (required when scope="specific_items")
    target       : "all_customers" | "segment" | list[customer_id str]
    segments     : list of IBM segment names (required when target="segment")
    start_date   : optional — used for history logging only
    end_date     : optional — used for history logging only
    force        : if True, override ML even when profit difference < threshold
    tie_threshold: fraction — if abs(profit_ml - profit_promo)/max < this → operator wins
    """
    name:           str
    discount_rate:  float
    promo_type:     str                      = "discount"
    scope:          str                      = "all_items"
    category:       str | None               = None
    items:          list[str]                = field(default_factory=list)
    target:         str | list[str]          = "all_customers"
    segments:       list[str]                = field(default_factory=list)
    start_date:     str | None               = None
    end_date:       str | None               = None
    force:          bool                     = False
    tie_threshold:  float                    = 0.10  # within 10% → conservative, keep operator


# ── CFO Guardrail applied to operator promo ──────────────────────────────────

def _promo_profit_per_customer(
    row: pd.Series,
    spec: PromoSpec,
    cogs_rate: float,
    min_margin: float,
) -> tuple[bool, float, float]:
    """
    Returns (viable, expected_profit, promo_cost) for the operator promo
    applied to this customer.

    Formula mirrors _viable() in uplift_engine:
      promo_cost  = avg_spent * discount_rate
      gross_rev   = avg_spent * (1 - cogs_rate)
      net_margin  = (gross_rev - promo_cost) / avg_spent
      incremental = gross_rev * uplift_score - promo_cost
      viable      = net_margin >= min_margin AND incremental > 0
    """
    avg_spent = float(row.get("avg_spent", 0.0))
    if avg_spent <= 0:
        return False, 0.0, 0.0

    uplift    = float(row.get("uplift_score", row.get("predicted_uplift", 0.0)))
    p_control = float(row.get("prob_control", 0.5))
    gross_margin = avg_spent * (1.0 - cogs_rate)

    promo_cost   = round(avg_spent * spec.discount_rate, 2)
    gross_rev    = gross_margin
    net_margin   = (gross_rev - promo_cost) / avg_spent if avg_spent > 0 else 0.0
    incremental  = round(gross_margin * uplift - promo_cost, 2)

    viable = (net_margin >= min_margin) and (incremental > 0)
    profit = round(gross_margin * (p_control + uplift) - promo_cost, 2)
    return viable, profit, promo_cost


def evaluate_operator_promo(
    ml_output: pd.DataFrame,
    cust_agg: pd.DataFrame,
    spec: PromoSpec,
    cogs_rate: float = 0.50,
    min_margin: float = 0.15,
) -> pd.DataFrame:
    """
    Evaluate operator promo against every customer.

    Returns cust_agg with new columns:
        promo_eligible         bool  — in target scope (segment / customer list)
        promo_viable           bool  — passes CFO guardrail
        operator_promo_profit  float
        operator_promo_cost    float
        operator_promo_blocked_reason  str | ""
    """
    df = cust_agg.copy()

    # ── Determine eligible customers ─────────────────────────────────────────
    if spec.target == "all_customers":
        df["promo_eligible"] = True
    elif spec.target == "segment" and spec.segments:
        seg_lower = {s.lower().strip() for s in spec.segments}
        df["promo_eligible"] = df.get(
            "cluster_segment_name",
            df.get("tier", pd.Series("", index=df.index))
        ).str.lower().str.strip().isin(seg_lower)
    elif isinstance(spec.target, list):
        target_set = {str(x) for x in spec.target}
        df["promo_eligible"] = df["customer_id"].astype(str).isin(target_set)
    else:
        df["promo_eligible"] = False

    # ── Category / item scope filter ─────────────────────────────────────────
    if spec.scope == "category" and spec.category:
        # Check if customer's top product category matches
        top_cat_col = next(
            (c for c in ["top_category", "primary_category"] if c in df.columns),
            None,
        )
        if top_cat_col:
            df["promo_eligible"] &= (
                df[top_cat_col].str.lower().str.strip()
                == spec.category.lower().strip()
            )
        # If no category column, leave eligible as is (can't filter further)

    elif spec.scope == "specific_items" and spec.items:
        item_lower = {i.lower().strip() for i in spec.items}
        top_item_col = next(
            (c for c in ["top_product", "top_bundle_name"] if c in df.columns),
            None,
        )
        if top_item_col:
            df["promo_eligible"] &= (
                df[top_item_col].str.lower().str.strip().isin(item_lower)
            )

    # ── Merge ML uplift / prob_control if available ───────────────────────────
    ml_cols = ["customer_id", "uplift_score", "predicted_uplift",
               "prob_control", "avg_spent"]
    avail_ml = [c for c in ml_cols if c in ml_output.columns]
    if avail_ml and "customer_id" in ml_output.columns:
        df = df.merge(
            ml_output[list(set(["customer_id"] + avail_ml))].drop_duplicates("customer_id"),
            on="customer_id",
            how="left",
            suffixes=("", "_ml"),
        )
        # Fill missing uplift / prob with neutral values
        for col, default in [("uplift_score", 0.0), ("prob_control", 0.5)]:
            if col not in df.columns:
                alt = "predicted_uplift" if col == "uplift_score" else col
                if alt in df.columns:
                    df[col] = df[alt]
                else:
                    df[col] = default
            df[col] = df[col].fillna(default)
    else:
        df["uplift_score"] = 0.0
        df["prob_control"]  = 0.5

    # ── Run CFO guardrail per customer ────────────────────────────────────────
    results = df.apply(
        lambda row: _promo_profit_per_customer(row, spec, cogs_rate, min_margin),
        axis=1,
    )
    df["promo_viable"]             = [r[0] for r in results]
    df["operator_promo_profit"]    = [r[1] for r in results]
    df["operator_promo_cost"]      = [r[2] for r in results]

    # Non-eligible → not viable
    df.loc[~df["promo_eligible"], "promo_viable"] = False
    df.loc[~df["promo_eligible"], "operator_promo_profit"] = 0.0

    def _block_reason(row):
        if not row["promo_eligible"]:
            return "not in target segment/list"
        if not row["promo_viable"]:
            avg = row.get("avg_spent", 0)
            cost = row["operator_promo_cost"]
            if avg <= 0:
                return "avg_spent = 0"
            margin = (avg * (1 - cogs_rate) - cost) / avg
            if margin < min_margin:
                return f"net margin {margin:.1%} < floor {min_margin:.0%}"
            return "incremental profit ≤ 0 (uplift too low vs promo cost)"
        return ""

    df["operator_promo_blocked_reason"] = df.apply(_block_reason, axis=1)

    return df


# ── Per-customer resolution ───────────────────────────────────────────────────

def resolve_per_customer(
    ml_output: pd.DataFrame,
    promo_eval: pd.DataFrame,
    spec: PromoSpec,
) -> pd.DataFrame:
    """
    Merge ML output with promo evaluation and pick the best action per customer.

    New columns added:
        ml_action          str
        ml_profit          float
        ml_should_send     int
        conflict_type      "same_type" | "compatible" | "none"
        final_source       "ml" | "operator" | "both_compatible" | "none"
        final_action       str
        final_profit       float
        final_promo_cost   float
        final_should_send  int
        advisor_reason     str
    """
    # ── Bring in ML columns ───────────────────────────────────────────────────
    ml_cols = ["customer_id", "recommended_action", "expected_profit",
               "promo_cost", "should_send", "treatment_type"]
    avail   = [c for c in ml_cols if c in ml_output.columns]
    ml_slim = ml_output[avail].copy().rename(columns={
        "recommended_action": "ml_action",
        "expected_profit":    "ml_profit",
        "promo_cost":         "ml_promo_cost",
        "should_send":        "ml_should_send",
        "treatment_type":     "ml_treatment_type",
    })

    df = promo_eval.merge(ml_slim, on="customer_id", how="left")
    df["ml_should_send"]  = df.get("ml_should_send", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["ml_profit"]       = df.get("ml_profit", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["ml_action"]       = df.get("ml_action", pd.Series("", index=df.index)).fillna("")
    df["ml_treatment_type"] = df.get("ml_treatment_type", pd.Series("none", index=df.index)).fillna("none")

    op_group = _promo_group(spec.promo_type)

    def _resolve(row):
        ml_send    = int(row["ml_should_send"]) == 1
        op_viable  = bool(row["promo_viable"])
        ml_profit  = float(row["ml_profit"])
        op_profit  = float(row["operator_promo_profit"])
        ml_action  = str(row["ml_action"])
        ml_group   = _promo_group(str(row.get("ml_treatment_type", "")))
        conflict   = (ml_group == op_group) and ml_group != "other"

        # Neither sends
        if not ml_send and not op_viable:
            return ("none", "none", "", 0.0, 0.0, 0,
                    "ทั้ง ML และโปรร้านไม่ผ่าน guardrail")

        # Only operator
        if op_viable and not ml_send:
            return ("none", "operator", spec.name, op_profit,
                    float(row["operator_promo_cost"]), 1,
                    "ML ไม่แนะนำส่ง → ใช้โปรร้านที่ผ่าน guardrail")

        # Only ML
        if ml_send and not op_viable:
            block = str(row.get("operator_promo_blocked_reason", "ไม่ผ่าน guardrail"))
            return ("none", "ml", ml_action, ml_profit,
                    float(row.get("ml_promo_cost", 0.0)), 1,
                    f"โปรร้านไม่ผ่าน guardrail ({block}) → ใช้ ML")

        # Both viable
        if conflict:
            # Same type — pick higher profit (or operator if within tie threshold)
            if spec.force:
                winner = "operator"
            else:
                diff_ratio = abs(ml_profit - op_profit) / max(abs(ml_profit), abs(op_profit), 0.01)
                winner = "operator" if (diff_ratio < spec.tie_threshold or op_profit >= ml_profit) else "ml"

            if winner == "operator":
                return ("same_type", "operator", spec.name, op_profit,
                        float(row["operator_promo_cost"]), 1,
                        f"ประเภทเดียวกัน ({op_group}) → เลือกโปรร้าน "
                        f"(กำไร ฿{op_profit:,.0f} vs ML ฿{ml_profit:,.0f})")
            else:
                return ("same_type", "ml", ml_action, ml_profit,
                        float(row.get("ml_promo_cost", 0.0)), 1,
                        f"ประเภทเดียวกัน ({op_group}) → ML กำไรสูงกว่า "
                        f"฿{ml_profit - op_profit:,.0f} (฿{ml_profit:,.0f} vs ฿{op_profit:,.0f})")
        else:
            # Compatible — pick higher profit
            if op_profit >= ml_profit:
                return ("compatible", "operator", spec.name, op_profit,
                        float(row["operator_promo_cost"]), 1,
                        f"โปรร้านกำไรสูงกว่า ฿{op_profit - ml_profit:,.0f} "
                        f"(฿{op_profit:,.0f} vs ML ฿{ml_profit:,.0f})")
            else:
                return ("compatible", "ml", ml_action, ml_profit,
                        float(row.get("ml_promo_cost", 0.0)), 1,
                        f"ML กำไรสูงกว่า ฿{ml_profit - op_profit:,.0f} "
                        f"(฿{ml_profit:,.0f} vs โปรร้าน ฿{op_profit:,.0f})")

    resolved = df.apply(_resolve, axis=1, result_type="expand")
    resolved.columns = [
        "conflict_type", "final_source", "final_action",
        "final_profit", "final_promo_cost", "final_should_send",
        "advisor_reason",
    ]
    df = pd.concat([df, resolved], axis=1)
    return df


# ── Scenario comparison ───────────────────────────────────────────────────────

def compare_scenarios(resolved: pd.DataFrame, spec: PromoSpec) -> dict:
    """
    Returns summary dict with Scenario A, B, C.
      A = ML only
      B = Operator promo only (guardrail applied)
      C = Optimal (promo_advisor resolution)
    """
    # Scenario A — ML
    ml_send   = resolved[resolved["ml_should_send"] == 1]
    a_send    = int(len(ml_send))
    a_profit  = round(float(ml_send["ml_profit"].sum()), 2)

    # Scenario B — operator promo (viable customers only)
    op_viable = resolved[resolved["promo_viable"] == True]
    op_eligible = resolved[resolved["promo_eligible"] == True]
    b_send      = int(len(op_viable))
    b_profit    = round(float(op_viable["operator_promo_profit"].sum()), 2)
    b_blocked   = int(len(op_eligible)) - b_send
    b_total_target = int(len(op_eligible))

    # Scenario C — optimal
    c_send    = resolved[resolved["final_should_send"] == 1]
    c_profit  = round(float(c_send["final_profit"].sum()), 2)
    c_n       = int(len(c_send))

    # Breakdown
    from_op   = int((c_send["final_source"] == "operator").sum())
    from_ml   = int((c_send["final_source"] == "ml").sum())
    conflicts = int((resolved["conflict_type"] == "same_type").sum())
    op_suppressed = int(
        ((resolved["promo_viable"] == True) & (resolved["final_source"] == "ml")).sum()
    )
    op_suppressed_reason = "ML กำไรสูงกว่า"

    total = int(len(resolved))

    # Profit deltas
    c_vs_a = round(c_profit - a_profit, 2)
    c_vs_b = round(c_profit - b_profit, 2)
    c_vs_a_pct = round((c_vs_a / max(abs(a_profit), 0.01)) * 100, 1) if a_profit != 0 else None
    c_vs_b_pct = round((c_vs_b / max(abs(b_profit), 0.01)) * 100, 1) if b_profit != 0 else None

    return {
        "promo_name":          spec.name,
        "promo_discount_rate": spec.discount_rate,
        "total_customers":     total,
        "scenario_A": {
            "label":       "ML Only",
            "send_count":  a_send,
            "profit":      a_profit,
        },
        "scenario_B": {
            "label":              "โปรร้านตามแผน",
            "target_count":       b_total_target,
            "send_count":         b_send,
            "blocked_guardrail":  b_blocked,
            "profit":             b_profit,
        },
        "scenario_C": {
            "label":       "Optimal (Promo Advisor)",
            "send_count":  c_n,
            "profit":      c_profit,
            "from_operator": from_op,
            "from_ml":       from_ml,
            "vs_A_profit":   c_vs_a,
            "vs_A_pct":      c_vs_a_pct,
            "vs_B_profit":   c_vs_b,
            "vs_B_pct":      c_vs_b_pct,
        },
        "conflicts_resolved":           conflicts,
        "operator_promos_replaced_by_ml": op_suppressed,
        "operator_promos_replaced_reason": op_suppressed_reason,
        "not_sent_count": total - c_n,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def promo_advisor(
    ml_output: pd.DataFrame,
    cust_agg: pd.DataFrame,
    spec: PromoSpec,
    cogs_rate: float = 0.50,
    min_margin: float = 0.15,
    history_path: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Main entry point.

    Parameters
    ----------
    ml_output    : DataFrame from decide_for_customers() (Layer 4 output)
    cust_agg     : customer-level aggregation DataFrame (from build_serving_context)
    spec         : PromoSpec describing the operator's planned promotion
    cogs_rate    : COGS rate (default 0.50; use data-driven value if available)
    min_margin   : CFO margin floor (default 0.15)
    history_path : path to promo_history.jsonl for logging (optional)

    Returns
    -------
    resolved_df  : per-customer DataFrame with final decisions + reasons
    report       : dict from compare_scenarios()
    """
    promo_eval = evaluate_operator_promo(ml_output, cust_agg, spec,
                                          cogs_rate=cogs_rate, min_margin=min_margin)
    resolved   = resolve_per_customer(ml_output, promo_eval, spec)
    report     = compare_scenarios(resolved, spec)

    # ── Log to promo history ──────────────────────────────────────────────────
    if history_path:
        _log_promo_history(spec, report, history_path)

    return resolved, report


# ── History logging ───────────────────────────────────────────────────────────

def _log_promo_history(spec: PromoSpec, report: dict, path: str) -> None:
    """Append this promo run to promo_history.jsonl for future analysis."""
    entry = {
        "logged_at":  datetime.now().isoformat(),
        "promo_spec": asdict(spec),
        "summary":    report,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Report printer ────────────────────────────────────────────────────────────

def print_promo_report(report: dict, currency: str = "฿") -> None:
    """Pretty-print the promo advisor report to stdout."""
    sep = "═" * 56
    c   = currency

    print(f"\n{sep}")
    print(f"  Promo Advisor Report")
    print(f"  โปร: {report['promo_name']}  (ส่วนลด {report['promo_discount_rate']:.0%})")
    print(f"  ลูกค้าทั้งหมด: {report['total_customers']:,} คน")
    print(sep)

    A = report["scenario_A"]
    B = report["scenario_B"]
    C = report["scenario_C"]

    print(f"\n  Scenario A — {A['label']}")
    print(f"    ส่ง {A['send_count']:,} คน  |  Δprofit คาด: {c}{A['profit']:,.0f}")

    print(f"\n  Scenario B — {B['label']}")
    print(f"    กลุ่มเป้าหมาย : {B['target_count']:,} คน")
    print(f"    ผ่าน guardrail: {B['send_count']:,} คน  |  Δprofit คาด: {c}{B['profit']:,.0f}")
    if B["blocked_guardrail"] > 0:
        print(f"    ถูกบล็อก      : {B['blocked_guardrail']:,} คน  (margin ติดลบ / uplift ต่ำ)")

    print(f"\n  Scenario C — {C['label']}  ◀ แนะนำ")
    print(f"    ส่ง {C['send_count']:,} คน  |  Δprofit คาด: {c}{C['profit']:,.0f}")
    print(f"      ├── ใช้โปรร้าน : {C['from_operator']:,} คน")
    print(f"      └── ใช้ ML     : {C['from_ml']:,} คน")
    if C["vs_A_profit"] is not None:
        sign_a = "+" if C["vs_A_profit"] >= 0 else ""
        pct_a  = f" ({sign_a}{C['vs_A_pct']:.1f}%)" if C["vs_A_pct"] else ""
        print(f"    vs Scenario A : {sign_a}{c}{C['vs_A_profit']:,.0f}{pct_a}")
    if C["vs_B_profit"] is not None:
        sign_b = "+" if C["vs_B_profit"] >= 0 else ""
        pct_b  = f" ({sign_b}{C['vs_B_pct']:.1f}%)" if C["vs_B_pct"] else ""
        print(f"    vs Scenario B : {sign_b}{c}{C['vs_B_profit']:,.0f}{pct_b}")

    print(f"\n  Conflicts (ประเภทเดียวกัน) : {report['conflicts_resolved']:,} คน → เลือกกำไรสูงกว่า")
    if report["operator_promos_replaced_by_ml"] > 0:
        print(f"  โปรร้านถูกแทน ML           : {report['operator_promos_replaced_by_ml']:,} คน "
              f"({report['operator_promos_replaced_reason']})")
    print(f"  ไม่ส่งเลย                  : {report['not_sent_count']:,} คน "
          f"(ทั้งคู่ไม่ผ่าน guardrail)")
    print(f"\n{sep}\n")
    print(f"  หมายเหตุ: Δprofit เป็นค่าประมาณจากโมเดล (model-estimated projection)")
    print(f"  บน observational/proxy labels — ไม่ใช่กำไรรับประกันจากแคมเปญจริง\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Historical Analysis · Cannibalization · Multi-Promo Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Historical Promo Analyzer ─────────────────────────────────────────────

def analyze_promo_history(
    pos_df: pd.DataFrame,
    history_path: str | None = None,
) -> dict:
    """Analyze past promo performance from two sources:

    Source A — POS ``promo_applied`` column (always available if POS has the field)
        Groups customers who received a promo vs those who didn't.
        Computes per-segment response lift = treated_visit_rate - control_visit_rate.

    Source B — promo_history.jsonl written by promo_advisor() calls (accumulates over time)
        Loads logged promo runs and aggregates scenario C profit per promo_type.

    Returns
    -------
    dict with keys:
        "pos_analysis"      : per-segment × promo_type response stats (from POS)
        "history_analysis"  : per-promo_type profit summary (from jsonl log)
        "recommendations"   : list of plain-text suggestions for the operator
        "confidence"        : "high" / "medium" / "low" based on sample size
    """
    results: dict = {
        "pos_analysis": {},
        "history_analysis": {},
        "recommendations": [],
        "confidence": "low",
    }

    # ── Source A: POS promo_applied ───────────────────────────────────────────
    if "promo_applied" in pos_df.columns and "post_promo_visit" in pos_df.columns:
        # Aggregate to customer level first
        cid_col = "customer_id" if "customer_id" in pos_df.columns else pos_df.index.name or "index"
        grp_cols = [cid_col]
        if "cluster_segment_name" in pos_df.columns:
            grp_cols.append("cluster_segment_name")

        cust = (
            pos_df.groupby(grp_cols)
            .agg(
                promo_applied=("promo_applied", "max"),
                converted=("post_promo_visit", "max"),
            )
            .reset_index()
        )

        n_total = len(cust)
        n_treated = int((cust["promo_applied"] == 1).sum())
        n_control = n_total - n_treated

        if n_treated >= 10 and n_control >= 10:
            treat_rate = float(cust[cust["promo_applied"] == 1]["converted"].mean())
            ctrl_rate  = float(cust[cust["promo_applied"] == 0]["converted"].mean())
            lift       = round(treat_rate - ctrl_rate, 4)

            results["pos_analysis"]["overall"] = {
                "n_treated":    n_treated,
                "n_control":    n_control,
                "treat_conv":   round(treat_rate, 4),
                "ctrl_conv":    round(ctrl_rate, 4),
                "lift":         lift,
            }

            if "cluster_segment_name" in cust.columns:
                seg_stats = {}
                for seg, grp in cust.groupby("cluster_segment_name"):
                    t = grp[grp["promo_applied"] == 1]
                    c = grp[grp["promo_applied"] == 0]
                    if len(t) >= 5 and len(c) >= 5:
                        seg_lift = round(float(t["converted"].mean()) - float(c["converted"].mean()), 4)
                        seg_stats[str(seg)] = {
                            "n_treated": len(t),
                            "n_control": len(c),
                            "lift":      seg_lift,
                            "treat_conv": round(float(t["converted"].mean()), 4),
                            "ctrl_conv":  round(float(c["converted"].mean()), 4),
                        }
                results["pos_analysis"]["by_segment"] = seg_stats

                # Recommendations from segment lifts
                if seg_stats:
                    best_seg  = max(seg_stats, key=lambda s: seg_stats[s]["lift"])
                    worst_seg = min(seg_stats, key=lambda s: seg_stats[s]["lift"])
                    best_lift = seg_stats[best_seg]["lift"]
                    worst_lift = seg_stats[worst_seg]["lift"]
                    results["recommendations"].append(
                        f"Segment ที่ตอบสนองดีที่สุด: '{best_seg}' (lift={best_lift:+.1%}) "
                        f"→ เน้นโปรประเภทนี้กับ {best_seg}"
                    )
                    if worst_lift <= 0:
                        results["recommendations"].append(
                            f"Segment '{worst_seg}' มี lift={worst_lift:+.1%} ≤ 0 "
                            f"→ ไม่ควรส่งโปรประเภทนี้ให้กลุ่มนี้"
                        )

            results["confidence"] = "medium" if n_treated >= 50 else "low"

    # ── Source B: promo_history.jsonl ─────────────────────────────────────────
    if history_path and os.path.exists(history_path):
        history_entries: list[dict] = []
        try:
            with open(history_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        history_entries.append(json.loads(line))
        except Exception:
            pass

        if history_entries:
            # Aggregate profit by promo_type
            type_profit: dict[str, list[float]] = {}
            type_count:  dict[str, int]  = {}
            for entry in history_entries:
                spec_dict = entry.get("promo_spec", {})
                p_type    = spec_dict.get("promo_type", "unknown")
                scenario_c = entry.get("summary", {}).get("scenario_C", {})
                profit     = scenario_c.get("profit", None)
                if profit is not None:
                    type_profit.setdefault(p_type, []).append(float(profit))
                    type_count[p_type]  = type_count.get(p_type, 0) + 1

            history_summary = {}
            for p_type, profits in type_profit.items():
                history_summary[p_type] = {
                    "runs":        type_count[p_type],
                    "avg_profit":  round(float(np.mean(profits)), 2),
                    "total_profit": round(float(np.sum(profits)), 2),
                    "best_profit": round(float(np.max(profits)), 2),
                }
            results["history_analysis"] = history_summary

            if history_summary:
                best_type = max(history_summary, key=lambda t: history_summary[t]["avg_profit"])
                results["recommendations"].append(
                    f"ประเภทโปรที่ให้กำไรเฉลี่ยสูงสุดจากประวัติ: '{best_type}' "
                    f"(เฉลี่ย ฿{history_summary[best_type]['avg_profit']:,.0f} / run)"
                )
                if len(history_entries) >= 5:
                    results["confidence"] = "high"
                elif len(history_entries) >= 2:
                    results["confidence"] = max(results["confidence"],
                                                "medium",
                                                key=lambda x: ["low","medium","high"].index(x))

    if not results["recommendations"]:
        results["recommendations"].append(
            "ข้อมูลประวัติโปรยังน้อย — ผลจะแม่นขึ้นหลังรัน Promo Advisor ไปอีก 2–3 ครั้ง"
        )

    return results


def print_history_report(analysis: dict, currency: str = "฿") -> None:
    """Pretty-print the historical analysis to stdout."""
    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  Historical Promo Analysis  (confidence: {analysis['confidence']})")
    print(sep)

    pos = analysis.get("pos_analysis", {})
    if pos.get("overall"):
        ov = pos["overall"]
        print(f"\n  POS Data — Overall (promo_applied)")
        print(f"    Treated : {ov['n_treated']:,} คน  conv={ov['treat_conv']:.1%}")
        print(f"    Control : {ov['n_control']:,} คน  conv={ov['ctrl_conv']:.1%}")
        print(f"    Lift    : {ov['lift']:+.1%}")
        by_seg = pos.get("by_segment", {})
        if by_seg:
            print(f"\n  By Segment:")
            for seg, s in sorted(by_seg.items(), key=lambda x: -x[1]["lift"]):
                bar = "▓" * max(0, int(s["lift"] * 100)) if s["lift"] > 0 else "░"
                print(f"    {seg:<22} lift={s['lift']:+.1%}  {bar}")

    hist = analysis.get("history_analysis", {})
    if hist:
        print(f"\n  Promo History Log — Avg Profit per Type:")
        for p_type, h in sorted(hist.items(), key=lambda x: -x[1]["avg_profit"]):
            print(f"    {p_type:<20} {currency}{h['avg_profit']:>8,.0f}/run  "
                  f"({h['runs']} runs, best={currency}{h['best_profit']:,.0f})")

    recs = analysis.get("recommendations", [])
    if recs:
        print(f"\n  คำแนะนำ:")
        for r in recs:
            print(f"    • {r}")
    print(f"\n{sep}\n")


# ── 2. Cannibalization Risk Score ─────────────────────────────────────────────

def add_cannibalization_risk(
    df: pd.DataFrame,
    baseline_col: str = "prob_control",
    high_baseline_threshold: float = 0.65,
    cannib_discount: float = 0.40,
) -> pd.DataFrame:
    """Flag customers likely to purchase regardless of the promo (cannibalization risk).

    A customer with high baseline conversion probability (prob_control ≥ threshold)
    is likely to buy anyway — the promo just gives them a discount they don't need
    to trigger a visit. This discounts their expected incremental profit.

    New columns:
        cannibalization_risk   : bool — True if prob_control >= threshold
        adjusted_profit        : expected_profit after cannibalization discount
        cannib_discount_applied: float — fraction discounted (0 if no risk)
    """
    df = df.copy()
    baseline = df[baseline_col].fillna(0.0) if baseline_col in df.columns else pd.Series(0.0, index=df.index)
    df["cannibalization_risk"] = baseline >= high_baseline_threshold

    if "operator_promo_profit" in df.columns:
        profit_col = "operator_promo_profit"
    elif "expected_profit" in df.columns:
        profit_col = "expected_profit"
    else:
        df["adjusted_profit"] = 0.0
        df["cannib_discount_applied"] = 0.0
        return df

    df["adjusted_profit"] = df.apply(
        lambda row: (
            round(float(row[profit_col]) * (1.0 - cannib_discount), 2)
            if row["cannibalization_risk"] and float(row[profit_col]) > 0
            else float(row[profit_col])
        ),
        axis=1,
    )
    df["cannib_discount_applied"] = df["cannibalization_risk"].apply(
        lambda r: cannib_discount if r else 0.0
    )
    return df


# ── 3. Multi-Promo Optimizer ──────────────────────────────────────────────────

def multi_promo_advisor(
    ml_output: pd.DataFrame,
    cust_agg: pd.DataFrame,
    promo_specs: list[PromoSpec],
    cogs_rate: float = 0.50,
    min_margin: float = 0.15,
    max_promos_per_customer: int = 1,
    apply_cannibalization: bool = True,
    history_path: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Optimize across multiple operator promos — each customer gets at most
    ``max_promos_per_customer`` offers (default 1) to prevent message spam.

    Algorithm:
      1. Evaluate each PromoSpec independently (CFO guardrail per customer).
      2. Apply cannibalization risk discount to high-baseline customers.
      3. For each customer, rank all viable promos + ML recommendation by
         adjusted profit. Assign the top-N (N = max_promos_per_customer).
      4. Report per-promo send counts + overall Scenario A/B_total/C.

    Parameters
    ----------
    promo_specs              : list of PromoSpec (operator's planned promos this month)
    max_promos_per_customer  : max number of different offers per customer (default 1)
    apply_cannibalization    : discount incremental profit for high-baseline customers
    history_path             : optional path to promo_history.jsonl for logging

    Returns
    -------
    resolved_df : per-customer DataFrame with assigned promo + reason
    report      : multi-promo summary dict
    """
    if not promo_specs:
        raise ValueError("promo_specs must contain at least one PromoSpec")

    n_cust = len(cust_agg)

    # ── Step 1: evaluate each promo independently ─────────────────────────────
    all_evals: list[pd.DataFrame] = []
    for spec in promo_specs:
        ev = evaluate_operator_promo(ml_output, cust_agg, spec,
                                     cogs_rate=cogs_rate, min_margin=min_margin)
        ev["_promo_name"] = spec.name
        ev["_promo_type"] = spec.promo_type
        ev["_promo_idx"]  = promo_specs.index(spec)
        if apply_cannibalization:
            ev = add_cannibalization_risk(ev, cannib_discount=0.40)
            profit_col = "adjusted_profit"
        else:
            ev["adjusted_profit"] = ev["operator_promo_profit"]
            profit_col = "adjusted_profit"
        all_evals.append(ev[["customer_id", "_promo_name", "_promo_type",
                               "_promo_idx", "promo_viable", profit_col,
                               "operator_promo_cost",
                               "cannibalization_risk" if apply_cannibalization else "promo_viable",
                               "operator_promo_blocked_reason"]])

    # ── Step 2: ML output (counts as one option per customer) ─────────────────
    ml_opts = ml_output[["customer_id", "recommended_action",
                           "expected_profit", "promo_cost", "should_send"]].copy()
    ml_opts = ml_opts.rename(columns={
        "recommended_action": "_promo_name",
        "expected_profit":    "adjusted_profit",
        "promo_cost":         "operator_promo_cost",
    })
    ml_opts["_promo_type"]  = "ml"
    ml_opts["_promo_idx"]   = -1
    ml_opts["promo_viable"] = ml_opts["should_send"] == 1
    ml_opts["operator_promo_blocked_reason"] = ""
    if apply_cannibalization:
        ml_opts["cannibalization_risk"] = False  # ML already accounts for uplift

    all_options = pd.concat(
        all_evals + [ml_opts[all_evals[0].columns]],
        ignore_index=True,
    )

    # ── Step 3: Per-customer assignment ───────────────────────────────────────
    assignments: list[dict] = []
    for cid, grp in all_options.groupby("customer_id"):
        viable = grp[grp["promo_viable"] == True].sort_values(
            "adjusted_profit", ascending=False
        )
        assigned = viable.head(max_promos_per_customer)

        if len(assigned) == 0:
            assignments.append({
                "customer_id":          cid,
                "assigned_promos":      [],
                "assigned_sources":     [],
                "total_assigned_profit": 0.0,
                "total_assigned_cost":   0.0,
                "final_should_send":    0,
                "multi_advisor_reason": "ไม่มีโปรหรือ ML ที่ผ่าน guardrail",
            })
        else:
            promos   = assigned["_promo_name"].tolist()
            sources  = assigned["_promo_type"].tolist()
            profits  = round(float(assigned["adjusted_profit"].sum()), 2)
            costs    = round(float(assigned["operator_promo_cost"].sum()), 2)
            if len(promos) == 1:
                reason = (f"เลือก '{promos[0]}' "
                          f"(กำไรสูงสุดจาก {len(grp[grp['promo_viable']==True])} ตัวเลือก)")
            else:
                reason = f"ส่ง {len(promos)} โปร: {', '.join(promos)}"
            assignments.append({
                "customer_id":           cid,
                "assigned_promos":       promos,
                "assigned_sources":      sources,
                "total_assigned_profit": profits,
                "total_assigned_cost":   costs,
                "final_should_send":     1,
                "multi_advisor_reason":  reason,
            })

    assign_df = pd.DataFrame(assignments)
    # Flatten for broadcast list (take first assigned promo as "final_action")
    assign_df["final_action"] = assign_df["assigned_promos"].apply(
        lambda x: x[0] if x else "Do Nothing (No Viable Promo)"
    )
    assign_df["final_source"] = assign_df["assigned_sources"].apply(
        lambda x: x[0] if x else "none"
    )

    resolved = cust_agg.merge(assign_df, on="customer_id", how="left")

    # ── Step 4: Build report ───────────────────────────────────────────────────
    total_send   = int((assign_df["final_should_send"] == 1).sum())
    total_profit = round(float(assign_df["total_assigned_profit"].sum()), 2)

    # Scenario A: ML only
    ml_send   = ml_output[ml_output["should_send"] == 1]
    a_profit  = round(float(ml_send["expected_profit"].sum()), 2)
    a_send    = int(len(ml_send))

    # Scenario B: all operator promos broadcast (guardrail applied, sum across promos)
    b_total_send = 0
    b_total_profit = 0.0
    per_promo_summary = {}
    for i, spec in enumerate(promo_specs):
        ev = all_evals[i]
        viable = ev[ev["promo_viable"] == True]
        b_total_send   += len(viable)
        b_total_profit += float(viable["adjusted_profit"].sum())
        per_promo_summary[spec.name] = {
            "target":    int((ev["promo_viable"] | True).sum()),
            "viable":    len(viable),
            "profit":    round(float(viable["adjusted_profit"].sum()), 2),
        }

    # Dedup stats
    n_from_op = int((assign_df["final_source"] != "ml").sum())
    n_from_ml = int((assign_df["final_source"] == "ml").sum())
    n_conflict = int(all_options.groupby("customer_id").apply(
        lambda g: (g["promo_viable"].sum() > 1)
    ).sum())

    # Log each spec
    if history_path:
        for spec in promo_specs:
            _log_promo_history(spec, {
                "multi_promo": True,
                "total_send": total_send,
                "total_profit": total_profit,
            }, history_path)

    report = {
        "n_promos":        len(promo_specs),
        "total_customers": n_cust,
        "scenario_A": {"label": "ML Only",          "send": a_send,       "profit": a_profit},
        "scenario_B": {"label": "All Operator Promos (guardrail applied, may double-send)",
                       "send": b_total_send, "profit": round(b_total_profit, 2)},
        "scenario_C": {
            "label":        "Multi-Promo Optimal (max 1 per customer)",
            "send":         total_send,
            "profit":       total_profit,
            "from_operator": n_from_op,
            "from_ml":       n_from_ml,
            "vs_A_profit":  round(total_profit - a_profit, 2),
            "vs_B_profit":  round(total_profit - b_total_profit, 2),
        },
        "per_promo":               per_promo_summary,
        "customers_with_multiple_viable": n_conflict,
        "max_promos_per_customer": max_promos_per_customer,
        "cannibalization_applied": apply_cannibalization,
    }
    return resolved, report


def print_multi_promo_report(report: dict, currency: str = "฿") -> None:
    """Pretty-print multi-promo optimizer report."""
    sep = "═" * 56
    c   = currency
    print(f"\n{sep}")
    print(f"  Multi-Promo Advisor Report  ({report['n_promos']} โปร)")
    print(f"  ลูกค้าทั้งหมด: {report['total_customers']:,} คน")
    print(sep)

    A = report["scenario_A"]
    B = report["scenario_B"]
    C = report["scenario_C"]

    print(f"\n  Scenario A — {A['label']}")
    print(f"    ส่ง {A['send']:,} คน  |  Δprofit: {c}{A['profit']:,.0f}")

    print(f"\n  Scenario B — {B['label']}")
    print(f"    ส่ง {B['send']:,} ครั้ง (อาจส่งซ้ำคนเดียวกัน)  |  Δprofit: {c}{B['profit']:,.0f}")

    print(f"\n  Scenario C — {C['label']}  ◀ แนะนำ")
    print(f"    ส่ง {C['send']:,} คน  |  Δprofit: {c}{C['profit']:,.0f}")
    print(f"      ├── ใช้โปรร้าน: {C['from_operator']:,} คน")
    print(f"      └── ใช้ ML    : {C['from_ml']:,} คน")
    sign_a = "+" if C["vs_A_profit"] >= 0 else ""
    sign_b = "+" if C["vs_B_profit"] >= 0 else ""
    print(f"    vs A: {sign_a}{c}{C['vs_A_profit']:,.0f}  |  vs B: {sign_b}{c}{C['vs_B_profit']:,.0f}")

    if report.get("per_promo"):
        print(f"\n  Per-Promo Breakdown:")
        for pname, ps in report["per_promo"].items():
            print(f"    {pname[:40]:<40}  viable={ps['viable']:,}  "
                  f"profit={c}{ps['profit']:,.0f}")

    print(f"\n  ลูกค้าที่มีหลายโปรให้เลือก: {report['customers_with_multiple_viable']:,} คน "
          f"(เลือกกำไรสูงสุด)")
    if report.get("cannibalization_applied"):
        print(f"  ✓ Cannibalization discount ถูก apply แล้ว (baseline conv ≥ 65% → ลด profit 40%)")
    print(f"\n{sep}\n")
