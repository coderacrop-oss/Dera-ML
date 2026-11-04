"""Promo Advisor — Phase 1

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
