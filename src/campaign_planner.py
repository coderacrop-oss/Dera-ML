"""Campaign Planner — single-entry-point wrapper for the ML Campaign Engine.

Accepts either a budget ceiling OR a target profit, plus an optional goal,
and returns a structured summary report with expected spend, profit, and ROI.

Usage
-----
    from campaign_planner import run as plan_campaign

    # Mode 1: budget-first — "มีงบ ฿3,000 ทำอะไรได้บ้าง?"
    report = plan_campaign(budget=3000, goal="win_back")

    # Mode 2: target-first — "อยากได้ profit ฿10,000 ต้องใช้งบเท่าไหร่?"
    report = plan_campaign(target_profit=10000)
"""

from __future__ import annotations

import json
import os
from typing import Literal

import pandas as pd

from budget_optimizer import select_within_budget
from pipeline_logger import log
from uplift_engine import REGISTRY_PATH, _check_features, build_serving_context, decide_for_customers
from model_registry import ModelRegistry, RegistryError
from ingest import detect_source, ingest_meripos, quality_gate
from store_intelligence import build_store_profile, get_store_embedding
from bundle_pairs import group_bundle, personal_pairs
from business_config import CONFIG
from action_rules import SURE_THING_RULES, PERSUADABLE_RULES, get_dotdash_offer, infer_dotdash_offer


# ---------------------------------------------------------------------------
# Goal definitions
# ---------------------------------------------------------------------------

GOAL = Literal["maximize_profit", "win_back", "retain_loyal", "maximize_reach"]

# Segment keywords to include per goal (partial, case-insensitive match).
# None = all viable segments.
_GOAL_SEGMENTS: dict[str, list[str] | None] = {
    "maximize_profit": None,
    "win_back":        ["at-risk", "hibernating", "can't lose"],
    "retain_loyal":    ["champion", "loyal", "potential loyalist", "need attention"],
    "maximize_reach":  None,   # all segments, sort by cheapest first
}

_GOAL_LABELS: dict[str, str] = {
    "maximize_profit": "กำไรสูงสุด (default)",
    "win_back":        "ดึงคนเก่ากลับ (At-Risk + Hibernating)",
    "retain_loyal":    "รักษาลูกค้าดี (Champions + Loyal)",
    "maximize_reach":  "ส่งให้ได้คนมากที่สุดในงบ",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_pipeline(
    pos_file: str,
    promo_response: str | None,
) -> tuple[pd.DataFrame, object]:
    """Run L0-L4 without budget filter. Returns (detail_df, ctx)."""
    ctx = build_serving_context(pos_file, promo_response)

    # Phase 1: build store-level intelligence from anonymous data
    try:
        raw_df = pd.read_csv(pos_file, encoding="utf-8", encoding_errors="replace", nrows=5)
        source = detect_source(raw_df)
        gate   = quality_gate(pd.read_csv(pos_file, encoding="utf-8", encoding_errors="replace"))
        ctx.quality_gate_result = gate
        if source == "meripos":
            tx_df, item_df = ingest_meripos(pos_file)
            ctx.store_profile = build_store_profile(tx_df, item_df)
        # AC-4.2: compute store embedding when profile is available
        if ctx.store_profile is not None:
            ctx.store_embedding = get_store_embedding(ctx.store_profile)
    except Exception:
        pass  # store_profile / store_embedding are optional — never block the main pipeline

    registry = ModelRegistry(os.path.dirname(REGISTRY_PATH))
    try:
        champion, _ref = registry.load_current()
        _check_features(champion, ctx.features)
        _, detail = decide_for_customers(champion, ctx)
    except RegistryError:
        # Graceful fallback: rule-based RFM scoring
        from campaign_pipeline import _rule_based_score
        detail = _rule_based_score(ctx)

    # Attach human-readable segment name
    # Priority: (1) cluster_segment_map from ctx, (2) cluster_segment column, (3) IBM name from ctx segments list
    cluster_col = "cluster" if "cluster" in detail.columns else "kmeans_cluster"
    detail = detail.copy()

    if ctx.cluster_segment_map and cluster_col in detail.columns:
        detail["segment_name"] = detail[cluster_col].map(ctx.cluster_segment_map).fillna("")
    elif "cluster_segment" in detail.columns:
        detail["segment_name"] = detail["cluster_segment"]
    else:
        detail["segment_name"] = ""

    # Fill remaining empty segment names using IBM mapping from rfm_df cluster stats
    if detail["segment_name"].eq("").any() and cluster_col in detail.columns:
        from campaign_pipeline import _ibm_segment_name
        import numpy as _np
        rfm = ctx.rfm_df
        if "kmeans_cluster" in rfm.columns:
            stats = rfm.groupby("kmeans_cluster").agg(
                recency=("recency", "mean"),
                frequency=("frequency", "mean"),
                monetary=("monetary", "mean"),
            ).reset_index()
            if len(stats) > 1:
                stats["_r_n"] = 1 - stats["recency"].rank(pct=True, ascending=True)
                stats["_f_n"] = stats["frequency"].rank(pct=True, ascending=True)
                stats["_m_n"] = stats["monetary"].rank(pct=True, ascending=True)
            else:
                stats["_r_n"] = stats["_f_n"] = stats["_m_n"] = 0.5
            seg_map = {
                int(row["kmeans_cluster"]): _ibm_segment_name(row["_r_n"], row["_f_n"], row["_m_n"])
                for _, row in stats.iterrows()
            }
            mask = detail["segment_name"].eq("")
            detail.loc[mask, "segment_name"] = detail.loc[mask, cluster_col].map(seg_map).fillna("Unknown")

    return detail, ctx


def _apply_goal_filter(detail: pd.DataFrame, goal: str) -> pd.DataFrame:
    """Return only CFO-approved customers matching the goal's segment list."""
    approved = detail[detail["should_send"] == 1].copy()
    target_segs = _GOAL_SEGMENTS.get(goal)

    if target_segs and "segment_name" in approved.columns:
        mask = approved["segment_name"].apply(
            lambda s: any(t in str(s).lower() for t in target_segs)
        )
        approved = approved[mask]

    # maximize_reach: cheapest customers first to maximise headcount per baht
    if goal == "maximize_reach":
        approved = approved.sort_values("promo_cost", ascending=True).reset_index(drop=True)
        # Override bang_per_buck logic: treat cost-per-customer as the ranking key
        # by injecting a synthetic bang_per_buck = 1 / promo_cost so select_within_budget
        # picks the cheapest ones first.
        approved["_reach_bpb"] = 1.0 / approved["promo_cost"].clip(lower=0.01)
        # We'll use the standard optimizer but pass a patched frame; restore after
        approved["_orig_profit"] = approved["expected_profit"]
        approved["expected_profit"] = approved["_reach_bpb"]

    return approved


def _restore_reach_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Undo the reach-mode profit substitution."""
    if "_orig_profit" in df.columns:
        df = df.copy()
        df["expected_profit"] = df["_orig_profit"]
        df = df.drop(columns=["_reach_bpb", "_orig_profit"], errors="ignore")
    return df


def _find_budget_for_target(approved: pd.DataFrame, target_profit: float) -> float:
    """Binary search for minimum budget that achieves target_profit.

    Returns the minimum budget required, or the full cost if target is unreachable.
    """
    max_profit = float(approved["expected_profit"].sum())
    if max_profit <= target_profit:
        # Target exceeds what's achievable — return cost of full send
        return round(float(approved["promo_cost"].sum()), 2)

    lo, hi = 0.0, float(approved["promo_cost"].sum())
    for _ in range(40):   # 40 iterations → sub-baht precision
        mid = (lo + hi) / 2.0
        result = select_within_budget(approved, budget=mid)
        achieved = float(result.loc[result["selected_for_budget"], "expected_profit"].sum())
        if achieved >= target_profit:
            hi = mid
        else:
            lo = mid
    return round(hi + 1, 2)   # +1 baht buffer for rounding


# ---------------------------------------------------------------------------
# DotDash offer helpers
# ---------------------------------------------------------------------------

def _infer_dotdash_offer(action: str) -> dict:
    """Deprecated text-only alias of action_rules.infer_dotdash_offer (kept one release)."""
    return infer_dotdash_offer(None, action)


def _with_bundle(offer: dict, grp: pd.DataFrame, pairs: dict | None) -> dict:
    """Add the group's bundle suggestion (always the same three fields, so the schema is stable)."""
    ids = grp["customer_id"].astype(str).tolist() if "customer_id" in grp.columns else []
    items, coverage = group_bundle(ids, pairs or {})
    return {**offer, "bundle_items": items, "bundle_coverage": coverage,
            "bundle_low_coverage": coverage < CONFIG["bundle_low_coverage_threshold"]}


def _build_dotdash_segments(selected: pd.DataFrame, pairs: dict | None = None) -> list[dict]:
    """Return DotDash draft-campaign specs: ONE GROUP PER (segment, action_key).

    DotDash creates one campaign per group with one reward, so a segment whose customers were
    given different actions is split — every customer is in the group whose offer they were given.
    Each entry: segment_name, action_key, recommended_action (most common text in the group),
    customer_count, customer_ids, expected_profit, actual_spend, dotdash_offer (incl. action_key).
    Rows without an action_key (data from before it existed) are grouped by action text and their
    offer is inferred from that text (legacy, logged).
    ``pairs`` (customer_id -> personal product pair, see bundle_pairs) adds bundle_items,
    bundle_coverage and bundle_low_coverage to each dotdash_offer, computed per group.
    """
    if selected.empty or "segment_name" not in selected.columns:
        return []

    df = selected.copy()
    if "action_key" not in df.columns:
        df["action_key"] = None
    if "recommended_action" not in df.columns:
        df["recommended_action"] = ""
    df["action_key"] = df["action_key"].astype(object).where(df["action_key"].notna(), None)
    # keyless rows group by their action text so they are still split by what they were given
    df["_group"] = [k if k is not None else f"text:{t}" for k, t in zip(df["action_key"], df["recommended_action"])]

    result: list[dict] = []
    for (seg_name, _g), grp in df.groupby(["segment_name", "_group"], sort=False):
        key = grp["action_key"].iloc[0]
        top_action = str(grp["recommended_action"].value_counts().index[0])
        result.append({
            "segment_name":       str(seg_name),
            "action_key":         key,
            "recommended_action": top_action,
            "customer_count":     len(grp),
            "customer_ids":       grp["customer_id"].tolist() if "customer_id" in grp.columns else [],
            "expected_profit":    round(float(grp["expected_profit"].sum()), 2),
            "actual_spend":       round(float(grp["promo_cost"].sum()), 2),
            "dotdash_offer":      _with_bundle(infer_dotdash_offer(key, top_action), grp, pairs),
        })

    # Most valuable group first
    result.sort(key=lambda x: x["expected_profit"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    *,
    budget: float | None = None,
    target_profit: float | None = None,
    goal: GOAL = "maximize_profit",
    pos_file: str = "data/transactions.csv",
    promo_response: str | None = None,
) -> dict:
    """Plan a campaign and return a structured summary report.

    Parameters
    ----------
    budget : float, optional
        Spend ceiling in baht.  Mutually exclusive with target_profit.
    target_profit : float, optional
        Desired incremental profit.  System finds the minimum budget needed.
    goal : str
        One of: maximize_profit | win_back | retain_loyal | maximize_reach.
        Default: maximize_profit.
    pos_file : str
        Path to POS CSV (default: data/transactions.csv).
    promo_response : str, optional
        Path to labels CSV containing is_treated and post_promo_visit columns.

    Returns
    -------
    dict  — structured campaign plan (also pretty-printed to stdout).
    """
    # ── Input validation ────────────────────────────────────────────────────
    if budget is None and target_profit is None:
        raise ValueError("ต้องระบุ budget หรือ target_profit อย่างใดอย่างหนึ่ง")
    if budget is not None and target_profit is not None:
        raise ValueError("ระบุได้แค่อย่างเดียว: budget หรือ target_profit ไม่ใช่ทั้งคู่")
    if goal not in _GOAL_SEGMENTS:
        raise ValueError(f"goal ต้องเป็นหนึ่งใน: {list(_GOAL_SEGMENTS.keys())}")

    # ── Step 1: Run pipeline ────────────────────────────────────────────────
    detail, _ctx = _run_pipeline(pos_file, promo_response)

    total_customers  = len(detail)
    total_viable     = int((detail["should_send"] == 1).sum())
    total_suppressed = int((detail["should_send"] == 0).sum())

    # ── Step 2: Goal filter ─────────────────────────────────────────────────
    approved = _apply_goal_filter(detail, goal)
    goal_filtered_count = len(approved)

    max_achievable_profit = round(float(approved["_orig_profit"].sum()
                                        if "_orig_profit" in approved.columns
                                        else approved["expected_profit"].sum()), 2)
    max_budget_needed = round(float(approved["promo_cost"].sum()), 2)

    # ── Step 3: Resolve budget ──────────────────────────────────────────────
    if target_profit is not None:
        effective_budget = _find_budget_for_target(approved, target_profit)
        mode = "target_profit"
    else:
        effective_budget = float(budget)
        mode = "budget"

    # ── Step 4: Budget optimizer ────────────────────────────────────────────
    if effective_budget > 0:
        optimized = select_within_budget(approved, budget=effective_budget)
        selected  = optimized[optimized["selected_for_budget"] == True].copy()
    else:
        selected = approved.copy()
        selected["selected_for_budget"] = True

    # Restore patched columns for maximize_reach
    selected  = _restore_reach_columns(selected)
    approved  = _restore_reach_columns(approved)

    # ── Step 5: Build report ────────────────────────────────────────────────
    actual_spend  = round(float(selected["promo_cost"].sum()), 2)
    actual_profit = round(float(selected["expected_profit"].sum()), 2)
    roi = round(actual_profit / actual_spend, 4) if actual_spend > 0 else None

    action_breakdown: dict[str, int] = {}
    if not selected.empty and "recommended_action" in selected.columns:
        action_breakdown = selected["recommended_action"].value_counts().to_dict()

    segment_breakdown: dict[str, int] = {}
    if not selected.empty and "segment_name" in selected.columns:
        segment_breakdown = selected["segment_name"].value_counts().to_dict()

    dotdash_segments = _build_dotdash_segments(selected, personal_pairs(_ctx.raw_df))

    report = {
        # Meta
        "mode":             mode,
        "goal":             goal,
        "goal_label":       _GOAL_LABELS[goal],
        # Funnel
        "total_customers":    total_customers,
        "viable_after_guardrail": total_viable,
        "suppressed_by_guardrail": total_suppressed,
        "goal_filtered_to":   goal_filtered_count,
        # Input
        "budget_input":        budget,
        "target_profit_input": target_profit,
        "effective_budget":    effective_budget,
        # Ceiling — what's achievable with no budget limit
        "max_achievable_profit": max_achievable_profit,
        "max_budget_needed":     max_budget_needed,
        # Results
        "customers_selected":    len(selected),
        "actual_spend":          actual_spend,
        "expected_incremental_profit": actual_profit,
        "roi":                   roi,
        "budget_utilisation_pct": (
            round(actual_spend / effective_budget * 100, 1)
            if effective_budget > 0 else None
        ),
        # Breakdowns
        "action_breakdown":  action_breakdown,
        "segment_breakdown": segment_breakdown,
        # DotDash: per-segment draft campaign specs (sorted by expected_profit desc)
        "dotdash_segments":  dotdash_segments,
        # Feasibility (target_profit mode only)
        "target_achievable": (actual_profit >= target_profit) if target_profit else None,
        "profit_gap":        round(target_profit - actual_profit, 2) if target_profit else None,
    }

    _print_report(report)
    log("planner", "campaign_plan",
        **{k: v for k, v in report.items() if not isinstance(v, dict)})

    return report


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_report(r: dict) -> None:
    W = 58
    print("\n" + "=" * W)
    print(f"  DERA Campaign Planner  |  {r['goal_label']}")
    print("=" * W)

    if r["mode"] == "budget":
        print(f"  งบที่กำหนด:              ฿{r['effective_budget']:>10,.0f}")
    else:
        print(f"  เป้า incremental profit: ฿{r['target_profit_input']:>10,.0f}")
        status = "✅ ถึงได้" if r["target_achievable"] else "⚠️  ถึงไม่ได้"
        print(f"  งบที่ต้องใช้:            ฿{r['effective_budget']:>10,.0f}  ({status})")

    print("-" * W)
    print(f"  ลูกค้าทั้งหมด:           {r['total_customers']:>10,} คน")
    print(f"  ผ่าน CFO guardrail:      {r['viable_after_guardrail']:>10,} คน")
    if r["goal"] not in ("maximize_profit", "maximize_reach"):
        print(f"  หลัง goal filter:        {r['goal_filtered_to']:>10,} คน")
    print(f"  ส่งโปรจริง:              {r['customers_selected']:>10,} คน")
    print("-" * W)
    print(f"  ใช้งบจริง:               ฿{r['actual_spend']:>10,.2f}")
    print(f"  คาด incremental profit:  ฿{r['expected_incremental_profit']:>10,.2f}")
    if r["roi"] is not None:
        print(f"  ROI:                     {r['roi']:>10.2f}x")
    if r["budget_utilisation_pct"] is not None:
        print(f"  Budget utilisation:      {r['budget_utilisation_pct']:>9.1f}%")

    if r["mode"] == "target_profit" and not r["target_achievable"]:
        print(f"\n  ⚠️  ขาดอีก ฿{abs(r['profit_gap'] or 0):,.2f}  "
              f"(max achievable: ฿{r['max_achievable_profit']:,.2f})")

    if r["action_breakdown"]:
        print("\n  Actions:")
        for action, cnt in sorted(r["action_breakdown"].items(), key=lambda x: -x[1]):
            print(f"    {action:<44} {cnt:>5,} คน")

    if r["segment_breakdown"]:
        print("\n  Segments:")
        for seg, cnt in sorted(r["segment_breakdown"].items(), key=lambda x: -x[1]):
            print(f"    {seg:<44} {cnt:>5,} คน")

    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DERA Campaign Planner — plan a campaign by budget or target profit"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--budget",        type=float, help="Spend ceiling in baht")
    group.add_argument("--target-profit", type=float, dest="target_profit",
                       help="Desired incremental profit in baht")
    parser.add_argument("--goal", default="maximize_profit",
                        choices=list(_GOAL_SEGMENTS.keys()),
                        help="Campaign goal (default: maximize_profit)")
    parser.add_argument("--pos-file", default="data/transactions.csv",
                        dest="pos_file", help="Path to POS CSV")
    parser.add_argument("--promo-response", default=None, dest="promo_response",
                        help="Path to labels CSV (is_treated, post_promo_visit)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Also dump report as JSON to stdout after table")

    args = parser.parse_args()
    report = run(
        budget=args.budget,
        target_profit=args.target_profit,
        goal=args.goal,
        pos_file=args.pos_file,
        promo_response=args.promo_response,
    )

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
