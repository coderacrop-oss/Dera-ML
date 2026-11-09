"""Agency Reports — 3 tools for marketing agency use cases.

1. audience_export()       — customer_id lists per segment for ad platform upload
2. post_campaign_report()  — treated vs control lift per segment
3. monthly_trend_report()  — segment migration matrix + next campaign proposal

Usage
-----
    from agency_reports import audience_export, post_campaign_report, monthly_trend_report

    # 1. Before campaign — export audiences
    exports = audience_export(layer4_df, segment_map, output_dir="data/agency/audiences")

    # 2. After campaign — measure lift
    report = post_campaign_report(campaign_df, pre_pos_df, post_pos_df)

    # 3. Monthly — trend + next proposal
    monthly = monthly_trend_report(current_df, previous_df, campaign_planner_report)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from pipeline_logger import log


# ═══════════════════════════════════════════════════════
# 1. AUDIENCE EXPORT
# ═══════════════════════════════════════════════════════

def audience_export(
    layer4_df: pd.DataFrame,
    segment_map: dict[int, str] | None = None,
    output_dir: str = "data/agency/audiences",
    platform: Literal["meta", "google", "generic"] = "generic",
    min_segment_size: int = 20,
) -> dict:
    """Export hashed customer_id lists per segment for ad platform upload.

    Parameters
    ----------
    layer4_df     : Layer 4 output from decide_for_customers() — must have
                    customer_id, should_send, cluster (or segment_name),
                    expected_profit, recommended_action
    segment_map   : {cluster_id: segment_name} — from ctx.cluster_segment_map
    output_dir    : where to write per-segment CSV files
    platform      : "meta" | "google" | "generic" — affects column naming
    min_segment_size : skip segments with fewer customers than this

    Returns
    -------
    dict — export summary with file paths and segment stats
    """
    os.makedirs(output_dir, exist_ok=True)
    df = layer4_df.copy()
    df["customer_id"] = df["customer_id"].astype(str)

    # Resolve segment name column
    if "segment_name" not in df.columns:
        if segment_map and "cluster" in df.columns:
            df["segment_name"] = df["cluster"].map(segment_map).fillna("Unknown")
        else:
            df["segment_name"] = "All"

    # Column name per platform
    id_col = {
        "meta":    "extern_id",
        "google":  "user_id",
        "generic": "customer_id",
    }[platform]

    exports = []
    skipped = []
    all_segments = df["segment_name"].unique()

    for seg in sorted(all_segments):
        seg_df = df[df["segment_name"] == seg]
        n_total  = len(seg_df)
        n_send   = int((seg_df["should_send"] == 1).sum())

        if n_total < min_segment_size:
            skipped.append({"segment": seg, "n": n_total, "reason": f"< {min_segment_size} customers"})
            continue

        # All customers in segment (for lookalike / exclusion audiences)
        all_path = os.path.join(output_dir, f"{_safe_name(seg)}_all.csv")
        seg_df[["customer_id"]].rename(columns={"customer_id": id_col}).to_csv(all_path, index=False)

        # Send-approved only (for direct targeting)
        send_df = seg_df[seg_df["should_send"] == 1]
        send_path = None
        if len(send_df) >= min_segment_size:
            send_path = os.path.join(output_dir, f"{_safe_name(seg)}_send.csv")
            send_df[["customer_id"]].rename(columns={"customer_id": id_col}).to_csv(send_path, index=False)

        # Suppression list — do NOT target these (sure-things / guardrail blocked)
        supp_df = seg_df[seg_df["should_send"] == 0]
        supp_path = None
        if len(supp_df) >= min_segment_size:
            supp_path = os.path.join(output_dir, f"{_safe_name(seg)}_suppress.csv")
            supp_df[["customer_id"]].rename(columns={"customer_id": id_col}).to_csv(supp_path, index=False)

        top_action = (
            send_df["recommended_action"].value_counts().index[0]
            if len(send_df) > 0 and "recommended_action" in send_df.columns
            else "—"
        )
        avg_profit = round(float(send_df["expected_profit"].mean()), 2) if len(send_df) > 0 else 0.0

        exports.append({
            "segment":          seg,
            "n_total":          n_total,
            "n_send":           n_send,
            "n_suppress":       n_total - n_send,
            "top_action":       top_action,
            "avg_profit_per_customer": avg_profit,
            "file_all":         all_path,
            "file_send":        send_path,
            "file_suppress":    supp_path,
        })

    # Master export — all send-approved customers with segment tag
    send_all = df[df["should_send"] == 1][["customer_id", "segment_name", "recommended_action", "expected_profit"]]
    send_all = send_all.rename(columns={"customer_id": id_col})
    master_path = os.path.join(output_dir, "master_send_list.csv")
    send_all.to_csv(master_path, index=False)

    summary = {
        "exported_at":     datetime.now().isoformat(),
        "platform":        platform,
        "total_customers": len(df),
        "total_send":      int((df["should_send"] == 1).sum()),
        "segments_exported": len(exports),
        "segments_skipped":  len(skipped),
        "master_file":     master_path,
        "segments":        exports,
        "skipped":         skipped,
    }

    # Save manifest
    manifest_path = os.path.join(output_dir, "export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _print_audience_export(summary)
    log("agency", "audience_export", segments=len(exports), total_send=summary["total_send"], platform=platform)
    return summary


def _safe_name(s: str) -> str:
    return s.lower().replace(" ", "_").replace("/", "_").replace("'", "").replace("(", "").replace(")", "")


def _print_audience_export(s: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Audience Export  |  platform: {s['platform']}")
    print(f"  {s['total_customers']:,} customers  →  {s['total_send']:,} send  |  {s['segments_exported']} segments")
    print(sep)
    for seg in s["segments"]:
        send_pct = round(seg["n_send"] / seg["n_total"] * 100, 1) if seg["n_total"] else 0
        print(f"  {seg['segment']:<28} total={seg['n_total']:>5,}  send={seg['n_send']:>5,} ({send_pct:.0f}%)"
              f"  avg_profit=฿{seg['avg_profit_per_customer']:>7,.2f}")
    if s["skipped"]:
        print(f"\n  Skipped ({len(s['skipped'])}): {', '.join(x['segment'] for x in s['skipped'])}")
    print(f"\n  Master file: {s['master_file']}")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════
# 2. POST-CAMPAIGN REPORT
# ═══════════════════════════════════════════════════════

def post_campaign_report(
    campaign_df: pd.DataFrame,
    pre_pos_df: pd.DataFrame,
    post_pos_df: pd.DataFrame,
    segment_col: str = "segment_name",
    observation_days: int = 30,
    history_path: str | None = None,
) -> dict:
    """Measure actual campaign lift per segment (treated vs control).

    Treated  = customers with should_send == 1
    Control  = customers with should_send == 0 (guardrail blocked — similar profile)

    Metrics per segment:
        - visit_rate_treated / visit_rate_control
        - incremental_lift = treated_rate - control_rate
        - avg_spend_treated / avg_spend_control
        - estimated_incremental_revenue

    Parameters
    ----------
    campaign_df  : Layer 4 or promo_advisor output — must have
                   customer_id, should_send, segment_name (or cluster), expected_profit
    pre_pos_df   : POS transactions BEFORE campaign (baseline period — same length as observation_days)
    post_pos_df  : POS transactions AFTER campaign (observation window)
    observation_days : length of post-campaign observation window in days
    history_path : optional — append outcome to promo_history.jsonl

    Returns
    -------
    dict — full report with per-segment and overall lift stats
    """
    df = campaign_df.copy()
    df["customer_id"] = df["customer_id"].astype(str)

    send_col = next((c for c in ["final_should_send", "should_send"] if c in df.columns), None)
    if send_col is None:
        raise ValueError("campaign_df must have should_send or final_should_send")

    if segment_col not in df.columns:
        df[segment_col] = "All"

    # ── Pre/Post spend aggregation ────────────────────────────────────────────
    def _agg_pos(pos_df: pd.DataFrame) -> pd.DataFrame:
        pos = pos_df.copy()
        pos["customer_id"] = pos["customer_id"].astype(str)
        amt_col = next((c for c in ["total_price", "amount"] if c in pos.columns), None)
        if amt_col is None:
            raise ValueError("POS df must have total_price or amount")
        ts_col = next((c for c in ["timestamp", "transaction_date"] if c in pos.columns), None)
        return (
            pos.groupby("customer_id")
            .agg(visits=("customer_id", "count"), total_spend=(amt_col, "sum"))
            .reset_index()
        )

    pre_agg  = _agg_pos(pre_pos_df)
    post_agg = _agg_pos(post_pos_df)

    # Merge into campaign_df
    df = df.merge(pre_agg.rename(columns={"visits": "pre_visits", "total_spend": "pre_spend"}),
                  on="customer_id", how="left")
    df = df.merge(post_agg.rename(columns={"visits": "post_visits", "total_spend": "post_spend"}),
                  on="customer_id", how="left")
    df["pre_visits"]  = df["pre_visits"].fillna(0)
    df["post_visits"] = df["post_visits"].fillna(0)
    df["pre_spend"]   = df["pre_spend"].fillna(0.0)
    df["post_spend"]  = df["post_spend"].fillna(0.0)
    df["returned"]    = df["post_visits"] > 0
    df["is_treated"]  = df[send_col] == 1

    # ── Overall lift ──────────────────────────────────────────────────────────
    treated = df[df["is_treated"]]
    control = df[~df["is_treated"]]

    def _lift_stats(t: pd.DataFrame, c: pd.DataFrame) -> dict:
        t_return = float(t["returned"].mean()) if len(t) else 0.0
        c_return = float(c["returned"].mean()) if len(c) else 0.0
        t_spend  = float(t["post_spend"].mean()) if len(t) else 0.0
        c_spend  = float(c["post_spend"].mean()) if len(c) else 0.0
        lift_rate  = round(t_return - c_return, 4)
        lift_spend = round(t_spend  - c_spend,  2)
        incremental_rev = round(lift_spend * len(t), 2)
        return {
            "n_treated":          len(t),
            "n_control":          len(c),
            "visit_rate_treated": round(t_return, 4),
            "visit_rate_control": round(c_return, 4),
            "incremental_visit_lift": lift_rate,
            "avg_spend_treated":  round(t_spend,  2),
            "avg_spend_control":  round(c_spend,  2),
            "incremental_spend_per_person": lift_spend,
            "estimated_incremental_revenue": incremental_rev,
        }

    overall = _lift_stats(treated, control)

    # ── Per-segment lift ──────────────────────────────────────────────────────
    segments = {}
    for seg, grp in df.groupby(segment_col):
        t = grp[grp["is_treated"]]
        c = grp[~grp["is_treated"]]
        if len(t) < 5 or len(c) < 5:
            continue
        segments[str(seg)] = _lift_stats(t, c)

    # ── Predicted vs actual ───────────────────────────────────────────────────
    pred_profit = round(float(df.loc[df["is_treated"], "expected_profit"].sum()
                              if "expected_profit" in df.columns else 0), 2)
    actual_inc_rev = overall["estimated_incremental_revenue"]
    calibration = round(actual_inc_rev / pred_profit, 4) if pred_profit > 0 else None

    # ── Learnings (auto-generated) ────────────────────────────────────────────
    learnings = []
    if segments:
        best_seg  = max(segments, key=lambda s: segments[s]["incremental_visit_lift"])
        worst_seg = min(segments, key=lambda s: segments[s]["incremental_visit_lift"])
        learnings.append(f"Segment ที่ตอบสนองดีที่สุด: '{best_seg}' "
                         f"(lift={segments[best_seg]['incremental_visit_lift']:+.1%})")
        if segments[worst_seg]["incremental_visit_lift"] <= 0:
            learnings.append(f"Segment '{worst_seg}' ไม่ตอบสนอง (lift≤0) "
                             f"→ ไม่ควรส่งโปรประเภทนี้ให้กลุ่มนี้รอบหน้า")
    if calibration is not None:
        if calibration < 0.7:
            learnings.append("โมเดลประมาณ profit สูงกว่าความจริงมาก — ควร retrain")
        elif calibration > 1.3:
            learnings.append("ผลจริงดีกว่าที่โมเดลคาด — อาจขยาย target ได้")
        else:
            learnings.append("โมเดลประมาณค่อนข้างตรงกับผลจริง")

    report = {
        "generated_at":        datetime.now().isoformat(),
        "observation_days":    observation_days,
        "overall":             overall,
        "by_segment":          segments,
        "predicted_profit":    pred_profit,
        "actual_incremental_revenue": actual_inc_rev,
        "calibration_ratio":   calibration,
        "learnings":           learnings,
    }

    if history_path:
        os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)
        entry = {"type": "post_campaign_report", "report": report}
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _print_post_campaign(report)
    log("agency", "post_campaign_report", segments=len(segments),
        overall_lift=overall["incremental_visit_lift"],
        incremental_revenue=actual_inc_rev)
    return report


def _print_post_campaign(r: dict) -> None:
    sep = "─" * 60
    ov  = r["overall"]
    print(f"\n{sep}")
    print(f"  Post-Campaign Report  ({r['observation_days']} days)")
    print(sep)
    print(f"  Treated : {ov['n_treated']:,} คน  visit={ov['visit_rate_treated']:.1%}  "
          f"avg_spend=฿{ov['avg_spend_treated']:,.2f}")
    print(f"  Control : {ov['n_control']:,} คน  visit={ov['visit_rate_control']:.1%}  "
          f"avg_spend=฿{ov['avg_spend_control']:,.2f}")
    print(f"  Lift    : {ov['incremental_visit_lift']:+.1%} visits  |  "
          f"+฿{ov['incremental_spend_per_person']:,.2f}/person")
    print(f"  Est. incremental revenue: ฿{ov['estimated_incremental_revenue']:,.2f}")
    if r["calibration_ratio"]:
        print(f"  Calibration: {r['calibration_ratio']:.2f}x  "
              f"(predicted ฿{r['predicted_profit']:,.2f}  actual ฿{r['actual_incremental_revenue']:,.2f})")

    if r["by_segment"]:
        print(f"\n  By Segment:")
        for seg, s in sorted(r["by_segment"].items(),
                              key=lambda x: -x[1]["incremental_visit_lift"]):
            bar = "▓" * max(0, int(s["incremental_visit_lift"] * 100))
            neg = "░" if s["incremental_visit_lift"] <= 0 else ""
            print(f"    {seg:<26} lift={s['incremental_visit_lift']:+.1%}  "
                  f"inc_rev=฿{s['estimated_incremental_revenue']:>8,.2f}  {bar}{neg}")

    if r["learnings"]:
        print(f"\n  สิ่งที่เรียนรู้:")
        for l in r["learnings"]:
            print(f"    • {l}")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════
# 3. MONTHLY TREND REPORT
# ═══════════════════════════════════════════════════════

def monthly_trend_report(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame | None = None,
    planner_report: dict | None = None,
    month_label: str | None = None,
    history_path: str | None = None,
) -> dict:
    """Monthly customer trend report with segment migration matrix + next campaign proposal.

    Parameters
    ----------
    current_df    : Layer 4 output for the current month
    previous_df   : Layer 4 output for the previous month (optional — enables migration matrix)
    planner_report: output from campaign_planner.run() — appended as next-campaign proposal
    month_label   : e.g. "November 2026" (default: current month)
    history_path  : optional — append to promo_history.jsonl

    Returns
    -------
    dict — trend report suitable for quarterly brand meeting
    """
    month_label = month_label or datetime.now().strftime("%B %Y")

    # ── Current month segment stats ───────────────────────────────────────────
    curr = current_df.copy()
    curr["customer_id"] = curr["customer_id"].astype(str)
    if "segment_name" not in curr.columns:
        curr["segment_name"] = "All"

    send_col = next((c for c in ["final_should_send", "should_send"] if c in curr.columns), None)

    seg_stats: dict[str, dict] = {}
    for seg, grp in curr.groupby("segment_name"):
        n = len(grp)
        n_send  = int((grp[send_col] == 1).sum()) if send_col else 0
        avg_profit = round(float(grp["expected_profit"].mean()), 2) if "expected_profit" in grp.columns else 0.0
        freq_rising    = int((grp["freq_trend"] == "rising").sum())   if "freq_trend" in grp.columns else None
        freq_declining = int((grp["freq_trend"] == "declining").sum()) if "freq_trend" in grp.columns else None
        seg_stats[str(seg)] = {
            "n":              n,
            "n_send":         n_send,
            "send_rate":      round(n_send / n, 4) if n else 0.0,
            "avg_profit":     avg_profit,
            "freq_rising":    freq_rising,
            "freq_declining": freq_declining,
        }

    total_customers = len(curr)
    total_send      = int((curr[send_col] == 1).sum()) if send_col else 0

    # ── Migration matrix (current vs previous) ───────────────────────────────
    migration: dict | None = None
    migration_insights: list[str] = []

    if previous_df is not None:
        prev = previous_df.copy()
        prev["customer_id"] = prev["customer_id"].astype(str)
        if "segment_name" not in prev.columns:
            prev["segment_name"] = "All"

        prev_slim = prev[["customer_id", "segment_name"]].rename(
            columns={"segment_name": "prev_segment"}
        )
        curr_slim = curr[["customer_id", "segment_name"]].rename(
            columns={"segment_name": "curr_segment"}
        )
        merged = curr_slim.merge(prev_slim, on="customer_id", how="outer")
        merged["prev_segment"] = merged["prev_segment"].fillna("New")
        merged["curr_segment"] = merged["curr_segment"].fillna("Churned")

        matrix = (
            merged.groupby(["prev_segment", "curr_segment"])
            .size()
            .reset_index(name="count")
        )
        migration = matrix.to_dict(orient="records")

        # Auto-insights from migration
        for _, row in matrix.iterrows():
            if row["prev_segment"] == row["curr_segment"]:
                continue
            pct = row["count"] / len(merged) * 100
            if pct >= 5:
                migration_insights.append(
                    f"{row['count']:,} คน ({pct:.1f}%) ย้ายจาก '{row['prev_segment']}' → '{row['curr_segment']}'"
                )

        # Flag growing / shrinking segments
        prev_counts = prev["segment_name"].value_counts().to_dict()
        curr_counts = curr["segment_name"].value_counts().to_dict()
        for seg in set(list(prev_counts.keys()) + list(curr_counts.keys())):
            p = prev_counts.get(seg, 0)
            c = curr_counts.get(seg, 0)
            if p > 0:
                chg = (c - p) / p
                if chg <= -0.15:
                    migration_insights.append(f"⚠️  '{seg}' ลดลง {abs(chg):.0%} — ระวังลูกค้าหาย")
                elif chg >= 0.15:
                    migration_insights.append(f"✅  '{seg}' เพิ่มขึ้น {chg:.0%} — segment กำลังโต")

    # ── Next campaign proposal ────────────────────────────────────────────────
    next_campaign: dict | None = None
    if planner_report:
        next_campaign = {
            "goal":                 planner_report.get("goal_label"),
            "effective_budget":     planner_report.get("effective_budget"),
            "customers_selected":   planner_report.get("customers_selected"),
            "expected_profit":      planner_report.get("expected_incremental_profit"),
            "roi":                  planner_report.get("roi"),
            "top_actions":          planner_report.get("action_breakdown", {}),
        }

    report = {
        "generated_at":    datetime.now().isoformat(),
        "month":           month_label,
        "total_customers": total_customers,
        "total_send":      total_send,
        "send_rate":       round(total_send / total_customers, 4) if total_customers else 0.0,
        "segment_stats":   seg_stats,
        "migration":       migration,
        "migration_insights": migration_insights,
        "next_campaign":   next_campaign,
    }

    if history_path:
        os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)
        entry = {"type": "monthly_trend", "report": report}
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _print_monthly_trend(report)
    log("agency", "monthly_trend_report", month=month_label,
        total_customers=total_customers, segments=len(seg_stats))
    return report


def _print_monthly_trend(r: dict) -> None:
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  Monthly Trend Report  |  {r['month']}")
    print(f"  {r['total_customers']:,} customers  |  send rate {r['send_rate']:.1%}")
    print(sep)

    print(f"\n  Segment Overview:")
    print(f"    {'Segment':<28} {'Size':>6}  {'Send':>6}  {'Send%':>6}  {'Avg Profit':>10}")
    print(f"    {'─'*28} {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}")
    for seg, s in sorted(r["segment_stats"].items(), key=lambda x: -x[1]["n"]):
        rising   = f"↑{s['freq_rising']}"   if s.get("freq_rising")    else ""
        declining= f"↓{s['freq_declining']}" if s.get("freq_declining") else ""
        trend_str = f"  {rising} {declining}".strip()
        print(f"    {seg:<28} {s['n']:>6,}  {s['n_send']:>6,}  {s['send_rate']:>5.0%}  "
              f"฿{s['avg_profit']:>8,.2f} {trend_str}")

    if r.get("migration_insights"):
        print(f"\n  Segment Migration Insights:")
        for ins in r["migration_insights"][:8]:
            print(f"    • {ins}")

    if r.get("next_campaign"):
        nc = r["next_campaign"]
        print(f"\n  Next Campaign Proposal:")
        print(f"    Goal    : {nc['goal']}")
        print(f"    Budget  : ฿{nc['effective_budget']:,.0f}"
              if nc.get("effective_budget") else "    Budget  : (target-profit mode)")
        print(f"    Targets : {nc['customers_selected']:,} คน")
        print(f"    Est ROI : {nc['roi']:.2f}x" if nc.get("roi") else "")
        if nc.get("top_actions"):
            top3 = sorted(nc["top_actions"].items(), key=lambda x: -x[1])[:3]
            print(f"    Top actions:")
            for action, cnt in top3:
                print(f"      {action:<44} {cnt:>5,} คน")

    print(f"\n{sep}\n")
