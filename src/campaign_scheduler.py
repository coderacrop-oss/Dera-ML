"""
campaign_scheduler.py — Monthly Campaign Automation Planner

Takes broadcast_list.csv (Layer 4 output) and produces:
  1. campaign_schedule.csv  — per-customer send datetime, wave, message, channel
  2. ab_holdout.csv         — 5% control group withheld from all sends
  3. campaign_summary.json  — monthly plan summary for ops/CRM team

Wave logic:
  - Wave 1 (Off-Peak Traffic Driver) : send BEFORE off-peak window opens
                                        goal → pull customers in during quiet hours
  - Wave 2 (Peak Upsell/VIP)         : send BEFORE peak window opens
                                        goal → upsell customers already planning to visit
  - Wave 3 (VIP Early Access)        : send 2 days early, exclusive feel
  - Spread across 4 weeks (not all on day 1) to avoid notification fatigue
  - 5% random holdout withheld as true control for lift measurement

Run:
    uv run --with "pandas,numpy" python3 src/campaign_scheduler.py
    uv run --with "pandas,numpy" python3 src/campaign_scheduler.py \\
        --broadcast data/processed/broadcast_list.csv \\
        --month 2026-12 \\
        --peak-hours 11,12,13,14,15 \\
        --off-peak-hours 8,9,16,17 \\
        --holdout 0.05
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from atomic_io import atomic_write_text  # noqa: E402,F401  (re-exported)
from business_config import CONFIG  # noqa: E402
from peak_hours import (DEFAULT_OFF_PEAK, DEFAULT_PEAK, FILENAME, derive_from_timestamps,  # noqa: E402
                        load_peak_hours)


def stable_hash(s: str) -> int:
    """Process-independent non-negative hash (Python's hash() is salted per run)."""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


# ── Constants ──────────────────────────────────────────────────────────────────

WAVE_MAP = {
    "Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ": 1,
    "Off-Peak VIP Points: ดึงลูกค้าช่วงเงียบ (แต้มพิเศษ)":   1,
    "VIP Early Access":                                       3,
    "VIP Reward: แจกแต้มพิเศษ 100 แต้ม":                          3,
    "VIP Reward: แจกแต้มพิเศษ 80 แต้ม":                           3,
    "Standard Retention (Festival Guard)":                    2,
}
# Anything not in WAVE_MAP → Wave 2 (default peak upsell)

# SMS ≤ 160 chars, LINE ≤ 500 chars — templates are filled at render time
MESSAGE_TEMPLATES = {
    "Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ": (
        "🛍️ ช่วงพิเศษ! แวะมาก่อน {peak_start}:00 น. วันนี้ รับส่วนลดพิเศษ {offer_short}. "
        "จำกัดเวลา — ดูรายละเอียด: {link}"
    ),
    "Off-Peak VIP Points: ดึงลูกค้าช่วงเงียบ (แต้มพิเศษ)": (
        "⭐ VIP เท่านั้น! รับแต้มพิเศษเมื่อช้อปก่อน {peak_start}:00 น. วันนี้. "
        "ใช้สิทธิ์ได้เลย: {link}"
    ),
    "Push Promotion: เซตคู่คุ้มค่า (Bundle)": (
        "🎁 เซตคู่สุดคุ้ม {offer_short} — ราคาพิเศษช่วงเวลานี้เท่านั้น! ดูเลย: {link}"
    ),
    "VIP Early Access": (
        "🌟 {customer_id} คุณได้รับสิทธิ์ VIP ก่อนใคร! {offer_short} พร้อมบริการพิเศษ. "
        "จองเลย: {link}"
    ),
    "VIP Reward: แจกแต้มพิเศษ 100 แต้ม": (
        "🎯 แต้มพิเศษสำหรับคุณ! {offer_short}. ช้อปเลยก่อนหมดเวลา: {link}"
    ),
    "VIP Reward: แจกแต้มพิเศษ 80 แต้ม": (
        "🎯 แต้มพิเศษสำหรับคุณ! {offer_short}. ช้อปเลยก่อนหมดเวลา: {link}"
    ),
    "Standard Retention": (
        "💌 เรายังอยากเจอคุณ! {offer_short} — กลับมาช้อปรับสิทธิ์นี้เลย: {link}"
    ),
    "Standard Retention (Festival Guard)": (
        "🎉 เทศกาลนี้ห้ามพลาด! {offer_short}. ดูโปรทั้งหมด: {link}"
    ),
    "_default": (
        "📢 ข้อเสนอพิเศษสำหรับคุณ: {offer_short}. ดูรายละเอียด: {link}"
    ),
}

CHANNEL_CHAR_LIMIT = {
    "SMS": 160,
    "LINE": 500,
    "Email": 2000,
    "Email/SMS": 160,  # conservative — use SMS limit
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _month_workdays(year: int, month: int) -> list[datetime]:
    """Return all weekdays in the given month."""
    first = datetime(year, month, 1)
    days = []
    d = first
    while d.month == month:
        if d.weekday() < 5:  # Mon–Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def _send_hour(send_window: str, peak_hours: list[int], off_peak_hours: list[int]) -> int:
    """
    Choose notification hour:
    - Off-Peak Window → 30 min before off-peak window opens
      (we pick the hour just BEFORE the earliest off-peak hour bucket)
    - Peak Window     → 30 min before peak window opens
      (we pick the hour just BEFORE the earliest peak hour)
    Since we only have hour granularity, we use the hour before window start.
    """
    if "Off-Peak" in send_window and off_peak_hours:
        earliest_offpeak = min(off_peak_hours)
        return max(0, earliest_offpeak - 1)
    elif peak_hours:
        earliest_peak = min(peak_hours)
        return max(0, earliest_peak - 1)
    return 9  # fallback: 9 AM


def _wave_for_action(action: str) -> int:
    for key, wave in WAVE_MAP.items():
        if key in action:
            return wave
    return 2


def _render_message(action: str, row: pd.Series, peak_start: int, link: str = "shop.example.com") -> str:
    template = MESSAGE_TEMPLATES.get(action, MESSAGE_TEMPLATES["_default"])
    offer_raw = str(row.get("offer", ""))
    # Shorten offer text for SMS
    offer_short = offer_raw[:60] + "…" if len(offer_raw) > 60 else offer_raw
    msg = template.format(
        customer_id=str(row.get("customer_id", "")),
        offer_short=offer_short,
        peak_start=peak_start,
        link=link,
    )
    return msg


def _truncate_for_channel(msg: str, channel: str) -> str:
    limit = CHANNEL_CHAR_LIMIT.get(channel, 160)
    if len(msg) > limit:
        return msg[:limit - 3] + "…"
    return msg


# ── Core scheduler ─────────────────────────────────────────────────────────────

def build_schedule(
    broadcast_path: str,
    month_str: str,                          # "YYYY-MM"
    peak_hours: list[int],
    off_peak_hours: list[int],
    holdout_rate: float = 0.05,
    random_state: int = 42,
    link: str = "shop.example.com",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns (schedule_df, holdout_df, summary_dict).
    """
    rng = np.random.default_rng(random_state)
    df = pd.read_csv(broadcast_path)
    year, month = map(int, month_str.split("-"))

    # ── 1. Holdout split ────────────────────────────────────────────────────
    n = len(df)
    holdout_idx = rng.choice(n, size=max(1, int(n * holdout_rate)), replace=False)
    holdout_mask = np.zeros(n, dtype=bool)
    holdout_mask[holdout_idx] = True

    holdout_df = df[holdout_mask].copy()
    holdout_df["holdout_reason"] = "5% control — withheld to measure true lift"
    schedule_df = df[~holdout_mask].copy()

    # ── 2. Wave assignment ─────────────────────────────────────────────────
    schedule_df["wave"] = schedule_df["recommended_action"].apply(_wave_for_action)

    # ── 3. Distribute across 4 weeks ──────────────────────────────────────
    # Wave 1 (off-peak traffic) → Week 1 + 3 (early in month, repeat mid-month)
    # Wave 2 (peak upsell)      → Week 2 + 4
    # Wave 3 (VIP)              → Week 1 (exclusive early access feel)
    workdays = _month_workdays(year, month)
    if not workdays:
        raise ValueError(f"No workdays found for {month_str}")

    # Split workdays into 4 weeks (roughly)
    week_size = max(1, len(workdays) // 4)
    weeks = [workdays[i:i + week_size] for i in range(0, len(workdays), week_size)]
    while len(weeks) < 4:
        weeks.append(weeks[-1])  # pad if month has < 4 full weeks

    wave_week_map = {
        1: [weeks[0], weeks[2]],   # Wave 1: week 1 + 3
        2: [weeks[1], weeks[3]],   # Wave 2: week 2 + 4
        3: [weeks[0]],             # Wave 3: week 1 only (VIP early)
    }

    send_datetimes = []
    for _, row in schedule_df.iterrows():
        wave = int(row["wave"])
        send_window = str(row.get("send_window", "Peak Window"))
        hour = _send_hour(send_window, peak_hours, off_peak_hours)

        candidate_weeks = wave_week_map.get(wave, [weeks[1]])
        # Pick week based on customer_id hash → consistent, spreadable
        cid_hash = stable_hash(str(row["customer_id"])) % len(candidate_weeks)
        week_days = candidate_weeks[cid_hash]

        # Pick day within the week based on hash
        day_idx = stable_hash(str(row["customer_id"]) + "day") % len(week_days)
        send_day = week_days[day_idx]
        send_dt = send_day.replace(hour=hour, minute=0, second=0)
        send_datetimes.append(send_dt)

    schedule_df["send_datetime"] = send_datetimes

    # ── 4. Message rendering ───────────────────────────────────────────────
    peak_start = min(peak_hours) if peak_hours else 11
    messages = []
    for _, row in schedule_df.iterrows():
        action = str(row["recommended_action"])
        msg = _render_message(action, row, peak_start=peak_start, link=link)
        channel = str(row.get("channel", "Email/SMS"))
        msg = _truncate_for_channel(msg, channel)
        messages.append(msg)
    schedule_df["message"] = messages

    # ── 5. Final columns ───────────────────────────────────────────────────
    out_cols = [
        "customer_id", "segment_name", "wave", "recommended_action",
        "send_datetime", "send_window", "send_time_hint", "channel",
        "message", "offer", "expected_profit", "should_send",
        "action_key", "action_group",   # the stable key and category of the reward, when the engine sends them (they let a CRM group and label rows)
    ]
    out_cols = [c for c in out_cols if c in schedule_df.columns]
    schedule_df = schedule_df[out_cols].sort_values("send_datetime").reset_index(drop=True)

    # ── 6. Summary ─────────────────────────────────────────────────────────
    wave_counts = schedule_df["wave"].value_counts().to_dict()
    window_counts = schedule_df["send_window"].value_counts().to_dict()
    daily_counts = (
        schedule_df["send_datetime"].dt.date.value_counts()
        .sort_index()
        .to_dict()
    )
    daily_counts = {str(k): int(v) for k, v in daily_counts.items()}

    total_expected_profit = round(float(schedule_df["expected_profit"].sum()), 2)

    summary = {
        "month": month_str,
        "total_scheduled": len(schedule_df),
        "total_holdout": len(holdout_df),
        "holdout_rate_pct": round(holdout_rate * 100, 1),
        "peak_hours_detected": sorted(peak_hours),
        "off_peak_hours_detected": sorted(off_peak_hours),
        "wave_breakdown": {
            "wave_1_off_peak_traffic": int(wave_counts.get(1, 0)),
            "wave_2_peak_upsell":      int(wave_counts.get(2, 0)),
            "wave_3_vip_early":        int(wave_counts.get(3, 0)),
        },
        "send_window_breakdown": {k: int(v) for k, v in window_counts.items()},
        "daily_send_counts": daily_counts,
        "total_expected_profit": total_expected_profit,
        "avg_profit_per_customer": round(total_expected_profit / max(1, len(schedule_df)), 2),
    }
    # one row per (segment, reward) with the hours and days to send and what the plan expects, and what the best row alone would earn
    from plan_insights import plan_rows, top1_counterfactual
    rows = plan_rows(schedule_df, holdout_df, peak_hours, off_peak_hours)
    summary["rows"] = rows
    summary["top1_counterfactual"] = top1_counterfactual(rows)

    return schedule_df, holdout_df, summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def detect_hours(processed_dir: str = "data/processed") -> tuple[list[int], list[int], str]:
    """(peak_hours, off_peak_hours, source). Layer 3 is the single source of truth:
    1. layer3_peak_hours.json (written by Layer 3);
    2. else recompute from layer3_context_flags.csv with Layer 3's rule and timezone (Asia/Bangkok);
    3. else built-in defaults."""
    saved = load_peak_hours(processed_dir)
    if saved:
        return saved["peak_hours"], saved["off_peak_hours"], FILENAME
    ctx_path = os.path.join(processed_dir, "layer3_context_flags.csv")
    if os.path.exists(ctx_path):
        peak, off_peak = derive_from_timestamps(pd.read_csv(ctx_path)["timestamp"], CONFIG["business_tz"])
        return peak, off_peak, "layer3_context_flags.csv"
    return list(DEFAULT_PEAK), list(DEFAULT_OFF_PEAK), "defaults"


def snapshot_segments(out_dir: str, month: str) -> str | None:
    """Copy segments_latest.json (written by the pipeline) to segments_<month>.json.

    Returns the new path, or None when the pipeline has not produced segments_latest.json.
    """
    src = os.path.join(out_dir, "segments_latest.json")
    if not os.path.exists(src):
        return None
    dst = os.path.join(out_dir, f"segments_{month}.json")
    with open(src, encoding="utf-8") as f:
        atomic_write_text(dst, f.read())
    return dst


def main():
    parser = argparse.ArgumentParser(description="Monthly campaign scheduler from broadcast_list.csv")
    parser.add_argument("--broadcast",    default="data/processed/broadcast_list.csv")
    parser.add_argument("--month",        default=None,
                        help="YYYY-MM (default: next calendar month)")
    parser.add_argument("--peak-hours",   default=None,
                        help="Comma-separated peak hours detected by Layer 3, e.g. '11,12,13,14,15'")
    parser.add_argument("--off-peak-hours", default=None,
                        help="Comma-separated off-peak hours, e.g. '8,9,16,17'")
    parser.add_argument("--holdout",      type=float, default=0.05)
    parser.add_argument("--link",         default="shop.example.com")
    parser.add_argument("--out-dir",      default="data/processed")
    args = parser.parse_args()

    # ── Auto-detect month ──────────────────────────────────────────────────
    if args.month is None:
        today = datetime.today()
        if today.month == 12:
            next_month = datetime(today.year + 1, 1, 1)
        else:
            next_month = datetime(today.year, today.month + 1, 1)
        args.month = next_month.strftime("%Y-%m")

    # ── Peak hours: use what Layer 3 computed (see detect_hours) ──────────
    if args.peak_hours is None:
        peak_hours, off_peak_hours, source = detect_hours("data/processed")
        print(f"ℹ️  Peak hours ({source}): {peak_hours}")
        print(f"ℹ️  Off-peak hours: {off_peak_hours}")
    else:
        peak_hours     = [int(h) for h in args.peak_hours.split(",")]
        off_peak_hours = ([int(h) for h in args.off_peak_hours.split(",")]
                          if args.off_peak_hours else
                          [h for h in range(6, 22) if h not in peak_hours])

    print(f"\n📅 Building campaign schedule for {args.month}...")

    schedule_df, holdout_df, summary = build_schedule(
        broadcast_path=args.broadcast,
        month_str=args.month,
        peak_hours=peak_hours,
        off_peak_hours=off_peak_hours,
        holdout_rate=args.holdout,
        link=args.link,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    schedule_path = os.path.join(args.out_dir, f"campaign_schedule_{args.month}.csv")
    holdout_path  = os.path.join(args.out_dir, f"ab_holdout_{args.month}.csv")
    summary_path  = os.path.join(args.out_dir, f"campaign_summary_{args.month}.json")

    atomic_write_text(schedule_path, schedule_df.to_csv(index=False))
    atomic_write_text(holdout_path, holdout_df.to_csv(index=False))
    atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2))
    snapshot_segments(args.out_dir, args.month)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Campaign Schedule — {args.month}")
    print(f"{'='*60}")
    print(f"  Total scheduled  : {summary['total_scheduled']:,} customers")
    print(f"  Holdout (control): {summary['total_holdout']:,} customers ({summary['holdout_rate_pct']}%)")
    print(f"\n  Wave breakdown:")
    print(f"    Wave 1 (Off-Peak traffic driver) : {summary['wave_breakdown']['wave_1_off_peak_traffic']:,}")
    print(f"    Wave 2 (Peak upsell / bundle)    : {summary['wave_breakdown']['wave_2_peak_upsell']:,}")
    print(f"    Wave 3 (VIP early access)        : {summary['wave_breakdown']['wave_3_vip_early']:,}")
    print(f"\n  Send window breakdown:")
    for k, v in summary["send_window_breakdown"].items():
        print(f"    {k:<25}: {v:,}")
    print(f"\n  Expected profit (total)  : {summary['total_expected_profit']:,.2f}")
    print(f"  Expected profit (per cust): {summary['avg_profit_per_customer']:,.2f}")
    print(f"\n  Daily send cadence (top 5 days):")
    top5 = sorted(summary["daily_send_counts"].items(), key=lambda x: -x[1])[:5]
    for d, n in top5:
        print(f"    {d} : {n:,} sends")
    print(f"\n  Saved:")
    print(f"    {schedule_path}")
    print(f"    {holdout_path}")
    print(f"    {summary_path}")
    print(f"{'='*60}\n")

    # ── Sample messages ────────────────────────────────────────────────────
    print("  Sample messages:")
    for wave in [1, 2, 3]:
        sample = schedule_df[schedule_df["wave"] == wave].head(1)
        if not sample.empty:
            row = sample.iloc[0]
            print(f"\n  Wave {wave} | {row['send_window']} | {row['send_datetime'].strftime('%Y-%m-%d %H:%M')}")
            print(f"  Action  : {row['recommended_action']}")
            print(f"  Message : {row['message']}")


if __name__ == "__main__":
    main()
