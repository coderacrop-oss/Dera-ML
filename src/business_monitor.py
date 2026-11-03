"""
business_monitor.py — Business metric monitoring over time

Appends a snapshot to data/processed/business_metrics_history.csv after each serve.
Tracks: timestamp, champion_version, send_rate_pct, avg_expected_profit,
        do_nothing_pct, guardrail_blocked_pct, n_customers.

Called automatically at the end of run_layer4() (serving pipeline).
Also usable standalone:
  python src/business_monitor.py               # print latest trend
  python src/business_monitor.py --history 10  # last 10 snapshots
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HISTORY_PATH = "data/processed/business_metrics_history.csv"


def record_snapshot(detail_df: pd.DataFrame, champion_version: str) -> dict:
    """
    Compute business metrics from the serving output DataFrame and append to history.

    Parameters
    ──────────
    detail_df        : per-customer DataFrame from decide_for_customers()
    champion_version : string like "v5"

    Returns the snapshot dict.
    """
    n = len(detail_df)
    if n == 0:
        return {}

    send_mask    = detail_df["should_send"] == 1
    blocked_mask = detail_df["guardrail_blocked"] == 1
    dn_mask      = detail_df["recommended_action"].str.startswith("Do Nothing")

    snapshot = {
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "champion_version":      champion_version,
        "n_customers":           n,
        "send_rate_pct":         round(float(send_mask.mean() * 100), 2),
        "avg_expected_profit":   round(float(detail_df["expected_profit"].mean()), 4),
        "total_expected_profit": round(float(detail_df["expected_profit"].sum()), 2),
        "do_nothing_pct":        round(float(dn_mask.mean() * 100), 2),
        "guardrail_blocked_pct": round(float(blocked_mask.mean() * 100), 2),
    }

    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    row = pd.DataFrame([snapshot])
    if os.path.exists(HISTORY_PATH):
        row.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(HISTORY_PATH, index=False)

    return snapshot


def load_history(n: int | None = None) -> pd.DataFrame:
    """Load business metrics history. Returns last `n` rows if specified."""
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
    if n is not None:
        df = df.tail(n)
    return df


def print_trend(n: int = 10) -> None:
    """Print a concise trend table of the last n snapshots."""
    df = load_history(n)
    if df.empty:
        print("ℹ️  [Business Monitor] No history yet. Run the serving pipeline first.")
        return

    print(f"\n📈 [Business Monitor] Last {len(df)} serving runs:")
    print(f"  {'Timestamp':<26} {'Ver':<5} {'Cust':>5}  "
          f"{'Send%':>6}  {'AvgProfit':>10}  {'DoNothing%':>10}  {'Blocked%':>8}")
    print("  " + "-" * 78)
    for _, row in df.iterrows():
        ts = str(row["timestamp"])[:19].replace("T", " ")
        print(f"  {ts:<26} {row.get('champion_version','?'):<5} "
              f"{int(row.get('n_customers', 0)):>5}  "
              f"{row.get('send_rate_pct', 0):>6.1f}%  "
              f"{row.get('avg_expected_profit', 0):>10.4f}  "
              f"{row.get('do_nothing_pct', 0):>9.1f}%  "
              f"{row.get('guardrail_blocked_pct', 0):>7.1f}%")

    # Simple trend: compare first vs last
    if len(df) >= 2:
        first, last = df.iloc[0], df.iloc[-1]
        delta_profit = last["avg_expected_profit"] - first["avg_expected_profit"]
        delta_send   = last["send_rate_pct"] - first["send_rate_pct"]
        trend_icon = "📈" if delta_profit > 0 else ("📉" if delta_profit < 0 else "➡️ ")
        print(f"\n  {trend_icon} Trend ({len(df)} runs): "
              f"avg_profit {delta_profit:+.4f}  send_rate {delta_send:+.1f}pp")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", type=int, default=10, metavar="N",
                   help="Show last N serving snapshots (default: 10)")
    args = p.parse_args(argv)
    print_trend(args.history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
