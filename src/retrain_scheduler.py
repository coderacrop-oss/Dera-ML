"""retrain_scheduler.py — Scheduled outcome capture and retrain trigger.

Runs as a nightly job (cron or manual CLI).  For each campaign plan that:
  - has status "awaiting_outcome"
  - was executed >= outcome_window_days ago

It captures the outcome (pre/post revenue comparison from POS data),
updates the record to "completed", and then checks whether enough
calibration data exists to trigger a model retrain.

AC-5.4 requirements met:
  - captures 30-day post-campaign revenue automatically
  - triggers retrain assessment after each captured outcome
  - returns structured summary dict

Usage
-----
    python src/retrain_scheduler.py --data-dir data --window 30
    python src/retrain_scheduler.py --data-dir data --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Any

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

_DEFAULT_OUTCOME_WINDOW_DAYS   = 30
_DEFAULT_MIN_OUTCOMES_RETRAIN  = 3
_DEFAULT_LOOKBACK_DAYS         = 90   # only process records from last N days


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_scheduler(
    data_dir: str,
    outcome_window_days: int | None = None,
    min_outcomes_for_retrain: int | None = None,
    lookback_days: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Process pending campaign records and trigger retrain if warranted.

    Parameters
    ----------
    data_dir : str
        Root data directory (contains campaign_history.jsonl, outcomes.jsonl,
        and stores/<store_id>/latest_pos.csv).
    outcome_window_days : int
        Days to wait after campaign execution before capturing outcome.
        Defaults to 30.
    min_outcomes_for_retrain : int
        Minimum number of captured outcomes before attempting retrain.
        Defaults to 3.
    lookback_days : int
        Only consider campaigns executed within the last N days.
        Defaults to 90.
    dry_run : bool
        If True, skip POS reads and outcome writes; simulate results.

    Returns
    -------
    dict
        ``{processed, skipped, retrain_triggered, outcomes}``
    """
    cfg = _resolve_config(data_dir)
    window  = outcome_window_days    if outcome_window_days    is not None else cfg.get("outcome_window_days",   _DEFAULT_OUTCOME_WINDOW_DAYS)
    min_out = min_outcomes_for_retrain if min_outcomes_for_retrain is not None else cfg.get("min_outcomes_for_retrain", _DEFAULT_MIN_OUTCOMES_RETRAIN)
    lookback = lookback_days          if lookback_days           is not None else cfg.get("lookback_days",        _DEFAULT_LOOKBACK_DAYS)

    now = _utcnow()
    cutoff = now - timedelta(days=lookback)

    # Load pending records (awaiting_outcome, within lookback window)
    pending = _load_pending(data_dir, cutoff)
    _logger.info("Scheduler: %d pending record(s) in lookback window (%d days)", len(pending), lookback)

    processed = 0
    skipped   = 0
    captured_outcomes: list[dict] = []

    for record in pending:
        exec_at = _parse_iso(record.get("executed_at") or record.get("saved_at", ""))
        if exec_at is None:
            _logger.warning("Record %s has no executable timestamp -- skip", record.get("plan_id"))
            skipped += 1
            continue

        age_days = (now - exec_at).total_seconds() / 86400
        if age_days < window:
            _logger.debug("Record %s is only %.1f days old (need %d) -- skip",
                          record.get("plan_id"), age_days, window)
            skipped += 1
            continue

        outcome = _capture_outcome(data_dir, record, window_days=window, dry_run=dry_run)
        if outcome is None:
            _logger.warning("Could not capture outcome for plan %s", record.get("plan_id"))
            skipped += 1
            continue

        if not dry_run:
            _write_outcome(data_dir, outcome)
            _mark_plan_completed(data_dir, record["plan_id"])

        captured_outcomes.append(outcome)
        processed += 1
        _logger.info("Scheduler: captured outcome for plan %s -- post_revenue=%.2f",
                     outcome["plan_id"], outcome.get("post_revenue", 0))

    # Trigger retrain assessment if we have enough outcomes
    retrain_triggered = False
    total_outcomes = _count_outcomes(data_dir) + (len(captured_outcomes) if dry_run else 0)
    if total_outcomes >= min_out:
        retrain_triggered = _maybe_trigger_retrain(data_dir)
    else:
        _logger.info("Scheduler: %d/%d outcomes -- retrain check deferred", total_outcomes, min_out)

    return {
        "processed":         processed,
        "skipped":           skipped,
        "retrain_triggered": retrain_triggered,
        "outcomes":          captured_outcomes,
    }


# ---------------------------------------------------------------------------
# Outcome capture
# ---------------------------------------------------------------------------

def _capture_outcome(
    data_dir: str,
    record: dict,
    window_days: int = _DEFAULT_OUTCOME_WINDOW_DAYS,
    dry_run: bool = False,
) -> dict | None:
    """Compare pre/post revenue from POS data for a campaign record.

    Returns an outcome dict or None if POS data is unavailable.
    """
    plan_id  = record.get("plan_id", "unknown")
    store_id = record.get("store_id")
    if not store_id:
        _logger.warning("_capture_outcome: record %s has no store_id", plan_id)
        return None

    exec_at = _parse_iso(record.get("executed_at") or record.get("saved_at", ""))
    if exec_at is None:
        return None

    pre_start  = exec_at - timedelta(days=window_days)
    post_end   = exec_at + timedelta(days=window_days)

    if dry_run:
        return {
            "plan_id":           plan_id,
            "store_id":          store_id,
            "pre_revenue":       0.0,
            "post_revenue":      0.0,
            "lift":              0.0,
            "lift_pct":          0.0,
            "window_days":       window_days,
            "executed_at":       record.get("executed_at") or record.get("saved_at", ""),
            "captured_at":       _utcnow().isoformat(),
            "dry_run":           True,
        }

    pos_df = _load_pos(data_dir, store_id)
    if pos_df is None:
        return None

    pos_df = _normalise_pos(pos_df)
    if "transaction_date" not in pos_df.columns or "amount" not in pos_df.columns:
        _logger.warning("_capture_outcome: POS data for %s missing required columns", store_id)
        return None

    try:
        import pandas as pd
        pos_df["transaction_date"] = pd.to_datetime(pos_df["transaction_date"], errors="coerce")
        pos_df = pos_df.dropna(subset=["transaction_date"])

        # Align timezone: strip tz from boundaries if column is tz-naive
        col_tz = getattr(pos_df["transaction_date"].dt, "tz", None)
        if col_tz is None:
            _pre_start = pre_start.replace(tzinfo=None)
            _exec_at   = exec_at.replace(tzinfo=None)
            _post_end  = post_end.replace(tzinfo=None)
        else:
            _pre_start, _exec_at, _post_end = pre_start, exec_at, post_end

        pre_mask  = (pos_df["transaction_date"] >= _pre_start) & (pos_df["transaction_date"] < _exec_at)
        post_mask = (pos_df["transaction_date"] >= _exec_at)   & (pos_df["transaction_date"] <= _post_end)

        pre_revenue  = float(pos_df.loc[pre_mask,  "amount"].sum())
        post_revenue = float(pos_df.loc[post_mask, "amount"].sum())
        lift         = post_revenue - pre_revenue
        lift_pct     = round(lift / pre_revenue * 100, 2) if pre_revenue > 0 else 0.0

    except Exception as exc:
        _logger.error("_capture_outcome: error computing revenue for %s -- %s", store_id, exc)
        return None

    return {
        "plan_id":      plan_id,
        "store_id":     store_id,
        "pre_revenue":  round(pre_revenue, 2),
        "post_revenue": round(post_revenue, 2),
        "lift":         round(lift, 2),
        "lift_pct":     lift_pct,
        "window_days":  window_days,
        "executed_at":  record.get("executed_at") or record.get("saved_at", ""),
        "captured_at":  _utcnow().isoformat(),
        "dry_run":      False,
    }


# ---------------------------------------------------------------------------
# Retrain trigger
# ---------------------------------------------------------------------------

def _maybe_trigger_retrain(data_dir: str) -> bool:
    """Call retrain_trigger.assess_retrain() and log the verdict.

    Returns True if retraining is recommended.
    """
    _add_src_to_path()
    try:
        from retrain_trigger import assess_retrain
        result = assess_retrain()
        if result.get("needs_retrain"):
            _logger.info(
                "Scheduler: retrain RECOMMENDED -- reasons: %s",
                "; ".join(result.get("reasons", [])),
            )
        else:
            _logger.info("Scheduler: retrain not needed -- %s",
                         "; ".join(result.get("warnings", ["ok"])))
        return bool(result.get("needs_retrain", False))
    except Exception as exc:
        _logger.error("Scheduler: retrain trigger failed -- %s", exc)
        return False


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _load_pending(data_dir: str, cutoff: datetime) -> list[dict]:
    """Return campaign history records with status 'awaiting_outcome' within cutoff."""
    path = os.path.join(data_dir, "campaign_history.jsonl")
    if not os.path.exists(path):
        return []

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec.get("status") != "awaiting_outcome":
                continue

            ts = _parse_iso(rec.get("executed_at") or rec.get("saved_at", ""))
            if ts and ts >= cutoff:
                records.append(rec)

    return records


def _write_outcome(data_dir: str, outcome: dict) -> None:
    path = os.path.join(data_dir, "outcomes.jsonl")
    os.makedirs(data_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(outcome, ensure_ascii=False) + "\n")


def _mark_plan_completed(data_dir: str, plan_id: str) -> None:
    """Rewrite campaign_history.jsonl updating the matching plan to 'completed'."""
    path = os.path.join(data_dir, "campaign_history.jsonl")
    if not os.path.exists(path):
        return

    lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            try:
                rec = json.loads(stripped)
                if rec.get("plan_id") == plan_id and rec.get("status") == "awaiting_outcome":
                    rec["status"] = "completed"
                    rec["completed_at"] = _utcnow().isoformat()
                    lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    lines.append(line)
            except json.JSONDecodeError:
                lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _count_outcomes(data_dir: str) -> int:
    path = os.path.join(data_dir, "outcomes.jsonl")
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_pos(data_dir: str, store_id: str):
    safe = "".join(c for c in store_id if c.isalnum() or c in ("-", "_"))
    pos_path = os.path.join(data_dir, "stores", safe, "latest_pos.csv")
    if not os.path.exists(pos_path):
        _logger.warning("_load_pos: no POS file for store %s at %s", store_id, pos_path)
        return None
    try:
        import pandas as pd
        return pd.read_csv(pos_path, low_memory=False)
    except Exception as exc:
        _logger.error("_load_pos: failed to read %s -- %s", pos_path, exc)
        return None


def _normalise_pos(df):
    """Rename known column variants to standard names."""
    col_map = {
        "date":        "transaction_date",
        "Date":        "transaction_date",
        "trans_date":  "transaction_date",
        "total":       "amount",
        "Total":       "amount",
        "net_amount":  "amount",
        "revenue":     "amount",
    }
    return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})


def _resolve_config(data_dir: str) -> dict:
    """Read scheduler settings from business_config if available."""
    _add_src_to_path()
    try:
        from business_config import BUSINESS_CONFIG  # type: ignore
        return {
            "outcome_window_days":      BUSINESS_CONFIG.get("outcome_window_days",      _DEFAULT_OUTCOME_WINDOW_DAYS),
            "min_outcomes_for_retrain": BUSINESS_CONFIG.get("min_outcomes_for_retrain", _DEFAULT_MIN_OUTCOMES_RETRAIN),
            "lookback_days":            BUSINESS_CONFIG.get("lookback_days",            _DEFAULT_LOOKBACK_DAYS),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: str) -> "datetime | None":
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _add_src_to_path() -> None:
    src = os.path.dirname(__file__)
    if src not in sys.path:
        sys.path.insert(0, src)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Capture campaign outcomes and trigger retrain if needed.")
    p.add_argument("--data-dir",     default="data",
                   help="Root data directory (default: data)")
    p.add_argument("--dry-run",      action="store_true",
                   help="Skip writes; simulate")
    p.add_argument("--window",       type=int, default=None,
                   help=f"Days after execution to capture outcome (default: {_DEFAULT_OUTCOME_WINDOW_DAYS})")
    p.add_argument("--lookback",     type=int, default=None,
                   help=f"Only process campaigns from last N days (default: {_DEFAULT_LOOKBACK_DAYS})")
    p.add_argument("--min-outcomes", type=int, default=None,
                   help=f"Min outcomes before retrain check (default: {_DEFAULT_MIN_OUTCOMES_RETRAIN})")
    p.add_argument("--json",         action="store_true",
                   help="Machine-readable JSON output")
    p.add_argument("--verbose",      action="store_true",
                   help="Debug logging")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    )

    result = run_scheduler(
        data_dir=args.data_dir,
        outcome_window_days=args.window,
        min_outcomes_for_retrain=args.min_outcomes,
        lookback_days=args.lookback,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Processed : {result['processed']}")
        print(f"Skipped   : {result['skipped']}")
        print(f"Retrain   : {'yes' if result['retrain_triggered'] else 'no'}")
        print(f"Outcomes  : {len(result['outcomes'])}")


if __name__ == "__main__":
    main()
