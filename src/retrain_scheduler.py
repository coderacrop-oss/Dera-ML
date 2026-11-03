"""
retrain_scheduler.py — Automated retraining orchestration

Checks whether retraining is needed (via retrain_trigger) and, if so,
runs train_champion.py to create a new candidate (does NOT auto-promote —
a human still approves via the CRM API page).

Designed to be called from cron or a CI pipeline:
  # cron example: every Monday at 2 AM
  0 2 * * 1  cd /path/to/ml-campaign-engine && \\
             python src/retrain_scheduler.py \\
               --store-data store-a:data/transactions.csv \\
               --store-data store-b:data/transactions_b.csv

Exit codes:
  0 — retrain not needed, or retrain ran and created a candidate successfully
  1 — retrain was needed but failed (check stderr)
  2 — configuration error (missing --store-data when retraining is triggered)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

from business_config import REGISTRY_PATH
from pipeline_logger import log
from retrain_trigger import assess_retrain


def _run_training(store_data: list[str], models_root: str, holdout_frac: float, seed: int) -> int:
    """Invoke train_champion.py as a subprocess. Returns its exit code."""
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "train_champion.py")]
    for s in store_data:
        cmd += ["--store-data", s]
    cmd += ["--models-root", models_root, "--holdout-frac", str(holdout_frac), "--seed", str(seed)]
    print(f"🔄 [Scheduler] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store-data", action="append", default=[], metavar="STORE_ID:CSV[:LABELS]",
                   help="Store data files (required if retraining is triggered)")
    p.add_argument("--models-root", default=os.path.dirname(REGISTRY_PATH))
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Retrain even if trigger says not needed")
    p.add_argument("--dry-run", action="store_true",
                   help="Check trigger only; do not actually train")
    args = p.parse_args(argv)

    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n⏰ [Scheduler] {ts}")

    trigger = assess_retrain(args.models_root)
    needs_retrain = trigger["needs_retrain"] or args.force

    print(f"{'⚠️ ' if needs_retrain else '✅'} [Scheduler] needs_retrain={needs_retrain}  "
          f"drift={trigger.get('drift_status', 'N/A')}  "
          f"qini={trigger.get('qini_current', 'N/A')}")

    if trigger["reasons"]:
        for r in trigger["reasons"]:
            print(f"   ❌ {r}")
    if trigger["warnings"]:
        for w in trigger["warnings"]:
            print(f"   ⚠️  {w}")

    if not needs_retrain:
        print("✅ [Scheduler] No retraining needed — skipping.")
        log("scheduler", "skip", reason="trigger_clean", **{
            k: v for k, v in trigger.items() if k != "reasons" and k != "warnings"
        })
        return 0

    if args.dry_run:
        print("ℹ️  [Scheduler] --dry-run: would retrain but skipping actual training.")
        return 0

    if not args.store_data:
        print("❌ [Scheduler] Retraining needed but no --store-data provided. "
              "Add --store-data STORE:data/file.csv to the scheduler command.")
        log("scheduler", "error", reason="no_store_data")
        return 2

    print("🔄 [Scheduler] Starting retraining...")
    log("scheduler", "retrain_start", reasons=trigger["reasons"])
    rc = _run_training(args.store_data, args.models_root, args.holdout_frac, args.seed)
    if rc == 0:
        print("✅ [Scheduler] Candidate created. Open the CRM API approval page to promote it.")
        log("scheduler", "retrain_success")
    else:
        print(f"❌ [Scheduler] Training failed (exit code {rc}). Check output above.")
        log("scheduler", "retrain_failed", exit_code=rc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
