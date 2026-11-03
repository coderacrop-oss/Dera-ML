"""
retrain_trigger.py — Retraining need assessment

Reads:
  data/processed/drift_report.json     (written by run_layer4 after each serve)
  models/registry/manifest.json        (registry — current champion metrics)

Exit codes:
  0 — no retraining needed
  1 — retraining recommended (drift or quality issue detected)

Usage:
  python src/retrain_trigger.py [--models-root models]
  python src/retrain_trigger.py --json   # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from business_config import REGISTRY_PATH

DRIFT_REPORT_PATH   = "data/processed/drift_report.json"
PSI_RETRAIN         = 0.25   # significant_drift threshold
QINI_MIN_ACCEPTABLE = 0.05   # below this → weak model, retrain


def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def assess_retrain(models_root: str | None = None) -> dict:
    """
    Returns a dict:
    {
        "needs_retrain": bool,
        "reasons":       [str, ...],
        "warnings":      [str, ...],
        "drift_status":  str | None,
        "qini_current":  float | None,
    }
    """
    reasons:  list[str] = []
    warnings: list[str] = []
    drift_status: str | None   = None
    qini_current: float | None = None

    # ── 1. Feature drift ──────────────────────────────────────────────────────
    drift = _load_json(DRIFT_REPORT_PATH)
    if drift is None:
        warnings.append(
            f"No drift report found at {DRIFT_REPORT_PATH}. "
            f"Run the serving pipeline first to generate it."
        )
    else:
        drift_status = drift.get("overall_status", "unknown")
        max_psi = drift.get("overall_max_psi", 0.0)

        if drift_status == "significant_drift":
            reasons.append(
                f"Significant feature drift detected (max PSI={max_psi:.3f} ≥ {PSI_RETRAIN}). "
                f"Drifted features: {', '.join(drift.get('drifted_features', []))}."
            )
        elif drift_status == "moderate_shift":
            warnings.append(
                f"Moderate feature shift (max PSI={max_psi:.3f}). "
                f"Monitor closely; retrain if shift persists next cycle."
            )

    # ── 2. Model quality from manifest ───────────────────────────────────────
    root = models_root or os.path.dirname(REGISTRY_PATH)
    manifest_path = os.path.join(root, "registry", "manifest.json")
    manifest = _load_json(manifest_path)

    if manifest is None:
        warnings.append("No model registry manifest found — cannot check Qini score.")
    else:
        current_ver = manifest.get("current")
        versions = manifest.get("versions", [])
        current_entry = next((v for v in versions if v["version"] == current_ver), None)

        if current_entry:
            metrics = current_entry.get("metrics") or {}
            # metrics = {"challenger": {..., "uplift_metrics": {...}}, "champion": ...}
            chal = metrics.get("challenger") or {}
            uplift_m = chal.get("uplift_metrics") or {}
            qini_current = uplift_m.get("qini")

            if qini_current is None:
                warnings.append(
                    f"Champion {current_ver} has no Qini score in manifest "
                    f"(trained before P0 upgrade). Retrain to get quality metrics."
                )
            elif qini_current < QINI_MIN_ACCEPTABLE:
                reasons.append(
                    f"Champion Qini={qini_current:.4f} < {QINI_MIN_ACCEPTABLE} "
                    f"(weak/no targeting signal). Model should be retrained with fresh data."
                )
            else:
                interpretation = uplift_m.get("interpretation", "")
                if qini_current < 0.08:
                    warnings.append(
                        f"Champion Qini={qini_current:.4f} (moderate). "
                        f"Consider retraining with more data if possible."
                    )

    return {
        "needs_retrain": len(reasons) > 0,
        "reasons":       reasons,
        "warnings":      warnings,
        "drift_status":  drift_status,
        "qini_current":  qini_current,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-root", default=None)
    p.add_argument("--json", action="store_true", help="output machine-readable JSON")
    args = p.parse_args(argv)

    result = assess_retrain(args.models_root)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        icon = "⚠️ " if result["needs_retrain"] else "✅"
        print(f"\n{icon} [Retrain Check] needs_retrain={result['needs_retrain']}")

        if result["reasons"]:
            print("  Reasons:")
            for r in result["reasons"]:
                print(f"    ❌ {r}")

        if result["warnings"]:
            print("  Warnings:")
            for w in result["warnings"]:
                print(f"    ⚠️  {w}")

        if not result["reasons"] and not result["warnings"]:
            print("  All checks passed — model and data look healthy.")

        if result["drift_status"]:
            print(f"  Drift status  : {result['drift_status']}")
        if result["qini_current"] is not None:
            print(f"  Current Qini  : {result['qini_current']:.4f}")

    return 1 if result["needs_retrain"] else 0


if __name__ == "__main__":
    sys.exit(main())
