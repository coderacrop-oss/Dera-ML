"""
drift_detector.py — Feature distribution monitoring (Population Stability Index)

Detects distribution shift between training data and serving data.

PSI thresholds (industry standard):
  < 0.10  — stable, no action needed
  0.10–0.25 — moderate shift, monitor closely
  > 0.25  — significant drift → retrain recommended

Usage
─────
Training time (attached to model):
    model.feature_baseline_stats = compute_baseline(train_df, features)

Serving time (called in run_layer4 after building cust_agg):
    report = check_drift(champion.feature_baseline_stats, serve_df, features)
    if report["overall_status"] != "stable":
        log warning / print alert
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

PSI_MODERATE = 0.10
PSI_CRITICAL = 0.25
N_BINS       = 10

# Features that follow power-law / heavy-tailed distributions.
# For PSI comparison we work in log1p-space so that the distance between
# freq=1 and freq=5 is comparable to the distance between freq=50 and freq=200,
# preventing a handful of high-volume wholesale customers from pushing bin edges
# so high that all retail/new-customer serving data crowds into the first bin.
# The model itself still receives RAW values (no transform applied at feature level).
_LOG1P_FEATURES: frozenset[str] = frozenset({"frequency", "total_price", "quantity"})


def _maybe_log1p(col: np.ndarray, feat: str) -> np.ndarray:
    """Return log1p(col) for skewed count/spend features, unchanged otherwise."""
    return np.log1p(col) if feat in _LOG1P_FEATURES else col


# ── Baseline computation ───────────────────────────────────────────────────────

def compute_baseline(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """
    Compute per-feature baseline statistics from training data.

    Stores quantile-based bin edges + expected probabilities per bin.
    Laplace-smoothed so no bin ever has probability 0.
    Skewed features (frequency, total_price, quantity) are compared in log1p-space
    to avoid scale domination by high-volume customers.

    Attach to model before pickling:
        model.feature_baseline_stats = compute_baseline(train_df, features)
    """
    stats: dict[str, Any] = {}
    for feat in features:
        if feat not in df.columns:
            continue
        col = df[feat].dropna().astype(float).values
        col = _maybe_log1p(col, feat)
        col = col[np.isfinite(col)]
        if len(col) < 5:
            continue

        # Quantile-based edges (equal-frequency bins → each bin has ≥1 training point)
        edges = np.unique(
            np.percentile(col, np.linspace(0, 100, N_BINS + 1))
        )
        # Need at least 2 unique edges
        if len(edges) < 2:
            edges = np.array([col.min() - 1e-9, col.max() + 1e-9])

        counts, _ = np.histogram(col, bins=edges)
        # Laplace smoothing: avoids log(0) in PSI
        total          = counts.sum() + len(counts) * 1e-6
        expected_p     = (counts + 1e-6) / total

        stats[feat] = {
            "edges":      edges.tolist(),
            "expected_p": expected_p.tolist(),
            "mean":       float(col.mean()),
            "std":        float(col.std()),
            "p10":        float(np.percentile(col, 10)),
            "p50":        float(np.percentile(col, 50)),
            "p90":        float(np.percentile(col, 90)),
            "n_train":    int(len(col)),
            "log1p_transformed": feat in _LOG1P_FEATURES,
        }
    return stats


# ── PSI computation ────────────────────────────────────────────────────────────

def _psi_for_feature(
    base: dict[str, Any],
    col: pd.Series,
    feat: str = "",
) -> tuple[float, str]:
    """Compute PSI for one feature.  Returns (psi_value, status_label)."""
    values   = col.dropna().astype(float).values
    # Apply the same transform that was used when building the baseline
    if base.get("log1p_transformed"):
        values = np.log1p(values.clip(min=0))
    edges    = np.array(base["edges"])
    exp_p    = np.array(base["expected_p"])

    if len(values) == 0:
        return float("nan"), "empty"

    counts, _ = np.histogram(values, bins=edges)
    total      = counts.sum() + len(counts) * 1e-6
    act_p      = (counts + 1e-6) / total

    # PSI = Σ (actual_i − expected_i) × ln(actual_i / expected_i)
    psi = float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))

    if psi < PSI_MODERATE:
        status = "stable"
    elif psi < PSI_CRITICAL:
        status = "moderate_shift"
    else:
        status = "significant_drift"

    return psi, status


def check_drift(
    baseline_stats: dict[str, Any],
    serving_df: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    """
    Compare serving data distribution to training baseline.

    Parameters
    ──────────
    baseline_stats : dict from compute_baseline() (stored on model)
    serving_df     : current cust_agg DataFrame from build_layer4_context()
    features       : list of feature names to check

    Returns
    ───────
    {
      "overall_status":  "stable" | "moderate_shift" | "significant_drift",
      "overall_max_psi": float,
      "drifted_features": [...],
      "recommendation":  str | None,
      "per_feature": { feat: {"psi": float, "status": str, ...} }
    }
    """
    if not baseline_stats:
        return {"overall_status": "no_baseline", "per_feature": {}}

    per_feat: dict[str, Any] = {}
    max_psi   = 0.0
    drifted   = []

    for feat in features:
        if feat not in baseline_stats:
            per_feat[feat] = {"psi": None, "status": "no_baseline"}
            continue
        if feat not in serving_df.columns:
            per_feat[feat] = {"psi": None, "status": "missing_in_serving"}
            continue

        base = baseline_stats[feat]
        psi, status = _psi_for_feature(base, serving_df[feat], feat)

        mean_serve = (
            float(serving_df[feat].dropna().mean())
            if len(serving_df[feat].dropna()) > 0 else None
        )

        per_feat[feat] = {
            "psi":        round(psi, 4) if not np.isnan(psi) else None,
            "status":     status,
            "mean_train": round(base["mean"], 4),
            "mean_serve": round(mean_serve, 4) if mean_serve is not None else None,
            "p50_train":  round(base["p50"], 4),
        }

        if not np.isnan(psi):
            max_psi = max(max_psi, psi)
        if status in ("moderate_shift", "significant_drift"):
            drifted.append(feat)

    # Overall assessment
    if max_psi < PSI_MODERATE:
        overall   = "stable"
        rec       = None
    elif max_psi < PSI_CRITICAL:
        overall   = "moderate_shift"
        rec       = (
            f"Moderate distribution shift in: {', '.join(drifted)}. "
            f"Monitor results closely; plan retraining if shift persists."
        )
    else:
        critical = [f for f in drifted
                    if per_feat[f].get("status") == "significant_drift"]
        overall  = "significant_drift"
        rec      = (
            f"⚠️  Significant drift in: {', '.join(critical)}. "
            f"Serving data looks different from training data — "
            f"model predictions may be unreliable. Retrain recommended."
        )

    return {
        "overall_status":   overall,
        "overall_max_psi":  round(max_psi, 4),
        "drifted_features": drifted,
        "recommendation":   rec,
        "per_feature":      per_feat,
    }


# ── Console printer ────────────────────────────────────────────────────────────

def print_drift_report(report: dict) -> None:
    """Print a concise drift report to stdout."""
    status = report.get("overall_status", "unknown")
    max_psi = report.get("overall_max_psi", 0)

    icon = {"stable": "✅", "moderate_shift": "⚠️ ", "significant_drift": "❌"}.get(status, "ℹ️ ")
    print(f"\n{icon} [Drift] Status: {status}  (max PSI={max_psi:.3f})")

    per_feat = report.get("per_feature", {})
    drifted  = [f for f, v in per_feat.items()
                if v.get("status") in ("moderate_shift", "significant_drift")]

    if drifted:
        print(f"   Drifted features:")
        for feat in drifted:
            v = per_feat[feat]
            psi_str = f"{v['psi']:.3f}" if v.get("psi") is not None else "N/A"
            print(f"     {feat:<30} PSI={psi_str}  "
                  f"train_mean={v.get('mean_train', '?')}  "
                  f"serve_mean={v.get('mean_serve', '?')}")

    rec = report.get("recommendation")
    if rec:
        print(f"   → {rec}")
