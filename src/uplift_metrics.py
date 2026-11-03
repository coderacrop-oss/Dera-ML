"""
uplift_metrics.py — Model quality metrics for uplift / CATE models

Called during training to populate the model registry with quality scores.

Metrics
───────
qini_coefficient  : normalized Qini coefficient (0 = random, >0 = beats random)
auuc              : avg incremental conversion rate across scored deciles
ate_accuracy      : predicted ATE vs observed ATE on holdout
baseline_ece      : Expected Calibration Error of mu0 outcome model

Qini interpretation (industry standard):
  < 0.05  — weak or no signal
  0.05–0.15 — moderate uplift model
  > 0.15  — strong model
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Qini curve ─────────────────────────────────────────────────────────────────

def _qini_gains(y: np.ndarray, t: np.ndarray, scores: np.ndarray):
    """Cumulative Qini gains sorted by predicted uplift (descending).

    Returns (fractions_targeted, qini_gains) arrays of length n.
    """
    idx     = np.argsort(scores)[::-1]
    y_s     = y[idx].astype(float)
    t_s     = t[idx].astype(float)
    n       = len(y_s)
    n_t     = t_s.sum()
    n_c     = n - n_t

    if n_t < 5 or n_c < 5:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    cum_t_conv = np.cumsum(y_s * t_s)
    cum_c_conv = np.cumsum(y_s * (1.0 - t_s))

    # Qini gain at position k:  (cum_t_conv_k / N_t  -  cum_c_conv_k / N_c) × N
    gains    = (cum_t_conv / n_t - cum_c_conv / n_c) * n
    fracs    = np.arange(1, n + 1) / n
    return fracs, gains


def qini_coefficient(y: np.ndarray, t: np.ndarray, scores: np.ndarray) -> float:
    """
    Qini coefficient: normalized area between model curve and random baseline.

    Returns a value where 0 ≈ random model, positive = better than random.
    Uses (AUC_model - AUC_random) / |AUC_random| normalization.
    """
    fracs, gains = _qini_gains(y, t, scores)

    # Random baseline: linear from 0 to gains[-1]
    q_random = fracs * gains[-1]

    auc_model  = float(np.trapezoid(gains,    fracs))
    auc_random = float(np.trapezoid(q_random, fracs))
    diff       = auc_model - auc_random

    if abs(auc_random) < 1e-9:
        # No overall lift → coefficient is sign of model improvement over chance
        return float(np.clip(diff, -1.0, 1.0))

    return float(np.clip(diff / abs(auc_random), -3.0, 3.0))


# ── AUUC (decile-based) ────────────────────────────────────────────────────────

def auuc(y: np.ndarray, t: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    """
    Area Under Uplift Curve — decile-based average incremental conversion.

    Splits customers into n_bins deciles by predicted uplift,
    computes (treated_rate - control_rate) per decile, returns weighted mean.

    Interpretation: average incremental conversion rate the model captures
    across all score deciles.  A value of 0.12 means the model on average
    identifies customers who convert 12 pp more than control when treated.
    """
    df = pd.DataFrame({"y": y, "t": t, "score": scores})
    try:
        df["bin"] = pd.qcut(df["score"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        # Too few unique scores
        return float("nan")

    lifts, weights = [], []
    for _, sub in df.groupby("bin"):
        n_tr = (sub["t"] == 1).sum()
        n_ct = (sub["t"] == 0).sum()
        if n_tr < 3 or n_ct < 3:
            continue
        lift = float(sub[sub["t"] == 1]["y"].mean() - sub[sub["t"] == 0]["y"].mean())
        lifts.append(lift)
        weights.append(len(sub))

    if not lifts:
        return float("nan")

    return float(np.average(lifts, weights=weights))


# ── ECE ────────────────────────────────────────────────────────────────────────

def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (lower = better calibrated)."""
    if len(y_true) < 5:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


# ── Main entry point ───────────────────────────────────────────────────────────

def compute_uplift_metrics(
    model,
    X: np.ndarray | pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
) -> dict:
    """
    Full uplift model quality metrics on holdout data.

    Parameters
    ──────────
    model  : trained XLearnerModel or TLearnerModel
    X      : feature matrix (shape n × p)
    T      : treatment indicator (0/1), shape (n,)
    Y      : observed outcome (0/1), shape (n,)

    Returns a dict ready to be stored in the model registry manifest.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    y = np.array(Y, dtype=float)
    t = np.array(T, dtype=float)

    n_total = len(y)
    n_treat = int(t.sum())
    n_ctrl  = n_total - n_treat

    if n_total < 10:
        return {"error": "holdout too small for metrics", "n_holdout": n_total}

    # Predicted uplift scores
    scores = np.array(model.predict_uplift(X), dtype=float)

    # ATE: predicted vs observed
    ate_pred   = float(np.mean(scores))
    ate_actual = (
        float(y[t == 1].mean() - y[t == 0].mean())
        if n_treat > 0 and n_ctrl > 0
        else float("nan")
    )

    q_coeff  = qini_coefficient(y, t, scores)
    auuc_val = auuc(y, t, scores)

    # ECE on baseline (control) outcome model
    ctrl_proba = np.array(model.predict_control_proba(X), dtype=float)
    ctrl_y     = y[t == 0]
    ctrl_p     = ctrl_proba[t == 0]
    ece_val    = _ece(ctrl_y, ctrl_p)

    def _r(v, d=4):
        return round(float(v), d) if not (v is None or np.isnan(v)) else None

    return {
        "n_holdout":    n_total,
        "n_treated":    n_treat,
        "n_control":    n_ctrl,
        "ate_pred":     _r(ate_pred),
        "ate_actual":   _r(ate_actual),
        "ate_bias":     _r(ate_pred - ate_actual) if ate_actual is not None and not np.isnan(ate_actual) else None,
        "qini":         _r(q_coeff),
        "auuc":         _r(auuc_val),
        "baseline_ece": _r(ece_val),
        "interpretation": _interpret(q_coeff, auuc_val),
    }


def _interpret(qini: float, auuc_val: float) -> str:
    if qini is None or np.isnan(qini):
        return "insufficient data"
    if qini < 0.02:
        return "weak/no signal — model performs near random"
    if qini < 0.08:
        return "moderate uplift model — usable but room for improvement"
    if qini < 0.20:
        return "strong uplift model — clear targeting signal"
    return "excellent uplift model"
