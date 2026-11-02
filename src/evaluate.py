"""
evaluate.py — SOTA Uplift Evaluation Suite
Corrected Qini AUUC, bootstrap confidence intervals, policy value,
calibration curve, uplift decile table, Kendall tau rank correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    auuc: float                    # corrected Qini AUUC
    auuc_ci: tuple                 # (lower, upper) 95% bootstrap CI
    auuc_se: float                 # standard error
    policy_value: float            # E[Y] under top-budget_pct targeting
    policy_value_random: float     # E[Y] under random targeting
    policy_lift: float             # policy_value - policy_value_random
    calibration_bins: pd.DataFrame # 10 bins: mean_uplift_pred vs mean_uplift_obs
    decile_table: pd.DataFrame     # 10 deciles: uplift, treatment rate, conversion rate
    kendall_tau: float             # rank correlation between predicted and observed uplift
    qini_curve: tuple              # (fractions, qini_values) for plotting
    summary: str                   # human-readable summary


# ── Core AUUC helpers ──────────────────────────────────────────────────────────

def _corrected_qini_curve(
    uplift: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Corrected Qini curve.

    Returns (fractions, qini_values) where qini_values[k] is the
    incremental gain (above the true random diagonal) achieved by
    targeting the top-k fraction.

    The random baseline is the STRAIGHT LINE from 0 to the overall ATE:
        ATE = E[Y|T=1] - E[Y|T=0]
    scaled by cumulative sample count — NOT (fraction * sum(Y*T)) which
    assumes 50/50 treatment split and bakes in a -0.15 floor bias when
    treatment rate != 0.5.
    """
    n = len(uplift)
    order = np.argsort(uplift)[::-1]
    T_s = T[order]
    Y_s = Y[order]

    N_T = T.sum()
    N_C = n - N_T

    # Overall ATE (used to anchor the random diagonal end-point)
    mu1 = Y[T == 1].mean() if N_T > 0 else 0.0
    mu0 = Y[T == 0].mean() if N_C > 0 else 0.0
    ate = mu1 - mu0                       # end-point of random diagonal

    cum_T = np.cumsum(T_s)
    cum_C = np.cumsum(1 - T_s)
    cum_T_conv = np.cumsum(Y_s * T_s)
    cum_C_conv = np.cumsum(Y_s * (1 - T_s))

    fracs = np.arange(1, n + 1) / n

    # Qini numerator: cum treated conversions - cum control conversions *
    # (cum treated / cum control) — i.e., incremental conversions above
    # the within-sample random rate
    safe_C = np.where(cum_C > 0, cum_C, 1)
    qini = np.where(
        cum_T > 0,
        cum_T_conv - cum_C_conv * (cum_T / safe_C),
        0.0,
    )

    # True random diagonal: straight line 0 → ate * N
    # (what a random policy achieves in absolute conversion units)
    random_end = ate * n
    random = fracs * random_end

    # Subtract random so the area we integrate is purely the model lift
    qini_corrected = qini - random

    fracs_full = np.concatenate([[0.0], fracs])
    qini_full  = np.concatenate([[0.0], qini_corrected])

    return fracs_full, qini_full


def _auuc_from_curve(fracs: np.ndarray, qini: np.ndarray) -> float:
    """Integrate Qini curve and normalize by population size."""
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    n = len(fracs) - 1          # number of data points (excluding origin)
    return trapz(qini, fracs) / max(n, 1)


def _compute_auuc(
    uplift: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
) -> float:
    fracs, qini = _corrected_qini_curve(uplift, T, Y)
    return _auuc_from_curve(fracs, qini)


# ── Bootstrap helper ───────────────────────────────────────────────────────────

def _stratified_resample(
    uplift: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample within treatment / control strata to preserve treatment rate."""
    idx_t = np.where(T == 1)[0]
    idx_c = np.where(T == 0)[0]
    boot_t = rng.choice(idx_t, size=len(idx_t), replace=True)
    boot_c = rng.choice(idx_c, size=len(idx_c), replace=True)
    idx = np.concatenate([boot_t, boot_c])
    return uplift[idx], T[idx], Y[idx]


# ── Policy value ───────────────────────────────────────────────────────────────

def _policy_value(
    uplift: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    budget_pct: float,
) -> Tuple[float, float]:
    """
    Simulate targeting the top budget_pct fraction of individuals
    (those with highest predicted uplift score).

    policy_value  = conversion rate among treated individuals in top-k
    random_value  = overall conversion rate (treatment-agnostic baseline)
    """
    n = len(uplift)
    k = max(1, int(np.round(n * budget_pct)))
    top_idx = np.argsort(uplift)[::-1][:k]

    # Among the top-k, look only at the treated arm
    treated_in_top = T[top_idx] == 1
    n_treated_top = treated_in_top.sum()

    if n_treated_top == 0:
        policy_value = float(Y[T == 1].mean()) if T.sum() > 0 else 0.0
    else:
        policy_value = float(Y[top_idx][treated_in_top].mean())

    # Random baseline: overall treated conversion rate
    random_value = float(Y[T == 1].mean()) if T.sum() > 0 else 0.0

    return policy_value, random_value


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _group_means(Y: np.ndarray, T: np.ndarray) -> Tuple[float, float]:
    """Return (mean_treated, mean_control) for a slice; np.nan when group is empty."""
    t_mask = T == 1
    mu1 = Y[t_mask].mean() if t_mask.any() else np.nan
    mu0 = Y[~t_mask].mean() if (~t_mask).any() else np.nan
    return mu1, mu0


def _safe_diff(a: float, b: float) -> float:
    """a - b, or np.nan if either is nan."""
    return (a - b) if (not np.isnan(a) and not np.isnan(b)) else np.nan


# ── Calibration curve ──────────────────────────────────────────────────────────

def _calibration_curve(
    uplift: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin predictions into n_bins equal-frequency bins.
    For each bin: mean predicted uplift vs observed uplift
    (E[Y|T=1, bin] - E[Y|T=0, bin]).
    """
    order = np.argsort(uplift)
    uplift_s, T_s, Y_s = uplift[order], T[order], Y[order]
    rows = []
    for b, idx in enumerate(np.array_split(np.arange(len(uplift)), n_bins)):
        obs_t, obs_c = _group_means(Y_s[idx], T_s[idx])
        rows.append({
            "bin": b + 1,
            "mean_uplift_pred": uplift_s[idx].mean(),
            "mean_uplift_obs": _safe_diff(obs_t, obs_c),
            "n": len(idx),
            "n_treated": int((T_s[idx] == 1).sum()),
            "n_control": int((T_s[idx] == 0).sum()),
        })
    return pd.DataFrame(rows)


# ── Decile table ───────────────────────────────────────────────────────────────

def _decile_table(
    uplift: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Sort by predicted uplift descending, split into n_deciles equal groups.
    For each decile: n, treatment_rate, conversion_treated,
    conversion_control, net_uplift.
    """
    order = np.argsort(uplift)[::-1]
    uplift_s, T_s, Y_s = uplift[order], T[order], Y[order]
    rows = []
    for d, idx in enumerate(np.array_split(np.arange(len(uplift)), n_deciles)):
        conv_t, conv_c = _group_means(Y_s[idx], T_s[idx])
        net = _safe_diff(conv_t, conv_c)
        rows.append({
            "decile": d + 1,
            "n": len(idx),
            "mean_pred_uplift": round(uplift_s[idx].mean(), 6),
            "treatment_rate": round((T_s[idx] == 1).mean(), 4),
            "conversion_treated": round(conv_t, 4) if not np.isnan(conv_t) else np.nan,
            "conversion_control": round(conv_c, 4) if not np.isnan(conv_c) else np.nan,
            "net_uplift": round(net, 4) if not np.isnan(net) else np.nan,
        })
    return pd.DataFrame(rows)


# ── Kendall tau ────────────────────────────────────────────────────────────────

def _kendall_tau_deciles(decile_df: pd.DataFrame) -> float:
    """
    Rank correlation between predicted uplift rank (decile order) and
    observed net_uplift rank.
    """
    valid = decile_df.dropna(subset=["net_uplift"])
    if len(valid) < 2:
        return float("nan")
    tau, _ = kendalltau(valid["decile"].values, valid["net_uplift"].values)
    return float(tau)


# ── Summary string ─────────────────────────────────────────────────────────────

def _build_summary(
    auuc: float,
    auuc_ci: tuple,
    auuc_se: float,
    policy_lift: float,
    budget_pct: float,
    decile_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    tau: float,
) -> str:
    lo, hi = auuc_ci
    pct_str = f"{budget_pct * 100:.0f}%"

    top_uplift = decile_df.iloc[0]["net_uplift"]
    bot_uplift = decile_df.iloc[-1]["net_uplift"]

    top_str = f"{top_uplift * 100:+.1f}%" if not np.isnan(top_uplift) else "N/A"
    bot_str = f"{bot_uplift * 100:+.1f}%" if not np.isnan(bot_uplift) else "N/A"

    # Calibration MAE
    valid_calib = calib_df.dropna(subset=["mean_uplift_obs"])
    if len(valid_calib) > 0:
        cal_mae = float(np.abs(valid_calib["mean_uplift_pred"] - valid_calib["mean_uplift_obs"]).mean())
        cal_str = f"{cal_mae:.3f}"
    else:
        cal_str = "N/A"

    tau_str = f"{tau:.2f}" if not np.isnan(tau) else "N/A"

    _interp_map = {
        (True,  True):  f"Model shows positive and statistically reliable signal. Safe to target top {pct_str}.",
        (True,  False): f"Model shows positive but uncertain signal. Recommend targeting top {pct_str} only when CI lower bound > 0.",
        (False, True):  "Model shows no reliable uplift. CI crosses zero — investigate feature quality or treatment mechanism.",
        (False, False): "Model shows negative AUUC. Predicted rankings may be inversely correlated with true uplift.",
    }
    interp = _interp_map[auuc > 0, lo > 0 if auuc > 0 else hi > 0]

    return (
        f"AUUC: {auuc:+.4f} +/- {auuc_se:.4f} (95% CI: [{lo:+.4f}, {hi:+.4f}])\n"
        f"Policy lift at {pct_str} targeting: {policy_lift * 100:+.2f}pp vs random\n"
        f"Top decile uplift: {top_str} | Bottom decile: {bot_str}\n"
        f"Calibration MAE: {cal_str} | Kendall tau: {tau_str}\n"
        f"Interpretation: {interp}"
    )


# ── Main evaluator class ───────────────────────────────────────────────────────

class UpliftEvaluator:
    """
    SOTA uplift model evaluation suite.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations for CI estimation (default 200).
    random_state : int
        Seed for reproducible bootstrap sampling (default 42).
    """

    def __init__(self, n_bootstrap: int = 200, random_state: int = 42) -> None:
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def evaluate(
        self,
        uplift_scores: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        budget_pct: float = 0.3,
    ) -> EvalReport:
        """
        Run the full evaluation suite.

        Parameters
        ----------
        uplift_scores : array-like, shape (n,)
            Predicted CATE / uplift scores.
        T : array-like, shape (n,)
            Binary treatment indicator (1 = treated, 0 = control).
        Y : array-like, shape (n,)
            Binary outcome.
        budget_pct : float
            Fraction of the population to target for policy value calculation.

        Returns
        -------
        EvalReport
        """
        uplift = np.asarray(uplift_scores, dtype=float)
        T      = np.asarray(T, dtype=int)
        Y      = np.asarray(Y, dtype=float)

        # ── 1. Corrected AUUC ─────────────────────────────────────────────────
        fracs, qini = _corrected_qini_curve(uplift, T, Y)
        auuc = _auuc_from_curve(fracs, qini)

        # ── 2. Bootstrap CI ───────────────────────────────────────────────────
        rng = np.random.default_rng(self.random_state)
        auuc_samples = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            u_b, t_b, y_b = _stratified_resample(uplift, T, Y, rng)
            auuc_samples[i] = _compute_auuc(u_b, t_b, y_b)

        if self.n_bootstrap > 0:
            ci = tuple(np.percentile(auuc_samples, [2.5, 97.5]))
            auuc_se = float(auuc_samples.std(ddof=1))
        else:
            ci = (float("nan"), float("nan"))
            auuc_se = float("nan")

        # ── 3. Policy value ───────────────────────────────────────────────────
        policy_value, policy_value_random = _policy_value(uplift, T, Y, budget_pct)
        policy_lift = policy_value - policy_value_random

        # ── 4. Calibration curve ──────────────────────────────────────────────
        calibration_bins = _calibration_curve(uplift, T, Y, n_bins=10)

        # ── 5. Decile table ───────────────────────────────────────────────────
        decile_table = _decile_table(uplift, T, Y, n_deciles=10)

        # ── 6. Kendall tau ────────────────────────────────────────────────────
        tau = _kendall_tau_deciles(decile_table)

        # ── 7. Summary ────────────────────────────────────────────────────────
        summary = _build_summary(
            auuc=auuc,
            auuc_ci=ci,
            auuc_se=auuc_se,
            policy_lift=policy_lift,
            budget_pct=budget_pct,
            decile_df=decile_table,
            calib_df=calibration_bins,
            tau=tau,
        )

        return EvalReport(
            auuc=auuc,
            auuc_ci=ci,
            auuc_se=auuc_se,
            policy_value=policy_value,
            policy_value_random=policy_value_random,
            policy_lift=policy_lift,
            calibration_bins=calibration_bins,
            decile_table=decile_table,
            kendall_tau=tau,
            qini_curve=(fracs, qini),
            summary=summary,
        )
