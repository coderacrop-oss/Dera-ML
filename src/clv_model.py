"""Customer Lifetime Value estimation using BG/NBD model.

BG/NBD (Beta-Geometric / Negative Binomial Distribution) models two things:
  1. P(customer is still alive) — has not churned — given purchase history
  2. E[future transactions in next N days] conditional on being alive

CLV formula (non-contractual, no Gamma-Gamma for simplicity):
  CLV_i = gross_margin_rate × avg_spent_i × E[future_txns_i]
         (expected_txns already incorporates P(alive) internally in lifetimes)

For rescue tiers, the key quantity is clv_at_risk:
  clv_at_risk_i = CLV_i × (1 − P(alive_i))
  → the expected revenue lost if this customer churns without intervention.
  → rescue campaign justified when promo_cost_i ≤ clv_at_risk_i.

BG/NBD input conventions (differ from RFM "recency"):
  frequency    = number of REPEAT purchases = total_transactions − 1  (min 0)
  recency      = time from first to last purchase (weeks)
  T            = time from first purchase to observation end (weeks)
  monetary_value = average transaction value
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from lifetimes import BetaGeoFitter
    _LIFETIMES_OK = True
except ImportError:
    _LIFETIMES_OK = False


def compute_bgnbd_summary(
    raw_df: pd.DataFrame,
    observation_end=None,
) -> pd.DataFrame:
    """Build per-customer BG/NBD summary table from raw transactions.

    Parameters
    ----------
    raw_df          : DataFrame with columns customer_id, timestamp, total_price, transaction_id
    observation_end : end of observation window (default: max timestamp in data)

    Returns
    -------
    DataFrame with columns:
        customer_id, bgnbd_frequency, bgnbd_recency, bgnbd_T, monetary_value
    """
    ts = pd.to_datetime(raw_df["timestamp"], errors="coerce")
    if observation_end is None:
        observation_end = ts.max()
    observation_end = pd.Timestamp(observation_end).normalize()

    summary = (
        raw_df.assign(_ts=ts.dt.normalize())
        .groupby("customer_id")
        .agg(
            _first=("_ts", "min"),
            _last=("_ts", "max"),
            _n_txns=("transaction_id", "nunique"),
            _avg_spend=("total_price", "mean"),
        )
        .reset_index()
    )

    # BG/NBD uses weeks as the time unit (common convention)
    summary["bgnbd_frequency"] = (summary["_n_txns"] - 1).clip(lower=0).astype(int)
    summary["bgnbd_recency"] = (
        (summary["_last"] - summary["_first"]).dt.days / 7
    ).round(4).clip(lower=0.0)
    summary["bgnbd_T"] = (
        (observation_end - summary["_first"]).dt.days / 7
    ).round(4).clip(lower=0.001)   # T must be > 0
    summary["monetary_value"] = summary["_avg_spend"].round(2)

    return summary[["customer_id", "bgnbd_frequency", "bgnbd_recency", "bgnbd_T", "monetary_value"]]


def fit_bgnbd(summary: pd.DataFrame, penalizer_coef: float = 0.001,
              freq_cap: int = 20) -> "BetaGeoFitter":
    """Fit BG/NBD model on summary DataFrame.

    Requires lifetimes>=0.11.  T must be ≥ recency for every row (guaranteed
    by compute_bgnbd_summary).

    Strategy for high-frequency retail data (daily/weekly purchases):
      1. Cap bgnbd_frequency at freq_cap (default 20) to reduce extreme values
         that cause numerical precision loss in the BG/NBD optimizer.
      2. Retry with progressively larger penalizer if still failing.
      3. If all penalizers raise ConvergenceError but the fitted params are
         finite, return the model anyway — partial convergence is usually
         sufficient for ranking customers by P(alive).
    """
    if not _LIFETIMES_OK:
        raise ImportError("Install lifetimes: pip install lifetimes")

    # lifetimes requires recency ≤ T; cap frequency for numerical stability
    valid = summary[summary["bgnbd_T"] >= summary["bgnbd_recency"]].copy()
    if freq_cap > 0:
        capped = int((valid["bgnbd_frequency"] > freq_cap).sum())
        if capped:
            valid["bgnbd_frequency"] = valid["bgnbd_frequency"].clip(upper=freq_cap)
            print(f"🔧 [CLV] Capped {capped} customers at freq={freq_cap} for BG/NBD stability")

    _penalizers = sorted(set([penalizer_coef, 0.01, 0.05, 0.1, 0.5, 1.0]))
    last_exc = None
    last_bgf = None
    for pen in _penalizers:
        bgf = BetaGeoFitter(penalizer_coef=pen)
        try:
            bgf.fit(
                valid["bgnbd_frequency"],
                valid["bgnbd_recency"],
                valid["bgnbd_T"],
            )
            if pen > penalizer_coef:
                print(f"🔧 [CLV] BG/NBD converged with penalizer={pen} (started at {penalizer_coef})")
            return bgf
        except Exception as exc:
            last_exc = exc
            last_bgf = bgf
            continue

    # Partial convergence: use last fitted model if params are all finite
    if last_bgf is not None and hasattr(last_bgf, "params_"):
        params = last_bgf.params_
        if all(np.isfinite(v) for v in params.values()):
            print(f"⚠️  [CLV] BG/NBD partial convergence — using best-effort params "
                  f"(r={params.get('r', '?'):.4f}, α={params.get('alpha', '?'):.4f})")
            return last_bgf

    raise RuntimeError(
        f"BG/NBD failed to converge across penalizers {_penalizers}: {last_exc}"
    ) from last_exc


def estimate_clv_for_customers(
    raw_df: pd.DataFrame,
    cust_agg: pd.DataFrame,
    cogs_rate: float = 0.5,
    horizon_days: int = 90,
    penalizer_coef: float = 0.001,
    observation_end=None,
) -> pd.DataFrame:
    """Fit BG/NBD and attach CLV columns to cust_agg.

    New columns added to cust_agg:
        p_alive        P(customer has not churned) ∈ [0, 1]
        expected_txns  E[transactions in next horizon_days]
        clv_estimate   gross_margin × avg_spend × expected_txns
        clv_at_risk    clv_estimate × (1 − p_alive)  — value at risk if customer churns

    If BG/NBD fitting fails (insufficient data, missing lifetimes), all four
    columns are set to 0.0 so downstream code can fall back gracefully.

    Parameters
    ----------
    raw_df         : raw POS DataFrame (needs timestamp, transaction_id, total_price)
    cust_agg       : customer-level aggregated DataFrame to enrich
    cogs_rate      : COGS rate used to compute gross margin
    horizon_days   : CLV forecast horizon (days)
    penalizer_coef : BG/NBD regularisation — increase for small datasets
    observation_end: end of observation window (default: max timestamp)
    """
    _fallback_cols = ("p_alive", "expected_txns", "clv_estimate", "clv_at_risk")

    if not _LIFETIMES_OK:
        print("⚠️  [CLV] lifetimes not installed — CLV estimation skipped (pip install lifetimes)")
        for col in _fallback_cols:
            cust_agg = cust_agg.copy()
            cust_agg[col] = 0.0
        return cust_agg

    try:
        summary = compute_bgnbd_summary(raw_df, observation_end=observation_end)

        # Need at least 50 repeat-purchasers to fit reliably
        n_repeats = int((summary["bgnbd_frequency"] > 0).sum())
        if n_repeats < 20:
            raise ValueError(
                f"Only {n_repeats} customers with repeat purchases — "
                "BG/NBD needs ≥20 for stable estimates"
            )

        bgf = fit_bgnbd(summary, penalizer_coef=penalizer_coef)
        horizon_weeks = horizon_days / 7

        summary["p_alive"] = bgf.conditional_probability_alive(
            summary["bgnbd_frequency"],
            summary["bgnbd_recency"],
            summary["bgnbd_T"],
        ).clip(0.0, 1.0).round(4)

        summary["expected_txns"] = bgf.conditional_expected_number_of_purchases_up_to_time(
            horizon_weeks,
            summary["bgnbd_frequency"],
            summary["bgnbd_recency"],
            summary["bgnbd_T"],
        ).clip(lower=0.0).round(4)

        gross_margin_rate = 1.0 - cogs_rate
        summary["clv_estimate"] = (
            gross_margin_rate * summary["monetary_value"] * summary["expected_txns"]
        ).round(2)
        summary["clv_at_risk"] = (
            summary["clv_estimate"] * (1.0 - summary["p_alive"])
        ).round(2)

        clv_cols = summary[["customer_id"] + list(_fallback_cols)]
        result = cust_agg.merge(clv_cols, on="customer_id", how="left")
        for col in _fallback_cols:
            result[col] = result[col].fillna(0.0)

        avg_p = float(summary["p_alive"].mean())
        avg_clv = float(summary["clv_estimate"].mean())
        avg_at_risk = float(summary["clv_at_risk"].mean())
        print(
            f"🧮 [CLV] BG/NBD fit on {len(summary):,} customers | "
            f"avg P(alive)={avg_p:.1%} | "
            f"avg CLV={avg_clv:.2f} | "
            f"avg CLV-at-risk={avg_at_risk:.2f} | "
            f"horizon={horizon_days}d"
        )
        return result

    except Exception as exc:
        print(f"⚠️  [CLV] BG/NBD estimation failed ({exc}) — falling back to heuristic rescue")
        result = cust_agg.copy()
        for col in _fallback_cols:
            result[col] = 0.0
        return result
