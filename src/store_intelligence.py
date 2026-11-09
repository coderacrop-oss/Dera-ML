"""store_intelligence.py — Store-level analytics from anonymous transaction data.

Works with 100 % anonymous MeriPOS data (no customer_id required).
Produces a StoreProfile that is attached to the serving context and used
by the campaign pipeline as store-level features.

Usage
-----
    from ingest import ingest_meripos
    from store_intelligence import build_store_profile

    tx_df, item_df = ingest_meripos("data/transactions.csv")
    profile = build_store_profile(tx_df, item_df)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from business_config import CONFIG
from peak_hours import derive_from_timestamps


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class StoreProfile:
    """Store-level analytics computed from anonymous transaction data."""

    peak_hours: list[int]
    off_peak_hours: list[int]

    # Top-selling items (by total qty)
    top_items: list[dict]           # [{item_name, avg_daily_qty, share_pct}]

    # Basket trends
    basket_avg: float               # average net_sales per bill (overall)
    basket_avg_trend: float         # (last-30% mean − first-30% mean) / first-30% mean

    # Channel breakdown
    channel_mix: dict[str, float]   # e.g. {"grab": 0.35, "in_store": 0.65}

    # Discount signal
    discount_depth: float           # total_discounts / total_gross_sales  (0–1)

    # Promo lift proxy (Mode C signal)
    promo_lift_proxy: float         # (avg basket on discount days − normal days) / normal days

    # Meta
    n_transactions: int
    date_range_days: int
    tz: str = "Asia/Bangkok"

    # Optional: quality gate result attached by caller
    quality: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_store_profile(
    tx_df: pd.DataFrame,
    item_df: pd.DataFrame | None = None,
    tz: str | None = None,
    top_n_items: int = 10,
) -> StoreProfile:
    """Build a StoreProfile from MeriPOS transaction data (anonymous OK).

    Parameters
    ----------
    tx_df       : normalised MeriPOS transaction DataFrame (from ingest_meripos).
                  Expected columns: timestamp, net_sales, gross_sales,
                  discount_amount, channel.  All optional — missing cols get
                  sensible defaults.
    item_df     : item-level DataFrame (from parse_meripos_items), optional.
    tz          : store timezone (default: CONFIG["business_tz"]).
    top_n_items : number of top items to include.
    """
    tz = tz or CONFIG.get("business_tz", "Asia/Bangkok")

    # -- Peak hours -------------------------------------------------------
    if "timestamp" in tx_df.columns and len(tx_df) >= 10:
        ts = pd.to_datetime(tx_df["timestamp"], errors="coerce").dropna()
        if len(ts) >= 10:
            peak, off_peak = derive_from_timestamps(ts, tz)
        else:
            peak, off_peak = _default_hours()
    else:
        peak, off_peak = _default_hours()

    # -- Top items --------------------------------------------------------
    top_items: list[dict] = []
    if item_df is not None and not item_df.empty and "item_name" in item_df.columns:
        item_df = item_df.copy()
        item_df["qty"] = pd.to_numeric(item_df.get("qty", 1), errors="coerce").fillna(1.0)
        sales = item_df.groupby("item_name")["qty"].sum().sort_values(ascending=False)
        total_qty = sales.sum() or 1.0

        ts_valid: pd.Series | None = None
        if "timestamp" in item_df.columns:
            ts_valid = pd.to_datetime(item_df["timestamp"], errors="coerce").dropna()

        n_days = max(1, int((ts_valid.max() - ts_valid.min()).days + 1)) if (
            ts_valid is not None and len(ts_valid) >= 2
        ) else 30

        for item_name, total in sales.head(top_n_items).items():
            top_items.append({
                "item_name":     str(item_name),
                "avg_daily_qty": round(float(total) / n_days, 2),
                "share_pct":     round(float(total) / total_qty * 100, 1),
            })

    # -- Basket average & trend -------------------------------------------
    amount_col = _amount_col(tx_df)
    basket_avg = 0.0
    basket_avg_trend = 0.0

    if amount_col and not tx_df.empty:
        amounts = pd.to_numeric(tx_df[amount_col], errors="coerce").dropna()
        basket_avg = round(float(amounts.mean()), 2) if len(amounts) else 0.0

        if "timestamp" in tx_df.columns and len(tx_df) >= 20:
            df_s = tx_df.assign(_amt=pd.to_numeric(tx_df[amount_col], errors="coerce"))
            df_s = df_s.dropna(subset=["_amt"]).sort_values("timestamp")
            cut = max(1, int(len(df_s) * 0.3))
            early = float(df_s["_amt"].iloc[:cut].mean())
            late  = float(df_s["_amt"].iloc[-cut:].mean())
            if early > 0:
                basket_avg_trend = round((late - early) / early, 4)

    # -- Channel mix ------------------------------------------------------
    channel_mix: dict[str, float] = {}
    if "channel" in tx_df.columns and not tx_df.empty:
        counts = tx_df["channel"].fillna("unknown").value_counts()
        total  = len(tx_df) or 1
        channel_mix = {str(k): round(int(v) / total, 4) for k, v in counts.items()}

    # -- Discount depth ---------------------------------------------------
    discount_depth = 0.0
    if "discount_amount" in tx_df.columns and "gross_sales" in tx_df.columns:
        d = pd.to_numeric(tx_df["discount_amount"], errors="coerce").fillna(0).sum()
        g = pd.to_numeric(tx_df["gross_sales"],    errors="coerce").fillna(0).sum()
        if g > 0:
            discount_depth = round(float(d / g), 4)

    # -- Promo lift proxy -------------------------------------------------
    promo_lift_proxy = 0.0
    if amount_col and "discount_amount" in tx_df.columns and "timestamp" in tx_df.columns:
        df_p = tx_df.assign(
            _amt  = pd.to_numeric(tx_df[amount_col],        errors="coerce"),
            _disc = pd.to_numeric(tx_df["discount_amount"], errors="coerce").fillna(0),
            _ts   = pd.to_datetime(tx_df["timestamp"],      errors="coerce"),
        ).dropna(subset=["_amt", "_ts"])

        if len(df_p) >= 20:
            df_p["_date"] = df_p["_ts"].dt.date
            daily = df_p.groupby("_date").agg(
                avg_amt=("_amt",  "mean"),
                disc   =("_disc", "sum"),
            )
            promo_avg  = daily.loc[daily["disc"] > 0,  "avg_amt"].mean()
            normal_avg = daily.loc[daily["disc"] == 0, "avg_amt"].mean()
            if (not math.isnan(float(promo_avg if promo_avg == promo_avg else float("nan")))
                    and normal_avg > 0):
                promo_lift_proxy = round(float((promo_avg - normal_avg) / normal_avg), 4)

    # -- Date range -------------------------------------------------------
    date_range_days = 0
    if "timestamp" in tx_df.columns and not tx_df.empty:
        ts_all = pd.to_datetime(tx_df["timestamp"], errors="coerce").dropna()
        if len(ts_all) >= 2:
            date_range_days = int((ts_all.max() - ts_all.min()).days)

    return StoreProfile(
        peak_hours       = peak,
        off_peak_hours   = off_peak,
        top_items        = top_items,
        basket_avg       = basket_avg,
        basket_avg_trend = basket_avg_trend,
        channel_mix      = channel_mix,
        discount_depth   = discount_depth,
        promo_lift_proxy = promo_lift_proxy,
        n_transactions   = len(tx_df),
        date_range_days  = date_range_days,
        tz               = tz,
    )


# ---------------------------------------------------------------------------
# AC-4.1 — Store Embedding (38-dim deterministic vector)
# ---------------------------------------------------------------------------

def get_store_embedding(profile: StoreProfile) -> np.ndarray:
    """Return a deterministic 38-dim numpy float32 array summarising StoreProfile.

    Dimensions
    ----------
    [0:24]  peak_hour_pattern    — normalised hourly fraction (sum to 1)
    [24:31] day_of_week_pattern  — normalised daily fraction (sum to 1)
    [31:36] top_category_mix     — top-5 item share (padded with 0 if < 5 items)
    [36]    discount_depth       — total_discounts / total_gross_sales (0–1 clipped)
    [37]    promo_lift_proxy     — (promo day basket − normal basket) / normal (clipped ±1)

    Total: 24 + 7 + 5 + 1 + 1 = 38 dimensions.

    Deterministic: same StoreProfile input always produces same output.
    Returns np.ndarray of shape (38,) dtype float32.
    """
    vec = np.zeros(38, dtype=np.float32)

    # --- [0:24] Peak hour pattern ---
    hour_counts = np.zeros(24, dtype=np.float32)
    # Mark peak hours with weight 2, off-peak with 1, rest 0
    for h in (profile.peak_hours or []):
        if 0 <= h < 24:
            hour_counts[h] += 2.0
    for h in (profile.off_peak_hours or []):
        if 0 <= h < 24:
            hour_counts[h] += 1.0
    total_h = hour_counts.sum()
    vec[0:24] = hour_counts / total_h if total_h > 0 else hour_counts

    # --- [24:31] Day-of-week pattern from channel_mix or uniform fallback ---
    # We don't store daily pattern in StoreProfile v1; derive proxy from
    # peak_hours length as a signal (longer peak ↔ more weekend traffic)
    # and discount_depth (high discount ↔ weekend promotion days).
    # This is a placeholder until tx_df day-of-week data is stored in profile.
    n_peak = len(profile.peak_hours or [])
    dow_weights = np.ones(7, dtype=np.float32)
    # Slight weekend boost proportional to discount depth
    weekend_boost = min(1.0, profile.discount_depth * 3)
    dow_weights[5] += weekend_boost  # Saturday
    dow_weights[6] += weekend_boost  # Sunday
    # Slight midweek boost proportional to n_peak hours
    midweek_boost = min(0.5, n_peak / 20.0)
    dow_weights[1] += midweek_boost  # Tuesday
    dow_weights[2] += midweek_boost  # Wednesday
    vec[24:31] = dow_weights / dow_weights.sum()

    # --- [31:36] Top-5 item share ---
    items = profile.top_items or []
    for i, item in enumerate(items[:5]):
        share = item.get("share_pct", 0.0)
        vec[31 + i] = float(share) / 100.0  # normalise 0–1

    # --- [36] Discount depth (clipped 0–1) ---
    vec[36] = float(np.clip(profile.discount_depth, 0.0, 1.0))

    # --- [37] Promo lift proxy (clipped -1 to +1) ---
    vec[37] = float(np.clip(profile.promo_lift_proxy, -1.0, 1.0))

    return vec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_hours() -> tuple[list[int], list[int]]:
    return [11, 12, 13, 14, 15], [8, 9, 10, 16, 17, 18]


def _amount_col(df: pd.DataFrame) -> str | None:
    for col in ("net_sales", "total_price", "amount", "Net sales"):
        if col in df.columns:
            return col
    return None
