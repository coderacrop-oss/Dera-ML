"""
feature_engineering.py
======================
Reusable feature engineering module for uplift modeling on restaurant/retail POS data.

Entry point:
    extract_features(df, ref_date=None, min_transactions=2) -> pd.DataFrame

Feature groups produced (one row per customer_id):
    1. RFM          — recency_days, frequency, monetary, avg_transaction
    2. Behavioral   — visit_regularity, preferred_hour, peak_ratio,
                      weekend_ratio, category_entropy, avg_items_per_visit
    3. Promo        — ever_used_promo, promo_frequency, promo_types_used,
                      avg_spend_on_promo, avg_spend_no_promo, promo_spend_lift
    4. Loyalty      — customer_age_days, repeat_rate, tenure_frequency
    5. Segment      — rfm_decile
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    df: pd.DataFrame,
    ref_date: str | pd.Timestamp | None = None,
    min_transactions: int = 2,
) -> pd.DataFrame:
    """Return a customer-level feature DataFrame from raw POS transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Raw POS transactions.  Must contain at minimum:
        customer_id, transaction_id, timestamp, total_price, quantity.
        Optional columns used when present:
        category, promo_type.
    ref_date : str or Timestamp, optional
        Reference date for recency/age calculations.
        Defaults to max(timestamp) + 1 day.
    min_transactions : int, default 2
        Minimum number of transactions a customer must have to be included.

    Returns
    -------
    pd.DataFrame
        One row per customer_id, all values float32, no NaNs.
        Outliers clipped at the 99th-percentile of each column.
    """
    df = _prepare(df)

    # Reference date
    if ref_date is None:
        ref_date = df["timestamp"].max() + pd.Timedelta(days=1)
    else:
        ref_date = pd.Timestamp(ref_date)

    # Filter customers with enough transactions
    tx_counts = df.groupby("customer_id")["transaction_id"].count()
    keep = tx_counts[tx_counts >= min_transactions].index
    df = df[df["customer_id"].isin(keep)].copy()

    # Build feature groups
    feat = _rfm(df, ref_date)
    feat = feat.join(_behavioral(df), how="left")
    feat = feat.join(_promo_sensitivity(df), how="left")
    feat = feat.join(_loyalty(df, ref_date), how="left")
    feat = feat.join(_segment(feat), how="left")

    # Post-processing: fill NaN → 0, clip 99th pctile, cast float32
    feat = feat.fillna(0.0)
    feat = _clip_outliers(feat)
    feat = feat.astype(np.float32)

    return feat.reset_index()


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["customer_id", "timestamp"])
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df = df[df["customer_id"] != ""]
    df = df[df["total_price"] > 0]

    # Optional columns — create with sensible defaults if absent
    if "category" not in df.columns:
        df["category"] = "unknown"
    if "promo_type" not in df.columns:
        df["promo_type"] = "none"
    if "quantity" not in df.columns:
        df["quantity"] = 1.0

    df["promo_type"] = df["promo_type"].fillna("none").astype(str).str.lower().str.strip()
    df["category"] = df["category"].fillna("unknown").astype(str)

    # Derived time columns
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek   # Mon=0, Sun=6
    df["date"] = df["timestamp"].dt.date

    return df


# ---------------------------------------------------------------------------
# 1. RFM
# ---------------------------------------------------------------------------

def _rfm(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    g = df.groupby("customer_id")

    recency = (ref_date - g["timestamp"].max()).dt.days.rename("recency_days")
    frequency = g["transaction_id"].count().rename("frequency")
    monetary = g["total_price"].sum().rename("monetary")
    avg_transaction = g["total_price"].mean().rename("avg_transaction")

    return pd.concat([recency, frequency, monetary, avg_transaction], axis=1)


# ---------------------------------------------------------------------------
# 2. Behavioral patterns
# ---------------------------------------------------------------------------

def _behavioral(df: pd.DataFrame) -> pd.DataFrame:
    records: dict[str, pd.Series] = {}

    # visit_regularity: std / mean of days between successive visits (per customer)
    # Lower value = more regular
    def _regularity(dates: pd.Series) -> float:
        sorted_dates = pd.to_datetime(sorted(dates.unique()))
        if len(sorted_dates) < 2:
            return 0.0
        gaps = np.diff(sorted_dates.astype(np.int64)) / 86_400_000_000_000  # ns → days
        m = gaps.mean()
        if m == 0:
            return 0.0
        return float(gaps.std() / m)

    records["visit_regularity"] = (
        df.groupby("customer_id")["timestamp"].apply(_regularity)
    )

    # preferred_hour: mode of transaction hour
    records["preferred_hour"] = (
        df.groupby("customer_id")["hour"]
        .agg(lambda h: float(h.mode().iloc[0]))
    )

    # peak_ratio: % transactions during peak hours 17-21
    df["_is_peak"] = df["hour"].between(17, 21).astype(float)
    records["peak_ratio"] = df.groupby("customer_id")["_is_peak"].mean()

    # weekend_ratio: % transactions on Sat (5) or Sun (6)
    df["_is_weekend"] = df["dayofweek"].isin([5, 6]).astype(float)
    records["weekend_ratio"] = df.groupby("customer_id")["_is_weekend"].mean()

    # category_entropy: Shannon entropy over spend distribution by category
    def _cat_entropy(sub: pd.DataFrame) -> float:
        spend_by_cat = sub.groupby("category")["total_price"].sum()
        total = spend_by_cat.sum()
        if total == 0 or len(spend_by_cat) < 2:
            return 0.0
        probs = spend_by_cat / total
        return float(scipy_entropy(probs))

    records["category_entropy"] = (
        df.groupby("customer_id").apply(_cat_entropy)
    )

    # avg_items_per_visit: mean quantity per transaction_id
    tx_qty = df.groupby(["customer_id", "transaction_id"])["quantity"].sum().reset_index()
    records["avg_items_per_visit"] = (
        tx_qty.groupby("customer_id")["quantity"].mean()
    )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Promo sensitivity
# ---------------------------------------------------------------------------

def _promo_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    records: dict[str, pd.Series] = {}

    is_promo = df["promo_type"] != "none"

    # ever_used_promo
    records["ever_used_promo"] = (
        df.groupby("customer_id")["promo_type"]
        .apply(lambda s: float((s != "none").any()))
    )

    # promo_frequency: # promo rows / total rows
    records["promo_frequency"] = (
        df.assign(_promo=(is_promo.astype(float)))
        .groupby("customer_id")["_promo"].mean()
    )

    # promo_types_used: count of distinct non-'none' promo types
    records["promo_types_used"] = (
        df[is_promo]
        .groupby("customer_id")["promo_type"]
        .nunique()
        .reindex(df["customer_id"].unique(), fill_value=0.0)
    )

    # avg_spend_on_promo
    promo_spend = (
        df[is_promo]
        .groupby("customer_id")["total_price"]
        .mean()
    )
    # avg_spend_no_promo
    no_promo_spend = (
        df[~is_promo]
        .groupby("customer_id")["total_price"]
        .mean()
    )

    all_customers = df["customer_id"].unique()
    records["avg_spend_on_promo"] = promo_spend.reindex(all_customers, fill_value=0.0)
    records["avg_spend_no_promo"] = no_promo_spend.reindex(all_customers, fill_value=0.0)

    # promo_spend_lift = avg_spend_on_promo / avg_spend_no_promo, clipped [0.5, 3.0]
    # Where no_promo_spend is 0 (customer only ever transacted with promos), default to 1.0
    combined = pd.DataFrame({
        "on":  records["avg_spend_on_promo"],
        "off": records["avg_spend_no_promo"],
    })
    # Only compute lift where both sides are > 0
    lift = np.where(
        (combined["on"] > 0) & (combined["off"] > 0),
        combined["on"] / combined["off"],
        1.0,   # neutral lift as default
    )
    records["promo_spend_lift"] = pd.Series(
        np.clip(lift, 0.5, 3.0), index=combined.index
    )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Loyalty signals
# ---------------------------------------------------------------------------

def _loyalty(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    records: dict[str, pd.Series] = {}

    first_tx = df.groupby("customer_id")["timestamp"].min()
    customer_age_days = (ref_date - first_tx).dt.days.rename("customer_age_days")

    frequency = df.groupby("customer_id")["transaction_id"].count()

    # repeat_rate at customer level: 1 if they visited more than once, else 0
    repeat_rate = (frequency > 1).astype(float).rename("repeat_rate")

    # tenure_frequency: monthly visit rate = frequency / customer_age_days * 30
    # Guard against zero-age customers (same day first and last tx)
    age_safe = customer_age_days.clip(lower=1)
    tenure_frequency = (frequency / age_safe * 30).rename("tenure_frequency")

    records["customer_age_days"] = customer_age_days
    records["repeat_rate"] = repeat_rate
    records["tenure_frequency"] = tenure_frequency

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Segment encoding
# ---------------------------------------------------------------------------

def _segment(feat: pd.DataFrame) -> pd.DataFrame:
    """rfm_decile: 0-9 based on qcut of monetary."""
    monetary = feat["monetary"]
    # pd.qcut fails when all values are identical — use rank to break ties
    try:
        rfm_decile = pd.qcut(monetary.rank(method="first"), 10, labels=False).astype(float)
    except ValueError:
        rfm_decile = pd.Series(0.0, index=monetary.index)
    return pd.DataFrame({"rfm_decile": rfm_decile})


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _clip_outliers(df: pd.DataFrame, upper_quantile: float = 0.99) -> pd.DataFrame:
    """Clip each numeric column at its 99th-percentile value (lower bound = 0)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude binary / bounded columns from upper clipping
    skip_clip = {"ever_used_promo", "repeat_rate", "peak_ratio", "weekend_ratio",
                 "promo_frequency", "rfm_decile", "preferred_hour"}
    for col in numeric_cols:
        if col in skip_clip:
            continue
        upper = df[col].quantile(upper_quantile)
        if upper > 0:
            df[col] = df[col].clip(upper=upper)
    return df
