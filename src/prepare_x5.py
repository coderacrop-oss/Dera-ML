"""
prepare_x5.py
-------------
Feature engineering pipeline for X5 RetailHero uplift dataset.

Builds one row per client with RFM features derived from purchase history
(purchases.csv.gz), then joins demographics (clients.csv.gz) and
treatment/target labels (uplift_train.csv.gz).

X5 RetailHero campaign context
-------------------------------
The experiment ran in early 2019.  The full purchase history covers
2018-11-22 through 2019-03-18.  We treat all transactions as pre-campaign
history (the outcome `target` is the binary purchase response to the SMS
campaign, not re-observable from purchases.csv.gz).  Recency is measured
relative to the max transaction date in the loaded data.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "x5_retailhero")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_gender(val: str | float) -> float:
    """
    Encode gender as numeric.
      F → 0  (female)
      M → 1  (male)
      U → 2  (unknown / unspecified)
      NaN → NaN
    """
    if pd.isna(val):
        return float("nan")
    mapping = {"F": 0.0, "M": 1.0, "U": 2.0}
    return mapping.get(str(val).strip().upper(), 2.0)


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------

def _build_rfm_features(purchases: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-client RFM features from the transaction-level purchases frame.

    Input columns expected (all present in purchases.csv.gz):
        client_id, transaction_id, transaction_datetime, purchase_sum, store_id

    Returns DataFrame indexed by client_id with columns:
        recency_days, frequency, monetary, avg_basket,
        weekly_visit_rate, n_stores_visited, purchase_regularity
    """
    # Parse datetime once
    purchases = purchases.copy()
    purchases["dt"] = pd.to_datetime(
        purchases["transaction_datetime"], errors="coerce"
    )

    # Reference date: latest transaction date in the loaded slice
    ref_date = purchases["dt"].max()

    # Collapse to one row per (client_id, transaction_id) to get basket-level data.
    # purchase_sum is constant within a transaction_id, so take first.
    baskets = (
        purchases.groupby(["client_id", "transaction_id"])
        .agg(
            dt=("dt", "first"),
            basket_sum=("purchase_sum", "first"),
            store_id=("store_id", "first"),
        )
        .reset_index()
    )

    # ---- Recency -----------------------------------------------------------
    last_dt = (
        baskets.groupby("client_id")["dt"]
        .max()
        .rename("last_dt")
    )
    recency_days = (
        (ref_date - last_dt)
        .dt.total_seconds()
        .div(86400)
        .clip(lower=0)
        .rename("recency_days")
    )

    # ---- Frequency (number of transactions) --------------------------------
    frequency = (
        baskets.groupby("client_id")["transaction_id"]
        .nunique()
        .rename("frequency")
        .astype(float)
    )

    # ---- Monetary (total spend) --------------------------------------------
    monetary = (
        baskets.groupby("client_id")["basket_sum"]
        .sum()
        .rename("monetary")
    )

    # ---- N stores visited --------------------------------------------------
    n_stores = (
        baskets.groupby("client_id")["store_id"]
        .nunique()
        .rename("n_stores_visited")
        .astype(float)
    )

    # ---- Observation window in weeks (for weekly_visit_rate) ---------------
    first_dt = (
        baskets.groupby("client_id")["dt"]
        .min()
        .rename("first_dt")
    )
    weeks_active = (
        (last_dt - first_dt)
        .dt.total_seconds()
        .div(86400 * 7)
        .clip(lower=1.0 / 7)   # at least 1 day expressed as fraction of week
        .rename("weeks_active")
    )

    # ---- Purchase regularity (std of inter-purchase days) ------------------
    # Sort baskets per client, compute day gaps, take std.
    # Clients with only 1 transaction get NaN → fill with 0.
    def _regularity(group: pd.DataFrame) -> float:
        sorted_dts = group["dt"].sort_values()
        if len(sorted_dts) < 2:
            return 0.0
        gaps = sorted_dts.diff().dropna().dt.total_seconds() / 86400
        return float(gaps.std(ddof=0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        purchase_regularity = (
            baskets.groupby("client_id")
            .apply(_regularity)
            .rename("purchase_regularity")
        )

    # ---- Assemble ----------------------------------------------------------
    rfm = pd.DataFrame(
        {
            "recency_days": recency_days,
            "frequency": frequency,
            "monetary": monetary,
            "n_stores_visited": n_stores,
            "weeks_active": weeks_active,
        }
    )

    rfm = rfm.join(purchase_regularity, how="left")
    rfm["purchase_regularity"] = rfm["purchase_regularity"].fillna(0.0)

    # Derived features
    rfm["avg_basket"] = np.where(
        rfm["frequency"] > 0,
        rfm["monetary"] / rfm["frequency"],
        0.0,
    )
    rfm["weekly_visit_rate"] = rfm["frequency"] / rfm["weeks_active"].clip(lower=1e-3)

    rfm = rfm.drop(columns=["weeks_active"])
    return rfm.reset_index()  # client_id becomes a column


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_x5(sample_clients: int | None = None) -> pd.DataFrame:
    """
    Returns DataFrame with one row per client:

        client_id, treatment, target, + RFM features from purchases

    Feature columns
    ---------------
    recency_days        : days since last purchase (relative to max date in data)
    frequency           : number of distinct transactions
    monetary            : total spend (purchase_sum)
    avg_basket          : monetary / frequency
    weekly_visit_rate   : frequency / weeks between first and last purchase
    n_stores_visited    : number of unique stores visited
    purchase_regularity : std of inter-purchase days (0 = single visit or perfect regularity)
    age                 : client age (from clients.csv.gz, NaN if missing)
    gender_enc          : F→0, M→1, U→2 (NaN if missing)

    Parameters
    ----------
    sample_clients : int or None
        If given, randomly sample this many clients from uplift_train before
        building features (useful for fast iteration).  Sampling is done on
        client_id before any join, so the heavy purchase load is filtered early.

    Returns
    -------
    pd.DataFrame  — one row per client
    """
    # ---- Load labels -------------------------------------------------------
    uplift_path = os.path.join(_DATA_DIR, "uplift_train.csv.gz")
    labels = pd.read_csv(uplift_path)
    labels = labels.rename(columns={"treatment_flg": "treatment"})

    if sample_clients is not None and sample_clients < len(labels):
        labels = labels.sample(n=sample_clients, random_state=42).reset_index(drop=True)

    client_set = set(labels["client_id"].unique())

    # ---- Load clients (demographics) ---------------------------------------
    clients_path = os.path.join(_DATA_DIR, "clients.csv.gz")
    clients = pd.read_csv(
        clients_path,
        usecols=["client_id", "age", "gender"],
    )
    clients = clients[clients["client_id"].isin(client_set)].copy()
    clients["gender_enc"] = clients["gender"].apply(_encode_gender)
    clients = clients.drop(columns=["gender"])

    # ---- Load purchases (filter to relevant clients early) -----------------
    purchases_path = os.path.join(_DATA_DIR, "purchases.csv.gz")
    chunks = []
    for chunk in pd.read_csv(
        purchases_path,
        usecols=[
            "client_id",
            "transaction_id",
            "transaction_datetime",
            "purchase_sum",
            "store_id",
        ],
        chunksize=500_000,
    ):
        filtered = chunk[chunk["client_id"].isin(client_set)]
        if not filtered.empty:
            chunks.append(filtered)

    if chunks:
        purchases = pd.concat(chunks, ignore_index=True)
    else:
        # No purchase history for sampled clients — return labels + demographics only
        purchases = pd.DataFrame(
            columns=["client_id", "transaction_id", "transaction_datetime",
                     "purchase_sum", "store_id"]
        )

    # ---- Build RFM features ------------------------------------------------
    if len(purchases) > 0:
        rfm = _build_rfm_features(purchases)
    else:
        rfm = pd.DataFrame(columns=["client_id", "recency_days", "frequency",
                                     "monetary", "n_stores_visited",
                                     "purchase_regularity", "avg_basket",
                                     "weekly_visit_rate"])

    # ---- Join everything ---------------------------------------------------
    df = labels.merge(clients, on="client_id", how="left")
    df = df.merge(rfm, on="client_id", how="left")

    # Fill RFM nulls for clients with no purchase history in the loaded data.
    rfm_cols = [
        "recency_days", "frequency", "monetary", "avg_basket",
        "weekly_visit_rate", "n_stores_visited", "purchase_regularity",
    ]
    # Clients with no history: set recency to max observed, everything else 0
    if "recency_days" in df.columns:
        max_recency = df["recency_days"].max()
        df["recency_days"] = df["recency_days"].fillna(
            max_recency if pd.notna(max_recency) else 0.0
        )
    for col in rfm_cols:
        if col in df.columns and col != "recency_days":
            df[col] = df[col].fillna(0.0)

    # Column order: id, labels, RFM, demographics
    ordered_cols = (
        ["client_id", "treatment", "target"]
        + rfm_cols
        + ["age", "gender_enc"]
    )
    # Keep only columns that actually exist
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df[ordered_cols].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare X5 RetailHero uplift dataset.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of clients to sample (default: all).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output CSV path.",
    )
    args = parser.parse_args()
    df = prepare_x5(sample_clients=args.sample)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Treatment dist: {df['treatment'].value_counts().to_dict()}")
    nan_counts = df.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        print(f"NaN counts: {nan_counts.to_dict()}")
    print(df.describe().round(2).to_string())
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")
