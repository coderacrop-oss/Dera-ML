"""
prepare_dunnhumby.py
--------------------
Feature engineering pipeline for Dunnhumby uplift training.

Propensity note: When training the propensity model P(T=1|X) downstream,
clip predicted propensities to [0.1, 0.9]. Treated households spent ~4.7x
more pre-campaign than controls, so the propensity model will assign extreme
scores to high-value shoppers without clipping — compressing the CATE
estimate entirely toward the treated-side tau_1 learner.
"""

from __future__ import annotations

import os
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dunnhumby")

# Default single-campaign: Campaign 26 (TypeA, days 224-264).
# Confirmed real effect in retrain_plan.md: ~$7 incremental spend, ~5%
# incremental conversion rate.
DEFAULT_CAMPAIGN_ID = 26


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tables() -> dict[str, pd.DataFrame]:
    """Load all raw Dunnhumby CSVs into a dict keyed by table name."""
    tables = {}
    for name in (
        "campaign_desc",
        "campaign_table",
        "transaction_data",
        "product",
        "coupon_redempt",
        "hh_demographic",
    ):
        path = os.path.join(_DATA_DIR, f"{name}.csv")
        tables[name] = pd.read_csv(path)
    return tables


def _encode_age(val: str | float) -> float:
    """Ordinal encode AGE_DESC.  NaN → NaN."""
    mapping = {
        "19-24": 0,
        "25-34": 1,
        "35-44": 2,
        "45-54": 3,
        "55-64": 4,
        "65+": 5,
    }
    if pd.isna(val):
        return float("nan")
    return float(mapping.get(str(val).strip(), float("nan")))


def _encode_income(val: str | float) -> float:
    """Ordinal encode INCOME_DESC.  NaN → NaN."""
    mapping = {
        "Under 15K": 0,
        "15-24K": 1,
        "25-34K": 2,
        "35-49K": 3,
        "50-74K": 4,
        "75-99K": 5,
        "100-124K": 6,
        "125-149K": 7,
        "150-174K": 8,
        "175-199K": 9,
        "200-249K": 10,
        "250K+": 11,
    }
    if pd.isna(val):
        return float("nan")
    return float(mapping.get(str(val).strip(), float("nan")))


def _encode_hh_size(val) -> float:
    """HOUSEHOLD_SIZE_DESC is already numeric (1,2,3,4,5+).  NaN → NaN."""
    if pd.isna(val):
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def _department_entropy(spend_series: pd.Series) -> float:
    """Shannon entropy of spend share across departments."""
    total = spend_series.sum()
    if total <= 0:
        return 0.0
    p = spend_series / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------

def _build_features_for_campaign(
    campaign_row: pd.Series,
    tx: pd.DataFrame,
    product: pd.DataFrame,
    coupon_redempt: pd.DataFrame,
    treated_hh: set[int],
    all_control_hh: set[int],
    le_dept: LabelEncoder,
) -> pd.DataFrame:
    """
    Build one row per household for a single campaign episode.

    Parameters
    ----------
    campaign_row : row from campaign_desc for this campaign
    tx           : full transaction_data
    product      : product lookup (PRODUCT_ID → DEPARTMENT, BRAND)
    coupon_redempt : coupon redemptions
    treated_hh   : set of household_keys targeted in this campaign
    all_control_hh : households never in any campaign (pure holdout)
    le_dept      : fitted LabelEncoder for department names
    """
    cid = int(campaign_row["CAMPAIGN"])
    start_day = int(campaign_row["START_DAY"])
    end_day = int(campaign_row["END_DAY"])
    campaign_type_str = str(campaign_row["DESCRIPTION"]).strip()
    campaign_type = {"TypeA": 2, "TypeB": 1, "TypeC": 0}.get(campaign_type_str, -1)
    campaign_duration = end_day - start_day

    # Households in scope: treated for this campaign + pure controls
    in_scope = treated_hh | all_control_hh

    # Pre-campaign transactions (strict cutoff — no leakage)
    tx_pre = tx[tx["DAY"] < start_day].copy()
    # Attach product info (department, brand)
    tx_pre = tx_pre.merge(
        product[["PRODUCT_ID", "DEPARTMENT", "BRAND"]],
        on="PRODUCT_ID",
        how="left",
    )

    # Campaign-window transactions (for outcome Y)
    tx_window = tx[
        (tx["DAY"] >= start_day) & (tx["DAY"] <= end_day)
    ].copy()

    # ---- RFM base features ------------------------------------------------
    # recency: days since last purchase before START_DAY
    last_day = (
        tx_pre.groupby("household_key")["DAY"]
        .max()
        .rename("last_day_pre")
    )
    # frequency: number of unique basket-days
    freq = (
        tx_pre.groupby("household_key")["DAY"]
        .nunique()
        .rename("frequency")
    )
    # monetary: total spend
    monetary = (
        tx_pre.groupby("household_key")["SALES_VALUE"]
        .sum()
        .rename("monetary")
    )
    # coupon discount (raw)
    coupon_disc_total = (
        tx_pre.groupby("household_key")["COUPON_DISC"]
        .sum()
        .rename("coupon_disc_total")
    )
    # retail discount (raw, stored as negative values in dataset)
    retail_disc_total = (
        tx_pre.groupby("household_key")["RETAIL_DISC"]
        .sum()
        .abs()
        .rename("retail_disc_total")
    )

    # ---- Department-level features ----------------------------------------
    dept_spend = (
        tx_pre.groupby(["household_key", "DEPARTMENT"])["SALES_VALUE"]
        .sum()
        .reset_index()
    )
    # top department per household
    top_dept = (
        dept_spend.sort_values("SALES_VALUE", ascending=False)
        .groupby("household_key")["DEPARTMENT"]
        .first()
        .rename("top_department_raw")
    )
    # share of spend in top department
    hh_total_spend = dept_spend.groupby("household_key")["SALES_VALUE"].sum()
    top_dept_spend = (
        dept_spend.sort_values("SALES_VALUE", ascending=False)
        .groupby("household_key")["SALES_VALUE"]
        .first()
    )
    top_dept_pct = (top_dept_spend / hh_total_spend.replace(0, np.nan)).rename(
        "top_dept_pct"
    )
    # n_departments_visited
    n_depts = (
        dept_spend.groupby("household_key")["DEPARTMENT"]
        .nunique()
        .rename("n_departments_visited")
    )
    # department entropy
    dept_entropy = (
        dept_spend.groupby("household_key")
        .apply(lambda g: _department_entropy(g.set_index("DEPARTMENT")["SALES_VALUE"]))
        .rename("department_entropy")
    )

    # ---- Private label ratio -----------------------------------------------
    pl_spend = (
        tx_pre[tx_pre["BRAND"] == "Private"]
        .groupby("household_key")["SALES_VALUE"]
        .sum()
        .rename("pl_spend")
    )

    # ---- Coupon sensitivity (from coupon_redempt, pre-campaign) ------------
    cr_pre = coupon_redempt[
        (coupon_redempt["DAY"] < start_day)
        & (coupon_redempt["CAMPAIGN"] != cid)
    ]
    n_coupons = (
        cr_pre.groupby("household_key")
        .size()
        .rename("n_coupons_redeemed_pre")
    )

    # ---- Assemble per-household frame -------------------------------------
    # Start from the in-scope household list
    hh_index = pd.Index(sorted(in_scope), name="hshd_num")
    df = pd.DataFrame(index=hh_index)

    df = df.join(last_day.rename_axis("hshd_num"), how="left")
    df = df.join(freq.rename_axis("hshd_num"), how="left")
    df = df.join(monetary.rename_axis("hshd_num"), how="left")
    df = df.join(coupon_disc_total.rename_axis("hshd_num"), how="left")
    df = df.join(retail_disc_total.rename_axis("hshd_num"), how="left")
    df = df.join(top_dept.rename_axis("hshd_num"), how="left")
    df = df.join(top_dept_pct.rename_axis("hshd_num"), how="left")
    df = df.join(n_depts.rename_axis("hshd_num"), how="left")
    df = df.join(dept_entropy.rename_axis("hshd_num"), how="left")
    df = df.join(pl_spend.rename_axis("hshd_num"), how="left")
    df = df.join(n_coupons.rename_axis("hshd_num"), how="left")

    df = df.reset_index()

    # ---- Derived RFM features ---------------------------------------------
    # Households with no pre-campaign transactions: frequency=0, monetary=0
    df["frequency"] = df["frequency"].fillna(0).astype(float)
    df["monetary"] = df["monetary"].fillna(0.0)
    df["coupon_disc_total"] = df["coupon_disc_total"].fillna(0.0)
    df["retail_disc_total"] = df["retail_disc_total"].fillna(0.0)
    df["pl_spend"] = df["pl_spend"].fillna(0.0)
    df["n_coupons_redeemed_pre"] = df["n_coupons_redeemed_pre"].fillna(0).astype(float)

    # recency_days: days before START_DAY since last purchase
    df["recency_days"] = (start_day - df["last_day_pre"]).clip(lower=0)
    # For households with no prior transactions, recency = start_day (max possible)
    df["recency_days"] = df["recency_days"].fillna(float(start_day))

    # avg_basket (guard div-by-zero)
    df["avg_basket"] = np.where(
        df["frequency"] > 0, df["monetary"] / df["frequency"], 0.0
    )

    # weekly_visit_rate: frequency / (start_day / 7)
    weeks_before = max(start_day / 7.0, 1.0)
    df["weekly_visit_rate"] = df["frequency"] / weeks_before

    # coupon_sensitivity: coupon_disc / monetary (ratio of coupon savings to total spend)
    df["coupon_sensitivity"] = np.where(
        df["monetary"] > 0, df["coupon_disc_total"] / df["monetary"], 0.0
    )

    # avg_retail_discount_rate
    df["avg_retail_discount_rate"] = np.where(
        df["monetary"] > 0, df["retail_disc_total"] / df["monetary"], 0.0
    )

    # private_label_ratio
    df["private_label_ratio"] = np.where(
        df["monetary"] > 0, df["pl_spend"] / df["monetary"], 0.0
    )

    # coupon_redeemer binary
    df["coupon_redeemer"] = (df["n_coupons_redeemed_pre"] > 0).astype(int)

    # top_department: label-encoded (unknown → -1)
    df["top_department_raw"] = df["top_department_raw"].fillna("UNKNOWN")
    known_classes = set(le_dept.classes_)
    df["top_department_raw"] = df["top_department_raw"].apply(
        lambda x: x if x in known_classes else "UNKNOWN"
    )
    df["top_department"] = le_dept.transform(df["top_department_raw"]).astype(float)

    # top_dept_pct, n_departments_visited, department_entropy
    df["top_dept_pct"] = df["top_dept_pct"].fillna(0.0)
    df["n_departments_visited"] = df["n_departments_visited"].fillna(0).astype(float)
    df["department_entropy"] = df["department_entropy"].fillna(0.0)

    # ---- Treatment T -------------------------------------------------------
    df["T"] = df["hshd_num"].isin(treated_hh).astype(int)

    # ---- Outcome Y ---------------------------------------------------------
    purchased_in_window = set(
        tx_window["household_key"].unique()
    )
    df["Y"] = df["hshd_num"].isin(purchased_in_window).astype(int)

    # ---- Campaign metadata -------------------------------------------------
    df["campaign_id"] = cid
    df["campaign_type"] = campaign_type
    df["campaign_duration_days"] = float(campaign_duration)

    # ---- Select and order final columns ------------------------------------
    feature_cols = [
        "recency_days",
        "frequency",
        "monetary",
        "avg_basket",
        "weekly_visit_rate",
        "coupon_sensitivity",
        "private_label_ratio",
        "top_department",
        "department_entropy",
        "top_dept_pct",
        "n_departments_visited",
        "n_coupons_redeemed_pre",
        "coupon_redeemer",
        "avg_retail_discount_rate",
        "campaign_type",
        "campaign_duration_days",
    ]

    out_cols = ["hshd_num", "campaign_id", "T", "Y"] + feature_cols
    return df[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_dunnhumby(campaign_id: int | None = DEFAULT_CAMPAIGN_ID) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (household, campaign):

        hshd_num, campaign_id, T (1=treated / 0=control),
        Y (1=purchased during campaign window / 0=not),
        + 15-20 pre-campaign features (no leakage past START_DAY).

    Treatment definition
    --------------------
    T = 1  household_key appears in campaign_table for this campaign.
    T = 0  household_key appears in transaction_data but NEVER in
           campaign_table (pure natural holdout; ~916 households).

    Outcome definition
    ------------------
    Y = 1  household made at least one purchase in [START_DAY, END_DAY].
    Y = 0  otherwise.

    Features (pre-campaign only — computed from DAY < START_DAY)
    ------------------------------------------------------------
    recency_days          : days since last purchase before START_DAY
    frequency             : number of unique purchase days before START_DAY
    monetary              : total SALES_VALUE before START_DAY
    avg_basket            : monetary / frequency
    weekly_visit_rate     : frequency / (START_DAY / 7)
    coupon_sensitivity    : coupon_disc / monetary ratio
    private_label_ratio   : private-label spend / total spend
    top_department        : most-visited department (label-encoded)
    department_entropy    : Shannon entropy of spend across departments
    top_dept_pct          : share of spend in top department
    n_departments_visited : count of distinct departments
    n_coupons_redeemed_pre: coupon redemptions before campaign window
    coupon_redeemer       : binary — ever redeemed coupon pre-campaign
    avg_retail_discount_rate : abs(RETAIL_DISC) / monetary ratio
    campaign_type         : TypeA=2 / TypeB=1 / TypeC=0 (ordinal)
    campaign_duration_days: END_DAY - START_DAY

    Propensity note
    ---------------
    Downstream propensity model P(T=1|X) MUST clip predictions to [0.1, 0.9].
    Treated households spend ~4.7x more pre-campaign; without clipping the
    propensity model assigns extreme weights that collapse the CATE estimate
    toward the treated-side tau_1 learner only.

    Parameters
    ----------
    campaign_id : int or None
        If int, filter to a single campaign (default = 26, TypeA, confirmed
        real effect).  If None, pool all 30 campaigns; each household-campaign
        pair becomes its own row, and campaign_id / campaign_type are included
        as covariates.

    Returns
    -------
    pd.DataFrame
    """
    tables = _load_tables()
    tx: pd.DataFrame = tables["transaction_data"]
    product: pd.DataFrame = tables["product"]
    campaign_desc: pd.DataFrame = tables["campaign_desc"]
    campaign_table: pd.DataFrame = tables["campaign_table"]
    coupon_redempt: pd.DataFrame = tables["coupon_redempt"]

    # Build a label encoder for departments (fit on all known + UNKNOWN)
    all_depts = sorted(product["DEPARTMENT"].dropna().unique().tolist()) + ["UNKNOWN"]
    le_dept = LabelEncoder()
    le_dept.fit(all_depts)

    # Pure control pool: in transactions but NEVER in campaign_table
    all_treated_hh = set(campaign_table["household_key"].unique())
    all_tx_hh = set(tx["household_key"].unique())
    all_control_hh = all_tx_hh - all_treated_hh  # ~916 households

    # Determine campaigns to process
    if campaign_id is not None:
        campaigns_to_run = campaign_desc[
            campaign_desc["CAMPAIGN"] == campaign_id
        ].copy()
        if campaigns_to_run.empty:
            raise ValueError(
                f"campaign_id={campaign_id} not found in campaign_desc.csv. "
                f"Available IDs: {sorted(campaign_desc['CAMPAIGN'].tolist())}"
            )
    else:
        campaigns_to_run = campaign_desc.copy()

    frames: list[pd.DataFrame] = []
    for _, row in campaigns_to_run.iterrows():
        cid = int(row["CAMPAIGN"])
        treated_hh = set(
            campaign_table.loc[campaign_table["CAMPAIGN"] == cid, "household_key"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            frame = _build_features_for_campaign(
                campaign_row=row,
                tx=tx,
                product=product,
                coupon_redempt=coupon_redempt,
                treated_hh=treated_hh,
                all_control_hh=all_control_hh,
                le_dept=le_dept,
            )
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)

    # For single-campaign mode, drop campaign_id/campaign_type from features
    # (they are constants and would leak through feature selection).
    # Keep them for reference columns anyway — callers can drop if needed.

    return result


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Dunnhumby uplift dataset.")
    parser.add_argument(
        "--campaign",
        type=int,
        default=DEFAULT_CAMPAIGN_ID,
        help="Campaign ID to prepare (default=26). Use 0 for all campaigns.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output CSV path.",
    )
    args = parser.parse_args()
    cid = args.campaign if args.campaign != 0 else None
    df = prepare_dunnhumby(campaign_id=cid)
    print(f"Shape: {df.shape}")
    print(f"T dist: {df['T'].value_counts().to_dict()}")
    print(f"Y dist: {df['Y'].value_counts().to_dict()}")
    print(f"Columns: {df.columns.tolist()}")
    nan_counts = df.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        print(f"NaN counts: {nan_counts.to_dict()}")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")
