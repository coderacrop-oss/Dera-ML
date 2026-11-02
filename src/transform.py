"""
transform.py — Feature Transformation for Uplift Modelling
Time-decay RFM, sequential behavioral features, promo response,
and interaction terms.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from scipy.stats.mstats import winsorize as scipy_winsorize
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class UpliftTransformer:
    """Feature transformer for uplift/CATE modelling.

    Produces one row per customer from raw POS transaction data.

    Feature groups
    --------------
    1. Time-decay RFM      — rfm_recency_score, rfm_frequency_decay,
                             rfm_monetary_decay, rfm_trend
    2. Sequential behav.   — inter_visit_cv, burst_ratio,
                             recency_acceleration, night_owl_ratio,
                             weekend_premium
    3. Promo response      — avg_basket_size, promo_elasticity,
                             category_diversity, coupon_redemption_rate
    4. Interaction terms   — recency_x_frequency, monetary_x_promo,
                             tenure_x_regularity

    All output columns are float32, winsorised at [1%, 99%], then
    StandardScaler-normalised (fit only on training data).
    """

    _FEATURE_NAMES: list[str] = [
        # Group 1 — Time-decay RFM
        "rfm_recency_score",
        "rfm_frequency_decay",
        "rfm_monetary_decay",
        "rfm_trend",
        # Group 2 — Sequential behavioral
        "inter_visit_cv",
        "burst_ratio",
        "recency_acceleration",
        "night_owl_ratio",
        "weekend_premium",
        # Group 3 — Promo response
        "avg_basket_size",
        "promo_elasticity",
        "category_diversity",
        "coupon_redemption_rate",
        # Group 4 — Interaction terms
        "recency_x_frequency",
        "monetary_x_promo",
        "tenure_x_regularity",
    ]

    def __init__(
        self,
        decay_halflife_days: int = 30,
        n_top_features: int = 15,
    ) -> None:
        self.decay_halflife_days = decay_halflife_days
        self.n_top_features = n_top_features
        self._ref_date: Optional[pd.Timestamp] = None
        self._scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, ref_date=None) -> "UpliftTransformer":
        """Fit winsorisation bounds and scaler on training data."""
        raw = self._build_raw_features(df, ref_date=ref_date)
        clipped = _winsorize(raw, self._FEATURE_NAMES)
        self._scaler.fit(clipped[self._FEATURE_NAMES])
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw transactions → normalised feature DataFrame.

        Returns one row per customer_id.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        raw = self._build_raw_features(df, ref_date=self._ref_date)
        clipped = _winsorize(raw, self._FEATURE_NAMES)
        scaled_vals = self._scaler.transform(clipped[self._FEATURE_NAMES])
        result = pd.DataFrame(scaled_vals, index=clipped.index,
                              columns=self._FEATURE_NAMES, dtype=np.float32)
        result.index.name = "customer_id"
        return result.reset_index()

    def fit_transform(self, df: pd.DataFrame, ref_date=None) -> pd.DataFrame:
        """Fit and transform in one step (training use only)."""
        return self.fit(df, ref_date=ref_date).transform(df)

    def get_feature_names(self) -> list[str]:
        return list(self._FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Internal: orchestration
    # ------------------------------------------------------------------

    def _build_raw_features(self, df: pd.DataFrame, ref_date=None) -> pd.DataFrame:
        """Build un-scaled features. Returns DataFrame indexed by customer_id."""
        df = _prepare(df)

        if ref_date is None:
            ref_date = self._ref_date or df["timestamp"].max() + pd.Timedelta(days=1)
        else:
            ref_date = pd.Timestamp(ref_date)
        self._ref_date = ref_date

        lam = np.log(2) / self.decay_halflife_days

        feat = (
            _decay_rfm(df, ref_date, lam)
            .join(_sequential_behavioral(df, ref_date), how="left")
            .join(_promo_response(df, ref_date, lam), how="left")
            .fillna(0.0)
        )

        feat["recency_x_frequency"] = feat["rfm_recency_score"] * feat["rfm_frequency_decay"]
        feat["monetary_x_promo"]    = feat["rfm_monetary_decay"] * feat["promo_elasticity"]

        age_days = (ref_date - df.groupby("customer_id")["timestamp"].min()).dt.days
        age_days = age_days.reindex(feat.index).fillna(1.0).clip(lower=1.0)
        cv = feat["inter_visit_cv"].replace(0.0, np.nan)
        feat["tenure_x_regularity"] = (age_days * (1.0 / cv)).fillna(0.0)

        return feat[self._FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Winsorisation — scipy one-liner per column (replaces custom loop)
# ---------------------------------------------------------------------------

def _winsorize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Winsorise each column at [1%, 99%] using scipy.stats.mstats.winsorize."""
    out = df.copy()
    for col in cols:
        out[col] = np.array(scipy_winsorize(out[col].values, limits=[0.01, 0.01]))
    return out


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["customer_id", "timestamp"])
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df = df[df["customer_id"] != ""]
    df = df[df["total_price"] > 0]

    for col, default in [
        ("category",       "unknown"),
        ("promo_type",     "none"),
        ("quantity",       1.0),
        ("is_treated",     0),
        ("coupon_offered", 0),
        ("coupon_redeemed", 0),
    ]:
        if col not in df.columns:
            df[col] = default

    df["promo_type"] = df["promo_type"].fillna("none").astype(str).str.lower().str.strip()
    df["category"]   = df["category"].fillna("unknown").astype(str)
    df["hour"]       = df["timestamp"].dt.hour
    df["dayofweek"]  = df["timestamp"].dt.dayofweek
    df["date"]       = df["timestamp"].dt.normalize()
    return df


# ---------------------------------------------------------------------------
# Group 1 — Time-decay RFM
# ---------------------------------------------------------------------------

def _decay_rfm(df: pd.DataFrame, ref_date: pd.Timestamp, lam: float) -> pd.DataFrame:
    """
    rfm_recency_score : exp(-recency_days / halflife)
    rfm_frequency_decay: Σ exp(-days_since_each_visit / halflife)
    rfm_monetary_decay : Σ amount_i × exp(-days_since_i / halflife)
    rfm_trend          : linear slope of 30-day spend windows (positive = growing)
    """
    df = df.copy()
    df["days_since"] = ((ref_date - df["timestamp"]).dt.total_seconds() / 86_400.0).clip(lower=0.0)
    df["decay_w"]    = np.exp(-lam * df["days_since"])

    records: dict[str, dict] = {k: {} for k in
                                 ["rfm_recency_score", "rfm_frequency_decay",
                                  "rfm_monetary_decay", "rfm_trend"]}

    for cid, grp in df.groupby("customer_id"):
        records["rfm_recency_score"][cid]  = float(np.exp(-lam * grp["days_since"].min()))
        records["rfm_frequency_decay"][cid] = float(grp.groupby("date")["decay_w"].max().sum())
        records["rfm_monetary_decay"][cid]  = float((grp["total_price"] * grp["decay_w"]).sum())
        records["rfm_trend"][cid]           = _spend_trend(grp, ref_date)

    return pd.DataFrame(records)


def _spend_trend(grp: pd.DataFrame, ref_date: pd.Timestamp) -> float:
    """Linear slope of per-30-day-bucket total spend. Returns 0 if < 2 buckets."""
    grp = grp.copy()
    grp["bucket"] = ((ref_date - grp["date"]).dt.days // 30).astype(int)
    bucket_spend = grp.groupby("bucket")["total_price"].sum()
    if len(bucket_spend) < 2:
        return 0.0
    x = bucket_spend.index.values.astype(float)
    y = bucket_spend.values.astype(float)
    x_norm = (x - x.mean()) / (x.std() + 1e-9)
    # Negative slope → bucket index increases away from now → spend was growing; flip sign
    return -float(np.polyfit(x_norm, y, 1)[0])


# ---------------------------------------------------------------------------
# Group 2 — Sequential behavioral
# ---------------------------------------------------------------------------

def _sequential_behavioral(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    inter_visit_cv        : CV of inter-visit intervals (regularity proxy)
    burst_ratio           : % visits within 3 days of another visit
    recency_acceleration  : recency_3 / recency_10 — > 1 means speeding up
    night_owl_ratio       : % transactions 22:00-06:00
    weekend_premium       : avg_spend_weekend / avg_spend_weekday
    """
    records: dict[str, dict] = {col: {} for col in [
        "inter_visit_cv", "burst_ratio", "recency_acceleration",
        "night_owl_ratio", "weekend_premium",
    ]}

    for cid, grp in df.groupby("customer_id"):
        visit_dates = sorted(grp["date"].unique())
        n_visits    = len(visit_dates)

        if n_visits >= 2:
            gaps     = np.diff([d.value for d in visit_dates]) / 86_400_000_000_000  # ns→days
            mean_gap = gaps.mean()
            records["inter_visit_cv"][cid] = float(gaps.std() / mean_gap) if mean_gap > 0 else 0.0
            records["burst_ratio"][cid]    = float((gaps <= 3).sum()) / float(n_visits)
        else:
            records["inter_visit_cv"][cid] = 0.0
            records["burst_ratio"][cid]    = 0.0

        visit_recencies = sorted([(ref_date - pd.Timestamp(d)).days for d in visit_dates])
        if len(visit_recencies) >= 3:
            r3  = float(np.mean(visit_recencies[:3]))
            r10 = float(np.mean(visit_recencies[:10]))
            records["recency_acceleration"][cid] = r10 / (r3 + 1.0)
        else:
            records["recency_acceleration"][cid] = 1.0

        records["night_owl_ratio"][cid] = float(grp["hour"].apply(lambda h: h >= 22 or h < 6).mean())

        is_wknd    = grp["dayofweek"].isin([5, 6])
        spend_wknd = grp.loc[is_wknd,  "total_price"].mean() if is_wknd.any()  else np.nan
        spend_wkdy = grp.loc[~is_wknd, "total_price"].mean() if (~is_wknd).any() else np.nan
        records["weekend_premium"][cid] = (
            float(spend_wknd / spend_wkdy)
            if pd.notna(spend_wknd) and pd.notna(spend_wkdy) and spend_wkdy > 0
            else 1.0
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Group 3 — Promo response
# ---------------------------------------------------------------------------

def _promo_response(df: pd.DataFrame, ref_date: pd.Timestamp, lam: float) -> pd.DataFrame:
    """
    avg_basket_size    : mean quantity per transaction (customer purchase intensity,
                         treatment-independent)
    promo_elasticity   : avg spend on promo days vs adjacent ±1-day non-promo
    category_diversity : Shannon entropy of ALL transactions over product categories
                         (heterogeneity signal; not treatment-dependent)
    coupon_redemption_rate : redeemed / offered (0 if columns absent)
    """
    records: dict[str, dict] = {col: {} for col in [
        "avg_basket_size", "promo_elasticity",
        "category_diversity", "coupon_redemption_rate",
    ]}

    is_promo       = df["promo_type"] != "none"
    has_coupon_cols = (
        "coupon_offered" in df.columns and "coupon_redeemed" in df.columns
        and df["coupon_offered"].sum() > 0
    )
    date_spend_idx = (
        df.groupby(["customer_id", "date"])["total_price"].sum()
    )

    for cid, grp in df.groupby("customer_id"):
        # avg_basket_size: mean quantity per transaction (treatment-independent)
        records["avg_basket_size"][cid] = float(grp["quantity"].mean()) if len(grp) > 0 else 1.0

        # category_diversity: Shannon entropy over ALL transactions' categories
        cat_counts = grp["category"].value_counts()
        if len(cat_counts) > 1:
            records["category_diversity"][cid] = float(scipy_entropy(cat_counts / cat_counts.sum()))
        else:
            records["category_diversity"][cid] = 0.0

        records["coupon_redemption_rate"][cid] = (
            float(grp["coupon_redeemed"].sum() / grp["coupon_offered"].sum())
            if has_coupon_cols and grp["coupon_offered"].sum() > 0 else 0.0
        )

        records["promo_elasticity"][cid] = _compute_promo_elasticity(
            grp, date_spend_idx, cid, is_promo
        )

    return pd.DataFrame(records)


def _compute_promo_elasticity(
    grp: pd.DataFrame,
    date_spend_idx: pd.Series,
    cid: str,
    is_promo: pd.Series,
) -> float:
    """Mean ratio of promo-day spend vs adjacent ±1 non-promo days. Returns 1.0 if insufficient."""
    promo_dates = grp.loc[is_promo.loc[grp.index], "date"].unique()
    if len(promo_dates) == 0:
        return 1.0

    all_dates     = set(grp["date"].unique())
    promo_date_set = set(promo_dates)
    ratios = []

    for pd_date in promo_dates:
        try:
            promo_spend = date_spend_idx.loc[(cid, pd_date)]
        except KeyError:
            continue
        adj_spends = [
            date_spend_idx.loc[(cid, pd_date + pd.Timedelta(days=d))]
            for d in (-1, 1)
            if (pd_date + pd.Timedelta(days=d)) in all_dates
            and (pd_date + pd.Timedelta(days=d)) not in promo_date_set
            and (cid, pd_date + pd.Timedelta(days=d)) in date_spend_idx.index
        ]
        if adj_spends:
            avg_adj = float(np.mean(adj_spends))
            if avg_adj > 0:
                ratios.append(float(promo_spend) / avg_adj)

    return float(np.mean(ratios)) if ratios else 1.0

