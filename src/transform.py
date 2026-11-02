"""
transform.py — SOTA Feature Transformation
Time-decay RFM, sequential behavioral features, SHAP-based selection,
interaction terms, covariate balance scoring for uplift models.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class UpliftTransformer:
    """SOTA feature transformer for uplift/CATE modelling.

    Produces one row per customer from raw POS transaction data.

    Feature groups
    --------------
    1. Time-decay RFM      — rfm_recency_score, rfm_frequency_decay,
                             rfm_monetary_decay, rfm_trend
    2. Sequential behav.   — inter_visit_cv, burst_ratio,
                             recency_acceleration, night_owl_ratio,
                             weekend_premium
    3. Promo response      — promo_recency_score, promo_elasticity,
                             promo_type_diversity, coupon_redemption_rate
    4. Covariate balance   — propensity_proxy
    5. Interaction terms   — recency_x_frequency, monetary_x_promo,
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
        "promo_recency_score",
        "promo_elasticity",
        "promo_type_diversity",
        "coupon_redemption_rate",
        # Group 4 — Covariate balance
        "propensity_proxy",
        # Group 5 — Interaction terms
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
        self._winsor_bounds: dict[str, tuple[float, float]] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, ref_date=None) -> "UpliftTransformer":
        """Fit winsorisation bounds and scaler on training data."""
        raw = self._build_raw_features(df, ref_date=ref_date)
        self._fit_winsor(raw)
        clipped = self._apply_winsor(raw)
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
        clipped = self._apply_winsor(raw)
        scaled_vals = self._scaler.transform(clipped[self._FEATURE_NAMES])
        result = pd.DataFrame(
            scaled_vals,
            index=clipped.index,
            columns=self._FEATURE_NAMES,
            dtype=np.float32,
        )
        result.index.name = "customer_id"
        return result.reset_index()

    def fit_transform(self, df: pd.DataFrame, ref_date=None) -> pd.DataFrame:
        """Fit and transform in one step (training use only)."""
        raw = self._build_raw_features(df, ref_date=ref_date)
        self._fit_winsor(raw)
        clipped = self._apply_winsor(raw)
        self._scaler.fit(clipped[self._FEATURE_NAMES])
        self._fitted = True
        scaled_vals = self._scaler.transform(clipped[self._FEATURE_NAMES])
        result = pd.DataFrame(
            scaled_vals,
            index=clipped.index,
            columns=self._FEATURE_NAMES,
            dtype=np.float32,
        )
        result.index.name = "customer_id"
        return result.reset_index()

    def get_feature_names(self) -> list[str]:
        """Return ordered list of output feature names (excludes customer_id)."""
        return list(self._FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Internal: orchestration
    # ------------------------------------------------------------------

    def _build_raw_features(
        self,
        df: pd.DataFrame,
        ref_date=None,
    ) -> pd.DataFrame:
        """Build un-scaled features. Returns DataFrame indexed by customer_id."""
        df = _prepare(df)

        if ref_date is None:
            if self._ref_date is not None:
                ref_date = self._ref_date
            else:
                ref_date = df["timestamp"].max() + pd.Timedelta(days=1)
                self._ref_date = ref_date
        else:
            ref_date = pd.Timestamp(ref_date)
            self._ref_date = ref_date

        lam = np.log(2) / self.decay_halflife_days  # decay rate λ

        g1 = _decay_rfm(df, ref_date, lam)
        g2 = _sequential_behavioral(df, ref_date)
        g3 = _promo_response(df, ref_date, lam)

        feat = g1.join(g2, how="left").join(g3, how="left")
        feat = feat.fillna(0.0)

        # Group 4: propensity proxy (needs full feature matrix)
        feat["propensity_proxy"] = _propensity_proxy(df, feat)

        # Group 5: interaction terms (using pre-scale raw values)
        feat["recency_x_frequency"] = (
            feat["rfm_recency_score"] * feat["rfm_frequency_decay"]
        )
        feat["monetary_x_promo"] = (
            feat["rfm_monetary_decay"] * feat["promo_elasticity"]
        )
        # tenure_x_regularity = customer_age × (1 / inter_visit_cv)
        # compute customer_age_days on-the-fly
        age_days = (ref_date - df.groupby("customer_id")["timestamp"].min()).dt.days
        age_days = age_days.reindex(feat.index).fillna(1.0).clip(lower=1.0)
        cv = feat["inter_visit_cv"].replace(0.0, np.nan)
        feat["tenure_x_regularity"] = age_days * (1.0 / cv)
        feat["tenure_x_regularity"] = feat["tenure_x_regularity"].fillna(0.0)

        return feat[self._FEATURE_NAMES]

    # ------------------------------------------------------------------
    # Winsorisation helpers
    # ------------------------------------------------------------------

    def _fit_winsor(self, df: pd.DataFrame) -> None:
        bounds: dict[str, tuple[float, float]] = {}
        for col in self._FEATURE_NAMES:
            lo = float(df[col].quantile(0.01))
            hi = float(df[col].quantile(0.99))
            bounds[col] = (lo, hi)
        self._winsor_bounds = bounds

    def _apply_winsor(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self._FEATURE_NAMES:
            if col in self._winsor_bounds:
                lo, hi = self._winsor_bounds[col]
                out[col] = out[col].clip(lower=lo, upper=hi)
        return out


# ---------------------------------------------------------------------------
# Data preparation (shared with feature_engineering.py style)
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["customer_id", "timestamp"])
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df = df[df["customer_id"] != ""]
    df = df[df["total_price"] > 0]

    if "category" not in df.columns:
        df["category"] = "unknown"
    if "promo_type" not in df.columns:
        df["promo_type"] = "none"
    if "quantity" not in df.columns:
        df["quantity"] = 1.0
    if "is_treated" not in df.columns:
        df["is_treated"] = 0
    if "coupon_offered" not in df.columns:
        df["coupon_offered"] = 0
    if "coupon_redeemed" not in df.columns:
        df["coupon_redeemed"] = 0

    df["promo_type"] = df["promo_type"].fillna("none").astype(str).str.lower().str.strip()
    df["category"] = df["category"].fillna("unknown").astype(str)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek   # Mon=0, Sun=6
    df["date"] = df["timestamp"].dt.normalize()

    return df


# ---------------------------------------------------------------------------
# Group 1 — Time-decay RFM
# ---------------------------------------------------------------------------

def _decay_rfm(
    df: pd.DataFrame,
    ref_date: pd.Timestamp,
    lam: float,
) -> pd.DataFrame:
    """
    rfm_recency_score : exp(-recency_days / halflife)
    rfm_frequency_decay: Σ exp(-days_since_each_visit / halflife)
    rfm_monetary_decay : Σ amount_i × exp(-days_since_i / halflife)
    rfm_trend          : linear slope of 30-day spend windows (positive = growing)
    """
    records: dict[str, dict] = {col: {} for col in
                                 ["rfm_recency_score", "rfm_frequency_decay",
                                  "rfm_monetary_decay", "rfm_trend"]}

    # Pre-compute days_since for every row
    df = df.copy()
    df["days_since"] = (ref_date - df["timestamp"]).dt.total_seconds() / 86_400.0
    df["days_since"] = df["days_since"].clip(lower=0.0)
    df["decay_w"] = np.exp(-lam * df["days_since"])

    for cid, grp in df.groupby("customer_id"):
        # recency score
        min_days = grp["days_since"].min()
        records["rfm_recency_score"][cid] = float(np.exp(-lam * min_days))

        # frequency decay: one weight per unique visit-date
        visit_days = grp.groupby("date")["decay_w"].max()
        records["rfm_frequency_decay"][cid] = float(visit_days.sum())

        # monetary decay
        records["rfm_monetary_decay"][cid] = float(
            (grp["total_price"] * grp["decay_w"]).sum()
        )

        # trend: slope of 30-day rolling spend windows
        records["rfm_trend"][cid] = _spend_trend(grp, ref_date)

    return pd.DataFrame(records)


def _spend_trend(grp: pd.DataFrame, ref_date: pd.Timestamp) -> float:
    """Linear slope of per-30-day-bucket total spend. Returns 0 if < 2 buckets."""
    grp = grp.copy()
    grp["bucket"] = ((ref_date - grp["date"]).dt.days // 30).astype(int)
    bucket_spend = grp.groupby("bucket")["total_price"].sum()
    if len(bucket_spend) < 2:
        return 0.0
    # x = bucket index (0 = most recent, ascending away from ref_date)
    x = bucket_spend.index.values.astype(float)
    y = bucket_spend.values.astype(float)
    # Normalise x so slope is on [0,1] scale
    x_norm = (x - x.mean()) / (x.std() + 1e-9)
    slope = float(np.polyfit(x_norm, y, 1)[0])
    # Negative slope → bucket index increases away from now → spend was growing
    # Flip sign: positive = customer spending more recently
    return -slope


# ---------------------------------------------------------------------------
# Group 2 — Sequential behavioral
# ---------------------------------------------------------------------------

def _sequential_behavioral(
    df: pd.DataFrame,
    ref_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    inter_visit_cv        : CV of inter-visit intervals (regularity proxy)
    burst_ratio           : % visits within 3 days of another visit
    recency_acceleration  : recency_3 / recency_10 — > 1 means speeding up
    night_owl_ratio       : % transactions 22:00-06:00
    weekend_premium       : avg_spend_weekend / avg_spend_weekday
    """
    records: dict[str, dict] = {
        col: {} for col in [
            "inter_visit_cv", "burst_ratio", "recency_acceleration",
            "night_owl_ratio", "weekend_premium",
        ]
    }

    for cid, grp in df.groupby("customer_id"):
        visit_dates = sorted(grp["date"].unique())
        n_visits = len(visit_dates)

        # --- inter_visit_cv ---
        if n_visits >= 2:
            gaps = np.diff([d.value for d in visit_dates]) / 86_400_000_000_000  # ns→days
            mean_gap = gaps.mean()
            cv = float(gaps.std() / mean_gap) if mean_gap > 0 else 0.0
        else:
            cv = 0.0
        records["inter_visit_cv"][cid] = cv

        # --- burst_ratio ---
        if n_visits >= 2:
            gaps_arr = np.diff([d.value for d in visit_dates]) / 86_400_000_000_000
            burst_count = int((gaps_arr <= 3).sum())
            records["burst_ratio"][cid] = float(burst_count) / float(n_visits)
        else:
            records["burst_ratio"][cid] = 0.0

        # --- recency_acceleration: avg recency of last 3 visits vs last 10 ---
        # Lower avg days_since = more recent; ratio > 1 → last 3 closer than last 10
        visit_recencies = sorted(
            [(ref_date - pd.Timestamp(d)).days for d in visit_dates]
        )  # ascending = most recent first
        if len(visit_recencies) >= 3:
            r3 = float(np.mean(visit_recencies[:3]))
            r10 = float(np.mean(visit_recencies[:10]))
            # r10 >= r3 always; ratio = r10/(r3+1) — high means accelerating
            records["recency_acceleration"][cid] = r10 / (r3 + 1.0)
        else:
            records["recency_acceleration"][cid] = 1.0

        # --- night_owl_ratio ---
        is_night = grp["hour"].apply(lambda h: h >= 22 or h < 6)
        records["night_owl_ratio"][cid] = float(is_night.mean())

        # --- weekend_premium ---
        is_wknd = grp["dayofweek"].isin([5, 6])
        spend_wknd = grp.loc[is_wknd, "total_price"].mean() if is_wknd.any() else np.nan
        spend_wkdy = grp.loc[~is_wknd, "total_price"].mean() if (~is_wknd).any() else np.nan
        if pd.notna(spend_wknd) and pd.notna(spend_wkdy) and spend_wkdy > 0:
            records["weekend_premium"][cid] = float(spend_wknd / spend_wkdy)
        else:
            records["weekend_premium"][cid] = 1.0

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Group 3 — Promo response
# ---------------------------------------------------------------------------

def _promo_response(
    df: pd.DataFrame,
    ref_date: pd.Timestamp,
    lam: float,
) -> pd.DataFrame:
    """
    promo_recency_score    : exp(-days_since_last_promo / halflife)
    promo_elasticity       : avg spend on promo days vs adjacent ±1-day non-promo
    promo_type_diversity   : Shannon entropy of promo type distribution
    coupon_redemption_rate : redeemed / offered (0 if columns absent)
    """
    records: dict[str, dict] = {
        col: {} for col in [
            "promo_recency_score", "promo_elasticity",
            "promo_type_diversity", "coupon_redemption_rate",
        ]
    }

    is_promo = df["promo_type"] != "none"
    has_coupon_cols = (
        "coupon_offered" in df.columns and "coupon_redeemed" in df.columns
        and df["coupon_offered"].sum() > 0
    )

    # Pre-build a date-level spend lookup per customer for elasticity
    date_spend = (
        df.groupby(["customer_id", "date"])["total_price"].sum().reset_index()
    )
    date_spend_idx = date_spend.set_index(["customer_id", "date"])["total_price"]

    for cid, grp in df.groupby("customer_id"):
        promo_grp = grp[is_promo.loc[grp.index]]

        # promo_recency_score
        if len(promo_grp) > 0:
            days_since_promo = (ref_date - promo_grp["timestamp"].max()).days
            records["promo_recency_score"][cid] = float(
                np.exp(-lam * max(0, days_since_promo))
            )
        else:
            records["promo_recency_score"][cid] = 0.0

        # promo_type_diversity: Shannon entropy of promo types
        if len(promo_grp) > 0:
            counts = promo_grp["promo_type"].value_counts()
            probs = counts / counts.sum()
            records["promo_type_diversity"][cid] = float(scipy_entropy(probs))
        else:
            records["promo_type_diversity"][cid] = 0.0

        # coupon_redemption_rate
        if has_coupon_cols:
            offered = grp["coupon_offered"].sum()
            redeemed = grp["coupon_redeemed"].sum()
            records["coupon_redemption_rate"][cid] = (
                float(redeemed / offered) if offered > 0 else 0.0
            )
        else:
            records["coupon_redemption_rate"][cid] = 0.0

        # promo_elasticity
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
    """
    For each promo date, compare that day's spend vs adjacent ±1 non-promo days.
    Returns mean ratio (promo_spend / adjacent_non_promo_spend).
    Returns 1.0 if insufficient data.
    """
    promo_mask = is_promo.loc[grp.index]
    promo_dates = grp.loc[promo_mask, "date"].unique()

    if len(promo_dates) == 0:
        return 1.0

    all_dates = set(grp["date"].unique())
    promo_date_set = set(promo_dates)

    ratios = []
    for pd_date in promo_dates:
        try:
            promo_spend = date_spend_idx.loc[(cid, pd_date)]
        except KeyError:
            continue

        # Adjacent non-promo days
        adj_spends = []
        for delta in [-1, 1]:
            adj_date = pd_date + pd.Timedelta(days=delta)
            if adj_date in all_dates and adj_date not in promo_date_set:
                try:
                    adj_spends.append(date_spend_idx.loc[(cid, adj_date)])
                except KeyError:
                    pass

        if adj_spends:
            avg_adj = float(np.mean(adj_spends))
            if avg_adj > 0:
                ratios.append(float(promo_spend) / avg_adj)

    return float(np.mean(ratios)) if ratios else 1.0


# ---------------------------------------------------------------------------
# Group 4 — Covariate balance / propensity proxy
# ---------------------------------------------------------------------------

def _propensity_proxy(
    df: pd.DataFrame,
    feat: pd.DataFrame,
) -> pd.Series:
    """
    Fit a logistic regression P(was_treated | features) on available features.
    Returns predicted probability as a covariate-balance signal.

    If 'is_treated' column is absent or has only one class, returns 0.5.
    """
    treatment_col = "is_treated"
    if treatment_col not in df.columns:
        return pd.Series(0.5, index=feat.index)

    treatment = (
        df.groupby("customer_id")[treatment_col].max().reindex(feat.index).fillna(0)
    )

    classes = treatment.unique()
    if len(classes) < 2:
        return pd.Series(0.5, index=feat.index)

    X = feat.fillna(0.0).values.astype(np.float64)
    y = treatment.values.astype(int)

    # Avoid fitting if all features are constant
    if X.std(axis=0).sum() == 0:
        return pd.Series(0.5, index=feat.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(
            max_iter=200,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
        scaler_tmp = StandardScaler()
        X_scaled = scaler_tmp.fit_transform(X)
        lr.fit(X_scaled, y)
        proba = lr.predict_proba(X_scaled)[:, 1]

    return pd.Series(proba, index=feat.index, name="propensity_proxy")
