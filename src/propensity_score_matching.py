"""propensity_score_matching.py — Observational treatment matching for uplift model retraining.

Supports 3 modes depending on what DotDash has available:
  A ("discount")       — treatment inferred from discount_amount / promo_code column in POS
  B ("campaign_list")  — treatment from external campaign record (set of customer_ids)
  C ("proxy")          — no treatment data: inferred from purchase spike pattern (weak, flagged)

Feature alignment
-----------------
Features used here MUST match serving pipeline in uplift_engine.py exactly:
  SERVING_FEATURES = ['total_price', 'quantity', 'is_campaign_spike',
                      'rfm_decile_segment', 'frequency']

These are customer-level aggregates derived from raw POS transactions — no labels needed.

Usage
-----
    from propensity_score_matching import match_treated_control, balance_report

    # Mode A — POS has discount column
    matched_df, bal = match_treated_control(pos_df, mode="discount", discount_col="discount_amount")

    # Mode B — have list of customer_ids who received campaign
    matched_df, bal = match_treated_control(
        pos_df, mode="campaign_list", campaign_ids={"C001", "C002", ...}
    )

    # Mode C — no treatment info at all
    matched_df, bal = match_treated_control(pos_df, mode="proxy")

    # Use matched_df to retrain R-Learner
    # matched_df columns: customer_id, is_treated, total_price, quantity,
    #                     is_campaign_spike, rfm_decile_segment, frequency,
    #                     propensity_score, matched_pair_id
"""

from __future__ import annotations

import warnings
from typing import Literal, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pipeline_logger import log


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Must stay in sync with uplift_engine.py build_serving_context()
SERVING_FEATURES = [
    "total_price",        # mean spend per transaction
    "quantity",           # mean items per transaction
    "is_campaign_spike",  # fraction of transactions above 2.5σ price spike
    "rfm_decile_segment", # RFM decile (1-10)
    "frequency",          # unique transaction count
]

_MODE = Literal["discount", "campaign_list", "proxy"]


# ---------------------------------------------------------------------------
# Step 1: Build serving-compatible customer-level features from raw POS
# ---------------------------------------------------------------------------

def _build_customer_features(pos_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level POS to one row per customer.

    Produces exactly SERVING_FEATURES so the trained model can be loaded
    by the serving pipeline without feature mismatch.

    Required POS columns: customer_id, transaction_id, timestamp, total_price
    Optional POS columns: quantity (default 1 if missing)
    """
    df = pos_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "quantity" not in df.columns:
        df["quantity"] = 1

    # is_campaign_spike: transaction total_price > mean + 2.5σ (same logic as serving)
    mean_price = df["total_price"].mean()
    std_price  = df["total_price"].std()
    threshold  = mean_price + 2.5 * std_price
    df["is_campaign_spike"] = (df["total_price"] > threshold).astype(int)

    # Customer-level aggregation
    agg = (
        df.groupby("customer_id")
        .agg(
            total_price=("total_price", "mean"),
            quantity=("quantity", "mean"),
            is_campaign_spike=("is_campaign_spike", "mean"),
            frequency=("transaction_id", "nunique"),
        )
        .reset_index()
    )

    # rfm_decile_segment: rank frequency into 10 deciles (1=lowest, 10=highest)
    agg["rfm_decile_segment"] = pd.qcut(
        agg["frequency"].rank(method="first"),
        q=10, labels=False, duplicates="drop"
    ).astype(float) + 1  # 1-indexed

    return agg


# ---------------------------------------------------------------------------
# Step 2: Assign treatment labels (3 modes)
# ---------------------------------------------------------------------------

def _infer_treatment_mode_a(
    pos_df: pd.DataFrame,
    discount_col: str = "discount_amount",
    promo_col: str | None = "promo_code",
) -> pd.DataFrame:
    """Mode A: treated = 1 if customer ever transacted with a discount or promo code."""
    records = {}
    for cid, grp in pos_df.groupby("customer_id"):
        treated = False
        if discount_col in grp.columns:
            treated = treated or bool((grp[discount_col].fillna(0) > 0).any())
        if promo_col and promo_col in grp.columns:
            treated = treated or bool(grp[promo_col].notna().any())
        records[cid] = int(treated)

    return pd.DataFrame(
        {"customer_id": list(records.keys()), "is_treated": list(records.values())}
    )


def _infer_treatment_mode_b(
    customer_ids: pd.Series,
    campaign_ids: Set[str],
) -> pd.DataFrame:
    """Mode B: treated = 1 if customer_id is in the campaign record."""
    return pd.DataFrame({
        "customer_id": customer_ids,
        "is_treated": customer_ids.map(lambda x: int(x in campaign_ids)).values,
    })


def _infer_treatment_mode_c(
    pos_df: pd.DataFrame,
    campaign_dates: list[str] | None = None,
    campaign_window_days: int = 7,
) -> tuple[pd.DataFrame, str]:
    """Mode C: proxy treatment from purchase-frequency spike.

    AC-4.3: When ``campaign_dates`` are provided (from campaign_history.jsonl),
    treatment assignment uses spikes that fall within ±``campaign_window_days``
    of a known campaign date.  This reduces false-positive treated assignments
    compared to spike-only detection.

    Without ``campaign_dates`` (original behaviour): any month with visits
    >1.5 std above the customer's personal mean is flagged as treated.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Raw POS transactions. Required cols: customer_id, transaction_id, timestamp.
    campaign_dates : list[str] | None
        ISO-format dates (or datetimes) when campaigns were executed.
        If provided, only spikes near a campaign date count as treated.
    campaign_window_days : int
        ±days around each campaign date to consider a spike as campaign-caused.

    Returns (treatment_df, warning_message).
    This is a proxy — retrain with modes A/B when real labels are available.
    """
    from datetime import timedelta

    df = pos_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year_month"] = df["timestamp"].dt.to_period("M")

    monthly = (
        df.groupby(["customer_id", "year_month"])["transaction_id"]
        .nunique()
        .reset_index(name="monthly_visits")
    )

    stats = (
        monthly.groupby("customer_id")["monthly_visits"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_visits", "std": "std_visits"})
    )
    stats["std_visits"] = stats["std_visits"].fillna(0)

    monthly = monthly.merge(stats, on="customer_id")
    monthly["is_spike"] = (
        (monthly["monthly_visits"] > monthly["mean_visits"] + 1.5 * monthly["std_visits"])
        & (monthly["std_visits"] > 0)
    )

    # AC-4.3: If campaign dates are available, further filter spikes to those
    # that occur within ±window_days of a known campaign date.
    history_used = False
    if campaign_dates:
        parsed_dates = []
        for d in campaign_dates:
            try:
                parsed_dates.append(pd.Timestamp(d))
            except Exception:
                continue

        if parsed_dates:
            history_used = True
            # For each spike row, check if its period start falls within any campaign window
            def _near_campaign(period) -> bool:
                period_start = period.start_time
                return any(
                    abs((period_start - cd).days) <= campaign_window_days
                    for cd in parsed_dates
                )

            monthly["near_campaign"] = monthly["year_month"].apply(_near_campaign)
            monthly["is_spike"] = monthly["is_spike"] & monthly["near_campaign"]

    treated = (
        monthly.groupby("customer_id")["is_spike"]
        .any()
        .astype(int)
        .reset_index(name="is_treated")
    )

    treat_rate = treated["is_treated"].mean()

    if history_used:
        warning = (
            f"[PSM Mode C+history] treatment labels inferred from frequency spikes "
            f"near {len(parsed_dates)} campaign dates (±{campaign_window_days}d window) "
            f"(treat rate: {treat_rate:.1%}). "
            "More accurate than spike-only but still a proxy — collect discount data for best results."
        )
    else:
        warning = (
            f"[PSM Mode C] treatment labels inferred from purchase frequency spikes "
            f"(treat rate: {treat_rate:.1%}). "
            "This is a proxy — results will have higher bias than modes A or B. "
            "Collect real campaign records (discount column or campaign list) before "
            "retraining the uplift model for production use."
        )
    return treated, warning


# ---------------------------------------------------------------------------
# Step 3: Propensity score estimation
# ---------------------------------------------------------------------------

def _estimate_propensity(features_df: pd.DataFrame, treatment: pd.Series) -> pd.Series:
    """Fit logistic regression on serving features → return P(treated=1 | X)."""
    feature_cols = [c for c in SERVING_FEATURES if c in features_df.columns]
    X = features_df[feature_cols].fillna(0).values
    y = treatment.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_scaled, y)

    ps = model.predict_proba(X_scaled)[:, 1]
    return pd.Series(ps, index=features_df.index, name="propensity_score")


# ---------------------------------------------------------------------------
# Step 4: 1:1 Nearest-neighbor matching with caliper
# ---------------------------------------------------------------------------

def _nearest_neighbor_match(
    df: pd.DataFrame,
    caliper: float | None = None,
) -> pd.DataFrame:
    """Match each treated customer to the nearest untreated on propensity score.

    Uses caliper = 0.2 * std(logit(PS)) by default (Austin 2011 recommendation).
    Each control unit is used at most once (without replacement).
    """
    treated = df[df["is_treated"] == 1].reset_index(drop=True)
    control = df[df["is_treated"] == 0].reset_index(drop=True)

    if caliper is None:
        logit_ps = np.log(
            df["propensity_score"].clip(1e-6, 1 - 1e-6)
            / (1 - df["propensity_score"].clip(1e-6, 1 - 1e-6))
        )
        caliper = 0.2 * float(logit_ps.std())

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(control[["propensity_score"]].values)

    distances, indices = nn.kneighbors(treated[["propensity_score"]].values)

    matched_rows: list[pd.Series] = []
    used_control: set[int] = set()
    pair_id = 0

    for i, (dist, ctrl_idx) in enumerate(zip(distances[:, 0], indices[:, 0])):
        if dist > caliper:
            continue
        if ctrl_idx in used_control:
            continue
        used_control.add(ctrl_idx)

        t_row = treated.iloc[i].copy()
        c_row = control.iloc[ctrl_idx].copy()
        t_row["matched_pair_id"] = pair_id
        c_row["matched_pair_id"] = pair_id
        matched_rows.extend([t_row, c_row])
        pair_id += 1

    if not matched_rows:
        raise ValueError(
            "PSM: ไม่มี matched pairs — caliper อาจแคบเกินไป หรือ propensity distributions "
            "ของ treated/control ไม่ overlap กัน ลอง caliper ใหญ่ขึ้น หรือตรวจสอบ treatment assignment"
        )

    matched_df = pd.DataFrame(matched_rows).reset_index(drop=True)
    log("psm", "matching_complete", n_pairs=pair_id, caliper=round(caliper, 4))
    return matched_df


# ---------------------------------------------------------------------------
# Step 5: Balance diagnostics
# ---------------------------------------------------------------------------

def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized Mean Difference — < 0.1 is considered well-balanced."""
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return 0.0 if pooled_std == 0 else abs(a.mean() - b.mean()) / pooled_std


def balance_report(matched_df: pd.DataFrame) -> dict:
    """Compute covariate balance statistics after matching.

    Returns dict with per-feature SMD and an _summary key.
    SMD < 0.1 → well-balanced. SMD > 0.25 → consider respecifying model.
    """
    feature_cols = [c for c in SERVING_FEATURES if c in matched_df.columns]
    treated = matched_df[matched_df["is_treated"] == 1]
    control = matched_df[matched_df["is_treated"] == 0]

    report: dict = {}
    for col in feature_cols:
        smd_val = _smd(treated[col].values, control[col].values)
        report[col] = {
            "smd": round(smd_val, 4),
            "balanced": smd_val < 0.1,
            "treated_mean": round(float(treated[col].mean()), 2),
            "control_mean": round(float(control[col].mean()), 2),
        }

    n_balanced = sum(1 for v in report.values() if v["balanced"])
    report["_summary"] = {
        "n_pairs": int(len(matched_df) // 2),
        "features_balanced": f"{n_balanced}/{len(feature_cols)}",
        "all_balanced": n_balanced == len(feature_cols),
    }
    return report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_treated_control(
    pos_df: pd.DataFrame,
    mode: _MODE = "discount",
    discount_col: str = "discount_amount",
    promo_col: str | None = "promo_code",
    campaign_ids: Set[str] | None = None,
    caliper: float | None = None,
    campaign_dates: list[str] | None = None,
    campaign_window_days: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """Main entry point for Propensity Score Matching.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Raw POS transactions. Required columns: customer_id, transaction_id,
        timestamp, total_price. Optional: discount_amount, promo_code.
    mode : "discount" | "campaign_list" | "proxy"
        How to infer treatment labels (see module docstring).
    discount_col : str
        Column name for discount amount (mode A). Default: "discount_amount".
    promo_col : str | None
        Column name for promo code (mode A). Pass None to skip. Default: "promo_code".
    campaign_ids : set[str] | None
        Customer IDs who received a campaign (mode B). Required for mode B.
    caliper : float | None
        Max propensity score distance for a valid match.
        Default: 0.2 × std(logit(PS)) — Austin (2011) recommendation.
    campaign_dates : list[str] | None
        AC-4.3: ISO-format campaign execution dates from campaign_history.jsonl.
        When provided, Mode C only counts spikes within ±campaign_window_days of
        a campaign date — reduces false-positive treated assignments.
    campaign_window_days : int
        AC-4.3: ±days around each campaign date (default 7).

    Returns
    -------
    matched_df : pd.DataFrame
        One row per customer (treated + matched control), with columns:
        customer_id, is_treated, recency_days, frequency, monetary_avg,
        monetary_total, propensity_score, matched_pair_id.
        Use this as training data for R-Learner retraining.
    bal : dict
        Balance report from balance_report(). Check _summary.all_balanced.
    """
    proxy_warning: str | None = None

    # ── Step 1: Customer features ───────────────────────────────────────────
    features = _build_customer_features(pos_df)

    # ── Step 2: Treatment assignment ────────────────────────────────────────
    if mode == "discount":
        treatment = _infer_treatment_mode_a(pos_df, discount_col, promo_col)

    elif mode == "campaign_list":
        if campaign_ids is None:
            raise ValueError("mode='campaign_list' ต้องส่ง campaign_ids มาด้วย")
        treatment = _infer_treatment_mode_b(features["customer_id"], campaign_ids)

    elif mode == "proxy":
        treatment, proxy_warning = _infer_treatment_mode_c(
            pos_df,
            campaign_dates=campaign_dates,
            campaign_window_days=campaign_window_days,
        )
        warnings.warn(proxy_warning, UserWarning, stacklevel=2)

    else:
        raise ValueError(f"mode ต้องเป็น 'discount', 'campaign_list', หรือ 'proxy' — ได้: '{mode}'")

    # ── Step 3: Merge + validate ────────────────────────────────────────────
    df = features.merge(treatment, on="customer_id", how="inner")

    n_treated = int(df["is_treated"].sum())
    n_control = int((df["is_treated"] == 0).sum())
    n_total   = len(df)

    log("psm", "treatment_assigned",
        mode=mode, n_treated=n_treated, n_control=n_control, n_total=n_total)

    if n_treated < 10:
        raise ValueError(
            f"Treated group เล็กเกินไป ({n_treated} คน) — "
            "PSM ต้องการอย่างน้อย 10 treated customers"
        )
    if n_control < n_treated:
        warnings.warn(
            f"Control group ({n_control} คน) น้อยกว่า treated ({n_treated} คน) — "
            "matching อาจได้ pairs ไม่ครบ",
            UserWarning, stacklevel=2,
        )

    # ── Step 4: Propensity scores ───────────────────────────────────────────
    df["propensity_score"] = _estimate_propensity(df, df["is_treated"])

    # ── Step 5: Match ───────────────────────────────────────────────────────
    matched = _nearest_neighbor_match(df, caliper=caliper)

    # ── Step 6: Balance ─────────────────────────────────────────────────────
    bal = balance_report(matched)

    _print_summary(matched, bal, proxy_warning)
    return matched, bal


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_summary(
    matched_df: pd.DataFrame,
    bal: dict,
    proxy_warning: str | None,
) -> None:
    summary = bal.get("_summary", {})
    W = 56
    print("\n" + "=" * W)
    print("  Propensity Score Matching — Result")
    print("=" * W)
    print(f"  Matched pairs:          {summary.get('n_pairs', '?'):>8,}")
    print(f"  Features balanced:      {summary.get('features_balanced', '?'):>8}")
    all_ok = summary.get("all_balanced", False)
    print(f"  All balanced (SMD<0.1): {'✅ Yes' if all_ok else '⚠️  No — ดู SMD ด้านล่าง':>6}")
    print("-" * W)
    for feat, stats in bal.items():
        if feat.startswith("_"):
            continue
        icon = "✅" if stats["balanced"] else "⚠️ "
        print(
            f"  {icon} {feat:<20} SMD={stats['smd']:.3f}  "
            f"(T={stats['treated_mean']:.1f} / C={stats['control_mean']:.1f})"
        )
    if proxy_warning:
        print(f"\n  ⚠️  {proxy_warning}")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# CLI — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="PSM smoke test")
    parser.add_argument("--pos-file", default="data/transactions.csv")
    parser.add_argument("--mode", default="discount",
                        choices=["discount", "campaign_list", "proxy"])
    parser.add_argument("--discount-col", default="discount_amount")
    parser.add_argument("--campaign-file", default=None,
                        help="Text file with one customer_id per line (mode B)")
    args = parser.parse_args()

    pos = pd.read_csv(args.pos_file)

    ids: Set[str] | None = None
    if args.campaign_file:
        with open(args.campaign_file) as f:
            ids = {line.strip() for line in f if line.strip()}

    matched, bal = match_treated_control(
        pos,
        mode=args.mode,
        discount_col=args.discount_col,
        campaign_ids=ids,
    )

    print(json.dumps(bal, indent=2, ensure_ascii=False))
    print(f"\nmatched_df shape: {matched.shape}")
    print(matched.head())
