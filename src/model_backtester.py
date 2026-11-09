"""model_backtester.py — Backtest harness for model accuracy validation.

Validates whether the prediction model is accurate enough to show point estimates
of expected profit to stores, or whether it should display a confidence interval
range instead (e.g. "คาดกำไร ฿900–1,600").

Two tracks (auto-selected):

  Track A — Segment Return Rate  (no campaign history needed)
    1. Split pos_df by time: train = first ``train_months`` months,
       test = remaining months (≥ 2).
    2. Run IBM segmentation on training data → customer → segment labels.
    3. For each segment, "predicted return rate" = % of segment customers
       who purchased in the last 30 days of the training window.
    4. "Actual return rate" = % of same customers who purchased in the
       first 30 days of the test window.
    5. MAPE per segment → median MAPE.  Pass if median ≤ threshold.

  Track B — Campaign Revenue Accuracy  (needs ≥ MIN_CAMPAIGNS outcomes)
    Use campaign_history entries that have recorded ``actual_revenue``
    (from POST /campaign/record-outcome).  Compare ``expected_profit`` vs
    ``actual_revenue`` weighted by ``customers_selected``.
    Pass if weighted MAPE ≤ threshold.

Auto-selects Track B when enough outcome-recorded campaigns exist in the
test window; falls back to Track A otherwise.

CI Width (optional):
    If the champion model is available, compute prediction intervals on
    test customers via CausalForestDML.predict_interval(alpha=0.3).
    ``ci_width_ratio`` = median( (high − low) / |point| ).
    If > ``ci_width_threshold`` the display_mode is forced to "range"
    even when MAPE passes.

Usage
-----
    from model_backtester import run_backtest

    result = run_backtest(pos_df=pos_df, campaign_history=history)
    print(result.passed, result.display_mode, result.summary_th)
    # → True  "point"  "โมเดลผ่านเกณฑ์ ..."
"""
from __future__ import annotations

import logging
import sys
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (confirmed with team before pilot)
# ---------------------------------------------------------------------------

PASS_THRESHOLD          = 0.25   # median/weighted MAPE ≤ 25% → pass
MIN_CAMPAIGNS_TRACK_B   = 5      # minimum outcome-recorded campaigns for Track B
CI_WIDTH_THRESHOLD      = 2.0    # CI (high−low)/point > 2× → force "range" display
RETURN_WINDOW_DAYS      = 30     # look-ahead window for return rate / revenue
DEFAULT_TRAIN_MONTHS    = 9      # first N months used for training reference


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SegmentMAPE:
    segment_name:   str
    predicted_rate: float          # % predicted to return
    actual_rate:    float          # % who actually returned
    mape:           float          # |pred − actual| / actual  (or vs pred if actual=0)
    n_customers:    int


@dataclass
class CampaignMAPE:
    execution_id:     str
    executed_at:      str
    segment_name:     str
    expected_profit:  float
    actual_revenue:   float
    mape:             float
    customers:        int


@dataclass
class BacktestResult:
    # Which track was used
    track:          Literal["A", "B"]
    passed:         bool
    pass_threshold: float

    # Track A
    segment_mapes:  list[SegmentMAPE] = field(default_factory=list)
    median_mape:    float | None = None

    # Track B
    campaign_mapes:  list[CampaignMAPE] = field(default_factory=list)
    weighted_mape:   float | None = None

    # CI width (None if model not available)
    ci_width_ratio:  float | None = None

    # Periods
    train_start:     str = ""
    train_end:       str = ""
    test_start:      str = ""
    test_end:        str = ""

    # Counts
    n_train_customers: int = 0
    n_test_customers:  int = 0
    n_campaigns_in_test: int = 0

    # UI guidance
    display_mode:    Literal["point", "range"] = "range"

    # Thai summary
    summary_th:      str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    pos_df: Any,
    campaign_history: list[dict] | None = None,
    train_months: int = DEFAULT_TRAIN_MONTHS,
    pass_threshold: float = PASS_THRESHOLD,
    min_campaigns_track_b: int = MIN_CAMPAIGNS_TRACK_B,
    ci_width_threshold: float = CI_WIDTH_THRESHOLD,
    return_window_days: int = RETURN_WINDOW_DAYS,
    models_dir: str | None = None,
) -> BacktestResult:
    """Run the full backtest harness.

    Parameters
    ----------
    pos_df : pd.DataFrame
        Historical POS transactions.  Required columns (normalised):
        ``customer_id``, ``transaction_date`` (datetime), ``amount``.
    campaign_history : list[dict], optional
        Campaign outcome records from ``store.py``.  Each record should have:
        ``execution_id``, ``executed_at``, ``segment_name``,
        ``expected_profit``, ``actual_revenue``, ``customers_selected``.
    train_months : int
        Number of months to use as training reference (default 9).
    pass_threshold : float
        MAPE threshold for passing (default 0.25 = 25%).
    min_campaigns_track_b : int
        Minimum campaigns with outcomes in test window to use Track B.
    ci_width_threshold : float
        Ratio threshold above which display_mode is forced to "range".
    return_window_days : int
        Look-ahead window for return-rate / revenue comparison (default 30).
    models_dir : str, optional
        Path to model registry directory (for CI computation).

    Returns
    -------
    BacktestResult
    """
    import pandas as pd

    campaign_history = campaign_history or []

    # -- Validate and prepare pos_df ----------------------------------------
    pos_df = _prepare_pos_df(pos_df)
    if pos_df is None or pos_df.empty:
        return _empty_result("pos_df empty or missing required columns")

    # -- Time split ----------------------------------------------------------
    split = _time_split(pos_df, train_months)
    if split is None:
        return _empty_result("ข้อมูลน้อยเกินไป — ต้องการอย่างน้อย 3 เดือน")

    train_df, test_df, train_end, test_start, test_end = split

    # -- Select track --------------------------------------------------------
    test_campaigns = _get_test_campaigns(campaign_history, test_start, test_end)
    track = "B" if len(test_campaigns) >= min_campaigns_track_b else "A"

    # -- Run selected track --------------------------------------------------
    if track == "B":
        result = _run_track_b(
            test_campaigns, pass_threshold,
            train_df=train_df, test_df=test_df,
        )
    else:
        result = _run_track_a(
            train_df, test_df, pass_threshold, return_window_days,
        )

    # -- CI width (optional) ------------------------------------------------
    ci_ratio = _compute_ci_width(test_df, models_dir)
    result.ci_width_ratio = ci_ratio

    # -- Fill metadata -------------------------------------------------------
    result.train_start = pos_df["transaction_date"].min().date().isoformat()
    result.train_end   = train_end.isoformat()
    result.test_start  = test_start.isoformat()
    result.test_end    = test_end.isoformat()
    result.n_train_customers = train_df["customer_id"].nunique()
    result.n_test_customers  = test_df["customer_id"].nunique()
    result.n_campaigns_in_test = len(test_campaigns)

    # -- Display mode -------------------------------------------------------
    ci_forces_range = ci_ratio is not None and ci_ratio > ci_width_threshold
    result.display_mode = "range" if (not result.passed or ci_forces_range) else "point"

    # -- Thai summary -------------------------------------------------------
    result.summary_th = _build_summary_th(result)
    return result


def backtest_display_guidance(result: BacktestResult) -> dict:
    """Return UI-ready display config from a BacktestResult.

    Returns
    -------
    dict
        ``{display_mode, badge, profit_label_template, show_ci}``
        where ``profit_label_template`` is a format string the UI fills in.
    """
    if result.display_mode == "point":
        return {
            "display_mode":          "point",
            "badge":                 None,
            "profit_label_template": "คาดกำไร ฿{profit:,.0f}",
            "show_ci":               False,
        }
    return {
        "display_mode":          "range",
        "badge":                 "กำลังเรียนรู้",
        "profit_label_template": "คาดกำไร ฿{low:,.0f}–{high:,.0f}",
        "show_ci":               True,
    }


# ---------------------------------------------------------------------------
# Track A — Segment Return Rate
# ---------------------------------------------------------------------------

def _run_track_a(
    train_df: Any,
    test_df: Any,
    pass_threshold: float,
    return_window_days: int,
) -> BacktestResult:
    """Compare segment return rates: training window vs test window."""
    import pandas as pd

    result = BacktestResult(track="A", passed=False, pass_threshold=pass_threshold)

    # Segment customers using training data
    seg_map = _segment_customers(train_df)   # {customer_id: segment_name}
    if not seg_map:
        result.summary_th = "ไม่สามารถ segment ได้ — ข้อมูล train ไม่เพียงพอ"
        return result

    train_end   = train_df["transaction_date"].max()
    test_start  = test_df["transaction_date"].min()

    # Predicted return rate: % of each segment's customers who purchased
    # in the last ``return_window_days`` of the training window.
    pred_window_start = train_end - pd.Timedelta(days=return_window_days)
    train_recent = train_df[train_df["transaction_date"] >= pred_window_start]
    pred_buyers  = set(train_recent["customer_id"].unique())

    # Actual return rate: % who purchased in first ``return_window_days``
    # of the test window.
    act_window_end = test_start + pd.Timedelta(days=return_window_days)
    test_recent  = test_df[test_df["transaction_date"] < act_window_end]
    act_buyers   = set(test_recent["customer_id"].unique())

    # Group by segment, compute rates
    from collections import defaultdict
    seg_customers: dict[str, list[str]] = defaultdict(list)
    for cid, seg in seg_map.items():
        seg_customers[seg].append(cid)

    segment_mapes: list[SegmentMAPE] = []
    for seg_name, customers in seg_customers.items():
        n = len(customers)
        if n == 0:
            continue
        cust_set   = set(customers)
        pred_rate  = len(cust_set & pred_buyers) / n
        actual_rate = len(cust_set & act_buyers) / n

        mape = _mape(pred_rate, actual_rate)
        segment_mapes.append(SegmentMAPE(
            segment_name=seg_name,
            predicted_rate=round(pred_rate, 4),
            actual_rate=round(actual_rate, 4),
            mape=round(mape, 4),
            n_customers=n,
        ))

    if not segment_mapes:
        result.summary_th = "ไม่มีข้อมูล segment เพียงพอ"
        return result

    mapes       = [s.mape for s in segment_mapes]
    median_mape = float(_median(mapes))

    result.segment_mapes = sorted(segment_mapes, key=lambda x: x.mape, reverse=True)
    result.median_mape   = round(median_mape, 4)
    result.passed        = median_mape <= pass_threshold
    return result


# ---------------------------------------------------------------------------
# Track B — Campaign Revenue Accuracy
# ---------------------------------------------------------------------------

def _run_track_b(
    test_campaigns: list[dict],
    pass_threshold: float,
    train_df: Any = None,
    test_df: Any = None,
) -> BacktestResult:
    """Compare expected_profit vs actual_revenue for campaigns with outcomes."""
    result = BacktestResult(track="B", passed=False, pass_threshold=pass_threshold)

    campaign_mapes: list[CampaignMAPE] = []
    total_customers = 0

    for c in test_campaigns:
        expected = float(c.get("expected_profit") or c.get("expected_incremental_profit") or 0)
        actual   = float(c.get("actual_revenue") or 0)
        customers = int(c.get("customers_selected") or 1)
        exec_id  = str(c.get("execution_id", ""))
        exec_at  = str(c.get("executed_at", ""))
        seg      = str(c.get("segment_name", ""))

        if expected == 0 and actual == 0:
            continue

        mape = _mape(expected, actual)
        campaign_mapes.append(CampaignMAPE(
            execution_id=exec_id,
            executed_at=exec_at,
            segment_name=seg,
            expected_profit=round(expected, 2),
            actual_revenue=round(actual, 2),
            mape=round(mape, 4),
            customers=customers,
        ))
        total_customers += customers

    if not campaign_mapes:
        result.summary_th = "ไม่มี campaign ที่บันทึก outcome ในช่วง test"
        return result

    # Weighted MAPE: weight by number of customers
    total_w = sum(c.customers for c in campaign_mapes)
    if total_w == 0:
        weighted_mape = float(_median([c.mape for c in campaign_mapes]))
    else:
        weighted_mape = sum(c.mape * c.customers for c in campaign_mapes) / total_w

    result.campaign_mapes = sorted(campaign_mapes, key=lambda x: x.mape, reverse=True)
    result.weighted_mape  = round(weighted_mape, 4)
    result.passed         = weighted_mape <= pass_threshold
    return result


# ---------------------------------------------------------------------------
# CI width computation (optional — requires champion model)
# ---------------------------------------------------------------------------

def _compute_ci_width(test_df: Any, models_dir: str | None) -> float | None:
    """Return median CI width ratio from champion model on test customers.

    Uses CausalForestDML.predict_interval(alpha=0.3) → 70% CI.
    Returns None if model unavailable or computation fails.
    """
    if models_dir is None:
        return None

    try:
        import numpy as np
        import pandas as pd

        _add_src_to_path()
        from model_registry import ModelRegistry
        from uplift_engine import build_serving_context

        registry = ModelRegistry(models_dir)
        champion = registry.get_champion()
        if champion is None:
            return None

        # Build serving context on test data
        ctx = build_serving_context(test_df, registry=registry, models_dir=models_dir)
        if ctx is None:
            return None

        model = champion.get("model_obj")
        if model is None or not hasattr(model, "predict_interval"):
            return None

        X = ctx.X_serve
        if X is None or len(X) == 0:
            return None

        # Predict 70% interval
        intervals = model.predict_interval(X, alpha=0.3)
        if intervals is None or len(intervals) != 2:
            return None

        low, high = np.asarray(intervals[0]), np.asarray(intervals[1])
        point     = model.predict_uplift(X)

        width  = high - low
        denom  = np.abs(point)
        # Avoid division by near-zero
        mask   = denom > 1e-6
        if mask.sum() == 0:
            return None

        ratios = width[mask] / denom[mask]
        return float(round(np.median(ratios), 3))

    except Exception as exc:
        _logger.debug("CI width computation skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Segmentation helper (wraps Layer 1)
# ---------------------------------------------------------------------------

def _segment_customers(train_df: Any) -> dict[str, str]:
    """Return {customer_id: segment_name} using Layer 1 IBM segmentation."""
    try:
        _add_src_to_path()
        from campaign_pipeline import layer1_csv_segmentation

        segments, rfm_df = layer1_csv_segmentation(train_df)

        # Build cluster → segment_name map
        cluster_to_seg: dict[int, str] = {}
        for seg in segments:
            cluster_to_seg[int(seg["cluster_id"])] = seg["segment_name"]

        # Map each customer to their segment
        seg_map: dict[str, str] = {}
        for _, row in rfm_df.iterrows():
            cluster = int(row.get("cluster", row.get("kmeans_cluster", -1)))
            cid     = str(row["customer_id"])
            seg_map[cid] = cluster_to_seg.get(cluster, "Unknown")

        return seg_map

    except Exception as exc:
        _logger.warning("Segmentation failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Time split
# ---------------------------------------------------------------------------

def _time_split(
    pos_df: Any,
    train_months: int,
) -> tuple | None:
    """Split pos_df into train / test by calendar month.

    Returns (train_df, test_df, train_end, test_start, test_end) or None.
    """
    import pandas as pd

    min_date  = pos_df["transaction_date"].min()
    max_date  = pos_df["transaction_date"].max()
    total_days = (max_date - min_date).days

    if total_days < 60:   # need at least 2 months
        return None

    train_end_approx = min_date + pd.DateOffset(months=train_months)

    # Snap to end of that month
    next_month = train_end_approx + pd.DateOffset(months=1)
    train_end  = next_month.replace(day=1) - pd.Timedelta(days=1)

    if train_end >= max_date - pd.Timedelta(days=29):
        # Not enough test data — reduce training window
        train_end = max_date - pd.Timedelta(days=30)

    test_start = train_end + pd.Timedelta(days=1)

    train_df = pos_df[pos_df["transaction_date"] <= train_end].copy()
    test_df  = pos_df[pos_df["transaction_date"] >  train_end].copy()

    if train_df.empty or test_df.empty:
        return None

    return (
        train_df,
        test_df,
        train_end.date() if hasattr(train_end, "date") else train_end,
        test_start.date() if hasattr(test_start, "date") else test_start,
        max_date.date() if hasattr(max_date, "date") else max_date,
    )


# ---------------------------------------------------------------------------
# Campaign history helpers
# ---------------------------------------------------------------------------

def _get_test_campaigns(
    campaign_history: list[dict],
    test_start: Any,
    test_end: Any,
) -> list[dict]:
    """Return campaigns in test window that have recorded actual_revenue."""
    from datetime import datetime

    result = []
    for c in campaign_history:
        if not c.get("actual_revenue"):
            continue
        exec_at_str = c.get("executed_at", "")
        if not exec_at_str:
            continue
        try:
            exec_dt = datetime.fromisoformat(exec_at_str.replace("Z", "+00:00"))
            exec_d  = exec_dt.date()
        except ValueError:
            continue

        ts = test_start if isinstance(test_start, date) else test_start
        te = test_end   if isinstance(test_end, date) else test_end

        if ts <= exec_d <= te:
            result.append(c)
    return result


# ---------------------------------------------------------------------------
# Prepare pos_df
# ---------------------------------------------------------------------------

def _prepare_pos_df(pos_df: Any) -> Any | None:
    """Normalise column names and types. Returns None on failure."""
    try:
        import pandas as pd

        if not isinstance(pos_df, pd.DataFrame) or pos_df.empty:
            return None

        df = pos_df.copy()

        # Normalise column names
        rename: dict[str, str] = {}
        col_lower = {c.lower(): c for c in df.columns}
        for std, variants in [
            ("customer_id",       ["customerid", "customer", "cust_id"]),
            ("transaction_date",  ["date", "timestamp", "created_at", "order_date"]),
            ("amount",            ["total_price", "total", "revenue", "price"]),
        ]:
            if std not in df.columns:
                for v in variants:
                    if v in col_lower:
                        rename[col_lower[v]] = std
                        break

        df = df.rename(columns=rename)

        required = {"customer_id", "transaction_date", "amount"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            _logger.warning("pos_df missing columns: %s", missing)
            return None

        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["amount"]           = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        df["customer_id"]      = df["customer_id"].astype(str).str.strip()
        df = df.dropna(subset=["transaction_date"]).copy()

        return df if not df.empty else None

    except Exception as exc:
        _logger.warning("_prepare_pos_df failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------

def _mape(predicted: float, actual: float) -> float:
    """Symmetric MAPE to handle zero actuals gracefully.

    Uses max(|actual|, |predicted|, ε) as denominator so a 0-vs-0 pair
    scores 0, and a large-vs-zero pair scores ≤ 1.
    """
    denom = max(abs(actual), abs(predicted), 1e-9)
    return abs(predicted - actual) / denom


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


# ---------------------------------------------------------------------------
# Thai summary builder
# ---------------------------------------------------------------------------

def _build_summary_th(result: BacktestResult) -> str:
    """Build a one-paragraph Thai summary of the backtest result."""
    track_label = "Segment Return Rate" if result.track == "A" else "Campaign Revenue"

    if result.track == "A" and result.median_mape is not None:
        mape_str  = f"median MAPE = {result.median_mape * 100:.1f}%"
        mape_val  = result.median_mape
    elif result.track == "B" and result.weighted_mape is not None:
        mape_str  = f"weighted MAPE = {result.weighted_mape * 100:.1f}%"
        mape_val  = result.weighted_mape
    else:
        return "ไม่มีข้อมูลเพียงพอสำหรับ backtest"

    threshold_str = f"{result.pass_threshold * 100:.0f}%"
    status_str    = "✅ ผ่านเกณฑ์" if result.passed else "⚠️ ไม่ผ่านเกณฑ์"
    display_str   = "แสดงตัวเลขเดียว" if result.display_mode == "point" else "แสดงเป็นช่วง + badge 'กำลังเรียนรู้'"

    ci_str = ""
    if result.ci_width_ratio is not None:
        ci_str = f"  ความกว้าง CI = {result.ci_width_ratio:.1f}× (เกณฑ์ {CI_WIDTH_THRESHOLD}×)"
        if result.ci_width_ratio > CI_WIDTH_THRESHOLD:
            ci_str += " → ช่วงกว้างเกิน บังคับแสดงเป็นช่วง"

    # Worst segments (Track A)
    worst_str = ""
    if result.track == "A" and result.segment_mapes:
        worst = result.segment_mapes[0]
        worst_str = f"  segment คลาดเคลื่อนสูงสุด: {worst.segment_name} ({worst.mape * 100:.1f}%)"

    lines = [
        f"Track {result.track} ({track_label}): {mape_str} — {status_str} (เกณฑ์ ≤{threshold_str})",
        f"  ช่วง train: {result.train_start} → {result.train_end}  |  test: {result.test_start} → {result.test_end}",
        f"  ลูกค้า train: {result.n_train_customers:,}  |  test: {result.n_test_customers:,}  |  campaigns in test: {result.n_campaigns_in_test}",
    ]
    if worst_str:
        lines.append(worst_str)
    if ci_str:
        lines.append(ci_str)
    lines.append(f"  การแสดงผล: {display_str}")

    return "\n".join(lines)


def _empty_result(reason: str) -> BacktestResult:
    r = BacktestResult(track="A", passed=False, pass_threshold=PASS_THRESHOLD)
    r.summary_th = reason
    return r


def _add_src_to_path() -> None:
    src_dir = str(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
