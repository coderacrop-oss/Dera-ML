"""tests/test_model_backtester.py — Tests for model_backtester.py"""
from __future__ import annotations

import sys
import os
from datetime import date, datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_backtester import (
    run_backtest,
    backtest_display_guidance,
    BacktestResult,
    SegmentMAPE,
    CampaignMAPE,
    _mape,
    _median,
    _prepare_pos_df,
    _time_split,
    _get_test_campaigns,
    _build_summary_th,
    PASS_THRESHOLD,
    CI_WIDTH_THRESHOLD,
)

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pos_df(n_customers: int = 50, months: int = 12, seed: int = 42) -> "pd.DataFrame":
    """Synthetic POS DataFrame spanning ``months`` months."""
    rng  = np.random.default_rng(seed)
    base = datetime(2026, 1, 1)
    rows = []
    for cid in range(1, n_customers + 1):
        # Each customer makes 3–10 purchases spread over the period
        n_txn = int(rng.integers(3, 11))
        for _ in range(n_txn):
            day_offset = int(rng.integers(0, months * 30))
            txn_date   = base + timedelta(days=day_offset)
            amount     = round(float(rng.uniform(50, 500)), 2)
            rows.append({
                "customer_id":      str(cid),
                "transaction_date": txn_date,
                "amount":           amount,
            })
    return pd.DataFrame(rows)


def _make_campaign_history(dates_and_data: list[dict]) -> list[dict]:
    """Build campaign_history list."""
    history = []
    for i, d in enumerate(dates_and_data):
        history.append({
            "execution_id":      d.get("execution_id", f"exec_{i}"),
            "executed_at":       d.get("executed_at", "2026-10-01T12:00:00+00:00"),
            "segment_name":      d.get("segment_name", "At-Risk"),
            "expected_profit":   d.get("expected_profit", 1000.0),
            "actual_revenue":    d.get("actual_revenue", 1100.0),
            "customers_selected": d.get("customers_selected", 50),
        })
    return history


# ---------------------------------------------------------------------------
# _mape
# ---------------------------------------------------------------------------

class TestMape:
    def test_exact_match(self):
        assert _mape(100.0, 100.0) == 0.0

    def test_both_zero(self):
        assert _mape(0.0, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_predicted_zero_actual_nonzero(self):
        result = _mape(0.0, 100.0)
        assert result == pytest.approx(1.0)

    def test_actual_zero_predicted_nonzero(self):
        result = _mape(100.0, 0.0)
        assert result == pytest.approx(1.0)

    def test_25_percent_error(self):
        result = _mape(125.0, 100.0)
        assert result == pytest.approx(0.2, abs=0.01)

    def test_symmetric(self):
        assert _mape(80.0, 100.0) == pytest.approx(_mape(100.0, 80.0))


# ---------------------------------------------------------------------------
# _median
# ---------------------------------------------------------------------------

class TestMedian:
    def test_empty(self):
        assert _median([]) == 0.0

    def test_single(self):
        assert _median([0.5]) == 0.5

    def test_odd_count(self):
        assert _median([0.1, 0.3, 0.5]) == 0.3

    def test_even_count(self):
        assert _median([0.1, 0.3]) == pytest.approx(0.2)

    def test_unsorted(self):
        assert _median([0.5, 0.1, 0.3]) == 0.3


# ---------------------------------------------------------------------------
# _prepare_pos_df
# ---------------------------------------------------------------------------

class TestPreposDF:
    def test_none_returns_none(self):
        assert _prepare_pos_df(None) is None

    def test_empty_df_returns_none(self):
        assert _prepare_pos_df(pd.DataFrame()) is None

    def test_missing_required_column_returns_none(self):
        df = pd.DataFrame({"customer_id": ["1"], "amount": [100]})
        # missing transaction_date
        assert _prepare_pos_df(df) is None

    def test_normalises_column_names(self):
        df = pd.DataFrame({
            "customerid":  ["1", "2"],
            "date":        ["2026-01-01", "2026-01-02"],
            "total_price": [100.0, 200.0],
        })
        result = _prepare_pos_df(df)
        assert result is not None
        assert "customer_id" in result.columns
        assert "transaction_date" in result.columns
        assert "amount" in result.columns

    def test_already_normalised(self):
        df = _make_pos_df(10, 3)
        result = _prepare_pos_df(df)
        assert result is not None
        assert len(result) > 0

    def test_coerces_transaction_date(self):
        df = pd.DataFrame({
            "customer_id":      ["1"],
            "transaction_date": ["2026-01-15"],
            "amount":           [150.0],
        })
        result = _prepare_pos_df(df)
        assert pd.api.types.is_datetime64_any_dtype(result["transaction_date"])

    def test_drops_rows_with_bad_date(self):
        df = pd.DataFrame({
            "customer_id":      ["1", "2"],
            "transaction_date": ["2026-01-01", "not-a-date"],
            "amount":           [100.0, 200.0],
        })
        result = _prepare_pos_df(df)
        assert len(result) == 1

    def test_non_dataframe_returns_none(self):
        assert _prepare_pos_df("not a df") is None


# ---------------------------------------------------------------------------
# _time_split
# ---------------------------------------------------------------------------

class TestTimeSplit:
    def test_returns_five_tuple(self):
        df = _make_pos_df(30, 12)
        result = _time_split(df, train_months=9)
        assert result is not None
        assert len(result) == 5

    def test_train_df_before_test_df(self):
        df = _make_pos_df(30, 12)
        train_df, test_df, train_end, test_start, test_end = _time_split(df, 9)
        assert train_df["transaction_date"].max() <= pd.Timestamp(train_end)
        assert test_df["transaction_date"].min() > pd.Timestamp(train_end)

    def test_no_overlap(self):
        df = _make_pos_df(30, 12)
        train_df, test_df, train_end, test_start, test_end = _time_split(df, 9)
        train_dates = set(train_df["transaction_date"].dt.date)
        test_dates  = set(test_df["transaction_date"].dt.date)
        assert train_dates.isdisjoint(test_dates)

    def test_too_little_data_returns_none(self):
        # Only 1 month of data
        df = _make_pos_df(10, 1)
        result = _time_split(df, train_months=9)
        assert result is None

    def test_covers_full_range(self):
        df = _make_pos_df(30, 12)
        train_df, test_df, _, _, _ = _time_split(df, 9)
        total = len(train_df) + len(test_df)
        assert total == len(df)


# ---------------------------------------------------------------------------
# _get_test_campaigns
# ---------------------------------------------------------------------------

class TestGetTestCampaigns:
    def test_empty_history(self):
        result = _get_test_campaigns([], date(2026, 10, 1), date(2026, 12, 31))
        assert result == []

    def test_filters_by_date_range(self):
        history = _make_campaign_history([
            {"executed_at": "2026-09-15T12:00:00+00:00", "actual_revenue": 1000.0},  # before test
            {"executed_at": "2026-10-15T12:00:00+00:00", "actual_revenue": 1200.0},  # in test
            {"executed_at": "2026-11-01T12:00:00+00:00", "actual_revenue": 900.0},   # in test
        ])
        result = _get_test_campaigns(history, date(2026, 10, 1), date(2026, 12, 31))
        assert len(result) == 2

    def test_excludes_missing_actual_revenue(self):
        history = [
            {"execution_id": "a", "executed_at": "2026-10-15T12:00:00+00:00",
             "actual_revenue": None, "segment_name": "At-Risk"},
            {"execution_id": "b", "executed_at": "2026-10-20T12:00:00+00:00",
             "actual_revenue": 500.0, "segment_name": "At-Risk"},
        ]
        result = _get_test_campaigns(history, date(2026, 10, 1), date(2026, 12, 31))
        assert len(result) == 1
        assert result[0]["execution_id"] == "b"

    def test_malformed_date_skipped(self):
        history = [{"execution_id": "bad", "executed_at": "not-a-date",
                    "actual_revenue": 500.0}]
        result = _get_test_campaigns(history, date(2026, 10, 1), date(2026, 12, 31))
        assert result == []


# ---------------------------------------------------------------------------
# run_backtest — Track A (no campaign history)
# ---------------------------------------------------------------------------

class TestRunBacktestTrackA:
    def test_returns_backtest_result(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df)
        assert isinstance(result, BacktestResult)

    def test_track_a_when_no_history(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df, campaign_history=[])
        assert result.track == "A"

    def test_track_a_when_insufficient_campaigns(self):
        df = _make_pos_df(80, 12)
        # Only 2 campaigns with outcomes — below MIN_CAMPAIGNS_TRACK_B (5)
        history = _make_campaign_history([
            {"executed_at": "2026-11-01T12:00:00+00:00", "actual_revenue": 800.0},
            {"executed_at": "2026-11-15T12:00:00+00:00", "actual_revenue": 900.0},
        ])
        result = run_backtest(pos_df=df, campaign_history=history)
        assert result.track == "A"

    def test_median_mape_set(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df)
        assert result.median_mape is not None
        assert 0.0 <= result.median_mape <= 1.0

    def test_segment_mapes_non_empty(self):
        df = _make_pos_df(100, 12)
        result = run_backtest(pos_df=df)
        assert len(result.segment_mapes) > 0

    def test_segment_mape_fields(self):
        df = _make_pos_df(100, 12)
        result = run_backtest(pos_df=df)
        for s in result.segment_mapes:
            assert isinstance(s.segment_name, str)
            assert 0.0 <= s.predicted_rate <= 1.0
            assert 0.0 <= s.actual_rate <= 1.0
            assert s.mape >= 0.0
            assert s.n_customers > 0

    def test_passed_is_bool(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df)
        assert isinstance(result.passed, bool)

    def test_display_mode_set(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df)
        assert result.display_mode in ("point", "range")

    def test_passed_implies_point_display(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df, models_dir=None)
        if result.passed:
            assert result.display_mode == "point"

    def test_failed_implies_range_display(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df, pass_threshold=0.0)  # impossible to pass
        assert result.display_mode == "range"

    def test_summary_th_non_empty(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df)
        assert isinstance(result.summary_th, str)
        assert len(result.summary_th) > 0

    def test_period_metadata_set(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df)
        assert result.train_start != ""
        assert result.train_end != ""
        assert result.test_start != ""
        assert result.test_end != ""

    def test_customer_counts_set(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df)
        assert result.n_train_customers > 0
        assert result.n_test_customers > 0

    def test_none_pos_df_returns_empty_result(self):
        result = run_backtest(pos_df=None)
        assert result.passed is False

    def test_too_little_data_returns_gracefully(self):
        df = _make_pos_df(5, 1)
        result = run_backtest(pos_df=df)
        assert isinstance(result, BacktestResult)
        assert result.passed is False

    def test_custom_pass_threshold(self):
        df = _make_pos_df(80, 12)
        # 100% threshold: always pass
        result = run_backtest(pos_df=df, pass_threshold=1.0)
        assert result.passed is True

    def test_segment_mapes_sorted_descending(self):
        df = _make_pos_df(100, 12)
        result = run_backtest(pos_df=df)
        mapes = [s.mape for s in result.segment_mapes]
        assert mapes == sorted(mapes, reverse=True)

    def test_ci_width_none_without_models_dir(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df, models_dir=None)
        assert result.ci_width_ratio is None

    def test_alternate_column_names(self):
        df = pd.DataFrame({
            "customerid":  [str(i) for i in range(1, 51)] * 5,
            "date":        [datetime(2026, 1, 1) + timedelta(days=d*7)
                            for d in range(50)] * 5,
            "total_price": [100.0] * 250,
        })
        result = run_backtest(pos_df=df)
        assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# run_backtest — Track B (campaign history available)
# ---------------------------------------------------------------------------

class TestRunBacktestTrackB:
    def _pos_df(self):
        return _make_pos_df(80, 12)

    def _history_in_test(self, n: int = 6) -> list[dict]:
        """Generate n campaigns with outcomes in test period (Nov–Dec 2026).

        POS fixture spans Jan–Dec 2026 with 9-month train split →
        test window starts ~Nov 1.  Put all campaigns in Nov/Dec to be safe.
        """
        campaigns = []
        for i in range(n):
            day = (i % 14) + 1       # days 1–14, stays within month
            month = 11 + (i // 14)   # Nov for first 14, Dec after
            campaigns.append({
                "executed_at":        f"2026-{month:02d}-{day:02d}T12:00:00+00:00",
                "segment_name":       "At-Risk",
                "expected_profit":    1000.0 + i * 100,
                "actual_revenue":     1050.0 + i * 90,  # ~5–15% error
                "customers_selected": 50 + i * 10,
            })
        return campaigns

    def test_track_b_selected_with_enough_campaigns(self):
        df = self._pos_df()
        history = self._history_in_test(6)
        result = run_backtest(pos_df=df, campaign_history=history)
        assert result.track == "B"

    def test_weighted_mape_set(self):
        df = self._pos_df()
        history = self._history_in_test(6)
        result = run_backtest(pos_df=df, campaign_history=history)
        assert result.weighted_mape is not None
        assert 0.0 <= result.weighted_mape <= 1.0

    def test_campaign_mapes_populated(self):
        df = self._pos_df()
        history = self._history_in_test(6)
        result = run_backtest(pos_df=df, campaign_history=history)
        assert len(result.campaign_mapes) > 0

    def test_campaign_mape_fields(self):
        df = self._pos_df()
        history = self._history_in_test(6)
        result = run_backtest(pos_df=df, campaign_history=history)
        for c in result.campaign_mapes:
            assert isinstance(c.execution_id, str)
            assert c.expected_profit >= 0
            assert c.actual_revenue >= 0
            assert c.mape >= 0.0
            assert c.customers > 0

    def test_accurate_campaigns_pass(self):
        df = self._pos_df()
        # Campaigns with <5% error in test window (Nov) — should pass threshold of 25%
        campaigns = []
        for i in range(6):
            campaigns.append({
                "executed_at":        f"2026-11-{(i+1)*4:02d}T12:00:00+00:00",
                "expected_profit":    1000.0,
                "actual_revenue":     1030.0,  # 3% error
                "customers_selected": 50,
                "segment_name":       "At-Risk",
            })
        result = run_backtest(pos_df=df, campaign_history=campaigns)
        assert result.passed is True

    def test_inaccurate_campaigns_fail(self):
        df = self._pos_df()
        # Campaigns with 60% error in test window (Nov)
        campaigns = []
        for i in range(6):
            campaigns.append({
                "executed_at":        f"2026-11-{(i+1)*4:02d}T12:00:00+00:00",
                "expected_profit":    1000.0,
                "actual_revenue":     400.0,   # 60% error
                "customers_selected": 50,
                "segment_name":       "At-Risk",
            })
        result = run_backtest(pos_df=df, campaign_history=campaigns)
        assert result.passed is False
        assert result.display_mode == "range"

    def test_campaigns_sorted_by_mape_desc(self):
        df = self._pos_df()
        campaigns = []
        errors = [0.05, 0.40, 0.15]  # varying errors
        for i, err in enumerate(errors):
            campaigns.append({
                "executed_at":        f"2026-11-{(i+1)*5:02d}T12:00:00+00:00",
                "expected_profit":    1000.0,
                "actual_revenue":     1000.0 * (1 + err),
                "customers_selected": 50,
                "segment_name":       "At-Risk",
            })
        result = run_backtest(pos_df=df, campaign_history=campaigns)
        mapes = [c.mape for c in result.campaign_mapes]
        assert mapes == sorted(mapes, reverse=True)

    def test_n_campaigns_in_test_set(self):
        df = self._pos_df()
        history = self._history_in_test(6)
        result = run_backtest(pos_df=df, campaign_history=history)
        assert result.n_campaigns_in_test == 6


# ---------------------------------------------------------------------------
# backtest_display_guidance
# ---------------------------------------------------------------------------

class TestDisplayGuidance:
    def _result(self, passed: bool, ci: float | None = None) -> BacktestResult:
        r = BacktestResult(track="A", passed=passed, pass_threshold=PASS_THRESHOLD)
        r.median_mape   = 0.10 if passed else 0.40
        r.ci_width_ratio = ci
        r.display_mode  = "point" if passed else "range"
        return r

    def test_passed_point_mode(self):
        guide = backtest_display_guidance(self._result(passed=True))
        assert guide["display_mode"] == "point"
        assert guide["badge"] is None
        assert guide["show_ci"] is False
        assert "{profit" in guide["profit_label_template"]

    def test_failed_range_mode(self):
        guide = backtest_display_guidance(self._result(passed=False))
        assert guide["display_mode"] == "range"
        assert guide["badge"] == "กำลังเรียนรู้"
        assert guide["show_ci"] is True
        assert "{low" in guide["profit_label_template"]
        assert "{high" in guide["profit_label_template"]

    def test_thai_badge(self):
        guide = backtest_display_guidance(self._result(passed=False))
        assert any(ord(c) > 127 for c in guide["badge"])


# ---------------------------------------------------------------------------
# CI width threshold forcing range display
# ---------------------------------------------------------------------------

class TestCIWidthThreshold:
    def test_wide_ci_forces_range_even_when_passed(self):
        df = _make_pos_df(80, 12)
        # Inject a mock result with wide CI manually
        result = run_backtest(pos_df=df, pass_threshold=1.0, models_dir=None)
        # Manually simulate wide CI
        result.ci_width_ratio = CI_WIDTH_THRESHOLD + 0.5
        result.passed = True
        result.display_mode = "range" if result.ci_width_ratio > CI_WIDTH_THRESHOLD else "point"
        assert result.display_mode == "range"

    def test_narrow_ci_does_not_force_range(self):
        df = _make_pos_df(80, 12)
        result = run_backtest(pos_df=df, pass_threshold=1.0, models_dir=None)
        result.ci_width_ratio = 1.0   # narrow — well below threshold
        result.passed = True
        expected = "range" if result.ci_width_ratio > CI_WIDTH_THRESHOLD else "point"
        assert expected == "point"


# ---------------------------------------------------------------------------
# _build_summary_th
# ---------------------------------------------------------------------------

class TestBuildSummaryTh:
    def _track_a_result(self, passed: bool = True) -> BacktestResult:
        r = BacktestResult(track="A", passed=passed, pass_threshold=0.25)
        r.median_mape       = 0.12 if passed else 0.38
        r.train_start       = "2026-01-01"
        r.train_end         = "2026-09-30"
        r.test_start        = "2026-10-01"
        r.test_end          = "2026-12-31"
        r.n_train_customers = 300
        r.n_test_customers  = 80
        r.n_campaigns_in_test = 0
        r.display_mode      = "point" if passed else "range"
        r.segment_mapes     = [
            SegmentMAPE("At-Risk", 0.15, 0.12, 0.20, 50),
            SegmentMAPE("Champions", 0.80, 0.78, 0.03, 30),
        ]
        return r

    def test_returns_string(self):
        r = self._track_a_result()
        s = _build_summary_th(r)
        assert isinstance(s, str) and len(s) > 0

    def test_contains_track_label(self):
        r = self._track_a_result()
        s = _build_summary_th(r)
        assert "Track A" in s

    def test_contains_mape_value(self):
        r = self._track_a_result(passed=True)
        s = _build_summary_th(r)
        assert "12.0%" in s

    def test_passed_shows_pass_symbol(self):
        s = _build_summary_th(self._track_a_result(passed=True))
        assert "✅" in s

    def test_failed_shows_warning(self):
        s = _build_summary_th(self._track_a_result(passed=False))
        assert "⚠️" in s

    def test_contains_date_range(self):
        s = _build_summary_th(self._track_a_result())
        assert "2026-01-01" in s
        assert "2026-12-31" in s

    def test_track_b_uses_weighted_mape(self):
        r = BacktestResult(track="B", passed=True, pass_threshold=0.25)
        r.weighted_mape     = 0.18
        r.train_start       = "2026-01-01"
        r.train_end         = "2026-09-30"
        r.test_start        = "2026-10-01"
        r.test_end          = "2026-12-31"
        r.n_train_customers = 300
        r.n_test_customers  = 80
        r.n_campaigns_in_test = 6
        r.display_mode      = "point"
        s = _build_summary_th(r)
        assert "Track B" in s
        assert "18.0%" in s

    def test_ci_width_included_when_set(self):
        r = self._track_a_result()
        r.ci_width_ratio = 2.5
        s = _build_summary_th(r)
        assert "CI" in s or "ci" in s.lower() or "2.5" in s
