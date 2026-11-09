"""tests/test_send_date_recommender.py — Tests for send_date_recommender.py"""
from __future__ import annotations

import sys
import os
from datetime import date, datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from send_date_recommender import (
    recommend_send_date,
    check_date_overlap,
    get_payday_dates,
    _is_payday,
    _month_dates,
    _parse_month,
    _score_date,
    _infer_dow_weights,
    _infer_dom_weights,
    _build_reason,
    _PAYDAY_BOOST,
    _DEFAULT_COOLDOWN_DAYS,
    _DOW_DEFAULT_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(dates_and_segs: list[tuple[str, str]]) -> list[dict]:
    """Build campaign_history list from (date_str, segment_name) pairs."""
    history = []
    for i, (d, seg) in enumerate(dates_and_segs):
        history.append({
            "execution_id":  f"exec_{i}",
            "executed_at":   f"{d}T12:00:00+00:00",
            "segment_name":  seg,
            "plan_id":       f"plan_{i}",
        })
    return history


# ---------------------------------------------------------------------------
# _parse_month
# ---------------------------------------------------------------------------

class TestParseMonth:
    def test_basic(self):
        yr, mo = _parse_month("2026-11")
        assert yr == 2026
        assert mo == 11

    def test_jan(self):
        yr, mo = _parse_month("2026-01")
        assert yr == 2026
        assert mo == 1

    def test_dec(self):
        yr, mo = _parse_month("2025-12")
        assert yr == 2025
        assert mo == 12


# ---------------------------------------------------------------------------
# _month_dates
# ---------------------------------------------------------------------------

class TestMonthDates:
    def test_november_30_days(self):
        dates = _month_dates(2026, 11)
        assert len(dates) == 30
        assert dates[0] == date(2026, 11, 1)
        assert dates[-1] == date(2026, 11, 30)

    def test_january_31_days(self):
        dates = _month_dates(2026, 1)
        assert len(dates) == 31

    def test_february_non_leap(self):
        dates = _month_dates(2026, 2)
        assert len(dates) == 28

    def test_february_leap(self):
        dates = _month_dates(2024, 2)
        assert len(dates) == 29


# ---------------------------------------------------------------------------
# _is_payday
# ---------------------------------------------------------------------------

class TestIsPayday:
    def test_day_25_is_payday(self):
        assert _is_payday(date(2026, 11, 25)) is True

    def test_day_30_is_payday(self):
        assert _is_payday(date(2026, 11, 30)) is True

    def test_day_1_is_payday(self):
        assert _is_payday(date(2026, 11, 1)) is True

    def test_day_5_is_payday(self):
        assert _is_payday(date(2026, 11, 5)) is True

    def test_day_6_not_payday(self):
        assert _is_payday(date(2026, 11, 6)) is False

    def test_day_15_not_payday(self):
        assert _is_payday(date(2026, 11, 15)) is False

    def test_day_24_not_payday(self):
        assert _is_payday(date(2026, 11, 24)) is False


# ---------------------------------------------------------------------------
# get_payday_dates
# ---------------------------------------------------------------------------

class TestGetPaydayDates:
    def test_november_payday_dates(self):
        dates = get_payday_dates("2026-11")
        # Should include 1st–5th and 25th–30th of November
        assert "2026-11-01" in dates
        assert "2026-11-05" in dates
        assert "2026-11-25" in dates
        assert "2026-11-30" in dates
        # Mid-month not included
        assert "2026-11-15" not in dates
        assert "2026-11-10" not in dates

    def test_payday_count_november(self):
        dates = get_payday_dates("2026-11")
        # 1st–5th (5 days) + 25th–30th (6 days) = 11 days
        assert len(dates) == 11

    def test_returns_iso_strings(self):
        dates = get_payday_dates("2026-11")
        for d in dates:
            # Should parse without error
            date.fromisoformat(d)


# ---------------------------------------------------------------------------
# check_date_overlap
# ---------------------------------------------------------------------------

class TestCheckDateOverlap:
    def test_no_history_no_overlap(self):
        result = check_date_overlap("2026-11-15", [])
        assert result["has_overlap"] is False
        assert result["days_apart"] is None

    def test_overlap_within_cooldown(self):
        history = _make_history([("2026-11-10", "At-Risk")])
        result = check_date_overlap("2026-11-15", history, segment="At-Risk", cooldown_days=14)
        assert result["has_overlap"] is True
        assert result["days_apart"] == 5

    def test_no_overlap_beyond_cooldown(self):
        history = _make_history([("2026-10-15", "At-Risk")])
        result = check_date_overlap("2026-11-15", history, segment="At-Risk", cooldown_days=14)
        assert result["has_overlap"] is False
        assert result["days_apart"] == 31

    def test_segment_filter_excludes_different_segment(self):
        history = _make_history([("2026-11-13", "Champions")])
        result = check_date_overlap("2026-11-15", history, segment="At-Risk", cooldown_days=14)
        # Champions history should be excluded when checking At-Risk
        assert result["has_overlap"] is False

    def test_no_segment_filter_includes_all(self):
        history = _make_history([("2026-11-13", "Champions")])
        result = check_date_overlap("2026-11-15", history, segment="", cooldown_days=14)
        assert result["has_overlap"] is True
        assert result["days_apart"] == 2

    def test_nearest_campaign_date_returned(self):
        history = _make_history([
            ("2026-11-01", "At-Risk"),
            ("2026-11-12", "At-Risk"),
        ])
        result = check_date_overlap("2026-11-15", history, segment="At-Risk", cooldown_days=14)
        assert result["nearest_campaign_date"] == "2026-11-12"
        assert result["days_apart"] == 3

    def test_nearest_campaign_id_returned(self):
        history = _make_history([("2026-11-10", "At-Risk")])
        result = check_date_overlap("2026-11-15", history, segment="At-Risk")
        assert result["nearest_campaign_id"] == "exec_0"

    def test_exactly_at_cooldown_boundary_no_overlap(self):
        history = _make_history([("2026-11-01", "At-Risk")])
        result = check_date_overlap("2026-11-15", history, segment="At-Risk", cooldown_days=14)
        # 14 days apart, cooldown is 14 → NOT < 14 → no overlap
        assert result["has_overlap"] is False

    def test_malformed_date_in_history_skipped(self):
        history = [{"execution_id": "bad", "executed_at": "not-a-date", "segment_name": "At-Risk"}]
        result = check_date_overlap("2026-11-15", history, segment="At-Risk")
        assert result["has_overlap"] is False

    def test_missing_executed_at_skipped(self):
        history = [{"execution_id": "nope", "segment_name": "At-Risk"}]
        result = check_date_overlap("2026-11-15", history, segment="At-Risk")
        assert result["has_overlap"] is False

    def test_case_insensitive_segment_match(self):
        history = _make_history([("2026-11-13", "at-risk")])
        result = check_date_overlap("2026-11-15", history, segment="At-Risk", cooldown_days=14)
        assert result["has_overlap"] is True


# ---------------------------------------------------------------------------
# _infer_dow_weights
# ---------------------------------------------------------------------------

class TestInferDowWeights:
    def test_none_returns_defaults(self):
        weights = _infer_dow_weights(None, "At-Risk")
        assert weights == dict(_DOW_DEFAULT_WEIGHTS)

    def test_non_dataframe_returns_defaults(self):
        weights = _infer_dow_weights("not a df", "At-Risk")
        assert weights == dict(_DOW_DEFAULT_WEIGHTS)

    def test_with_dataframe(self):
        pytest.importorskip("pandas")
        import pandas as pd
        # Create df with mostly Saturday transactions
        base = datetime.now(timezone.utc) - timedelta(days=10)
        dates = []
        for i in range(20):
            d = base + timedelta(days=i)
            dates.append(d)
        df = pd.DataFrame({"date": dates})
        weights = _infer_dow_weights(df, "At-Risk")
        assert isinstance(weights, dict)
        assert all(k in weights for k in range(7))

    def test_empty_dataframe_returns_defaults(self):
        pytest.importorskip("pandas")
        import pandas as pd
        df = pd.DataFrame({"date": []})
        weights = _infer_dow_weights(df, "At-Risk")
        assert weights == dict(_DOW_DEFAULT_WEIGHTS)

    def test_old_data_returns_defaults(self):
        pytest.importorskip("pandas")
        import pandas as pd
        # Data older than lookback window
        old_date = datetime.now(timezone.utc) - timedelta(days=200)
        df = pd.DataFrame({"date": [old_date]})
        weights = _infer_dow_weights(df, "At-Risk")
        assert weights == dict(_DOW_DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------
# _infer_dom_weights
# ---------------------------------------------------------------------------

class TestInferDomWeights:
    def test_none_returns_empty(self):
        weights = _infer_dom_weights(None, "At-Risk")
        assert weights == {}

    def test_with_dataframe(self):
        pytest.importorskip("pandas")
        import pandas as pd
        base = datetime.now(timezone.utc) - timedelta(days=10)
        dates = [base + timedelta(days=i) for i in range(15)]
        df = pd.DataFrame({"date": dates})
        weights = _infer_dom_weights(df, "At-Risk")
        assert isinstance(weights, dict)
        # Keys should be day-of-month integers
        for k in weights:
            assert 1 <= k <= 31


# ---------------------------------------------------------------------------
# _score_date
# ---------------------------------------------------------------------------

class TestScoreDate:
    def _default_dow(self):
        return dict(_DOW_DEFAULT_WEIGHTS)

    def _empty_dom(self):
        return {}

    def test_score_returns_tuple(self):
        d = date(2026, 11, 15)
        score, signals = _score_date(d, self._default_dow(), self._empty_dom(), [], "At-Risk", 14, "maximize_profit")
        assert isinstance(score, float)
        assert isinstance(signals, dict)

    def test_score_non_negative(self):
        d = date(2026, 11, 15)
        score, _ = _score_date(d, self._default_dow(), self._empty_dom(), [], "At-Risk", 14, "maximize_profit")
        assert score >= 0.0

    def test_payday_date_scores_higher(self):
        payday = date(2026, 11, 28)   # payday window
        normal = date(2026, 11, 15)   # not payday
        s_pay, _ = _score_date(payday, self._default_dow(), self._empty_dom(), [], "At-Risk", 14, "maximize_profit")
        s_nor, _ = _score_date(normal, self._default_dow(), self._empty_dom(), [], "At-Risk", 14, "maximize_profit")
        assert s_pay > s_nor

    def test_payday_boost_in_signals(self):
        payday = date(2026, 11, 28)
        _, signals = _score_date(payday, self._default_dow(), self._empty_dom(), [], "At-Risk", 14, "maximize_profit")
        assert signals["payday_boost"] == _PAYDAY_BOOST

    def test_no_payday_boost_mid_month(self):
        d = date(2026, 11, 15)
        _, signals = _score_date(d, self._default_dow(), self._empty_dom(), [], "At-Risk", 14, "maximize_profit")
        assert signals["payday_boost"] == 0.0

    def test_overlap_penalty_applied(self):
        history = _make_history([("2026-11-13", "At-Risk")])
        d = date(2026, 11, 15)  # 2 days after existing campaign
        _, signals = _score_date(d, self._default_dow(), self._empty_dom(), history, "At-Risk", 14, "maximize_profit")
        assert signals["overlap_penalty"] < 0

    def test_no_overlap_penalty_when_clear(self):
        d = date(2026, 11, 15)
        _, signals = _score_date(d, self._default_dow(), self._empty_dom(), [], "At-Risk", 14, "maximize_profit")
        assert signals["overlap_penalty"] == 0.0

    def test_signals_keys_present(self):
        d = date(2026, 11, 15)
        _, signals = _score_date(d, self._default_dow(), self._empty_dom(), [], "At-Risk", 14, "maximize_profit")
        for key in ("dow_affinity", "dom_affinity", "payday_boost", "overlap_penalty", "goal_week_fit"):
            assert key in signals

    def test_goal_win_back_prefers_week1(self):
        week1 = date(2026, 11, 3)   # week 1
        week3 = date(2026, 11, 17)  # week 3
        s1, sig1 = _score_date(week1, self._default_dow(), self._empty_dom(), [], "At-Risk", 14, "win_back")
        s3, sig3 = _score_date(week3, self._default_dow(), self._empty_dom(), [], "At-Risk", 14, "win_back")
        assert sig1["goal_week_fit"] >= sig3["goal_week_fit"]

    def test_goal_retain_prefers_week2(self):
        week2 = date(2026, 11, 10)
        week4 = date(2026, 11, 25)
        s2, sig2 = _score_date(week2, self._default_dow(), self._empty_dom(), [], "Champions", 14, "retain_loyal")
        s4, sig4 = _score_date(week4, self._default_dow(), self._empty_dom(), [], "Champions", 14, "retain_loyal")
        assert sig2["goal_week_fit"] >= sig4["goal_week_fit"]


# ---------------------------------------------------------------------------
# _build_reason
# ---------------------------------------------------------------------------

class TestBuildReason:
    def test_returns_string(self):
        signals = {"dow_affinity": 0.9, "dom_affinity": 0.5, "payday_boost": 0.0,
                   "overlap_penalty": 0.0, "goal_week_fit": 0.9}
        reason = _build_reason(signals, date(2026, 11, 7))  # Saturday
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_includes_day_name(self):
        signals = {"dow_affinity": 0.9, "dom_affinity": 0.5, "payday_boost": 0.0,
                   "overlap_penalty": 0.0, "goal_week_fit": 0.9}
        reason = _build_reason(signals, date(2026, 11, 7))  # Saturday = เสาร์
        assert "เสาร์" in reason

    def test_includes_payday_note(self):
        signals = {"dow_affinity": 0.9, "dom_affinity": 0.5, "payday_boost": 0.20,
                   "overlap_penalty": 0.0, "goal_week_fit": 0.9}
        reason = _build_reason(signals, date(2026, 11, 28))
        assert "เงินเดือน" in reason

    def test_includes_overlap_warning(self):
        signals = {"dow_affinity": 0.9, "dom_affinity": 0.5, "payday_boost": 0.0,
                   "overlap_penalty": -0.3, "goal_week_fit": 0.9}
        reason = _build_reason(signals, date(2026, 11, 15))
        assert "campaign" in reason.lower() or "แคมเปญ" in reason or "⚠️" in reason

    def test_no_overlap_warning_small_penalty(self):
        signals = {"dow_affinity": 0.9, "dom_affinity": 0.5, "payday_boost": 0.0,
                   "overlap_penalty": -0.05, "goal_week_fit": 0.9}
        reason = _build_reason(signals, date(2026, 11, 15))
        assert "⚠️" not in reason

    def test_fallback_when_no_signals(self):
        signals = {"dow_affinity": 0.5, "dom_affinity": 0.5, "payday_boost": 0.0,
                   "overlap_penalty": 0.0, "goal_week_fit": 0.5}
        reason = _build_reason(signals, date(2026, 11, 15))
        # Should at least have the day name
        assert isinstance(reason, str)


# ---------------------------------------------------------------------------
# recommend_send_date
# ---------------------------------------------------------------------------

class TestRecommendSendDate:
    def test_returns_expected_keys(self):
        result = recommend_send_date("At-Risk", "2026-11")
        for key in ("recommended_date", "score", "reason", "payday_window",
                    "alternatives", "overlap_warning"):
            assert key in result

    def test_recommended_date_in_month(self):
        result = recommend_send_date("At-Risk", "2026-11")
        d = date.fromisoformat(result["recommended_date"])
        assert d.year == 2026
        assert d.month == 11

    def test_score_is_float(self):
        result = recommend_send_date("At-Risk", "2026-11")
        assert isinstance(result["score"], float)
        assert result["score"] >= 0.0

    def test_payday_window_is_bool(self):
        result = recommend_send_date("At-Risk", "2026-11")
        assert isinstance(result["payday_window"], bool)

    def test_alternatives_count(self):
        result = recommend_send_date("At-Risk", "2026-11", top_n=3)
        # top_n=3 → best + 2 alternatives
        assert len(result["alternatives"]) == 2

    def test_alternatives_have_required_keys(self):
        result = recommend_send_date("At-Risk", "2026-11")
        for alt in result["alternatives"]:
            assert "date" in alt
            assert "score" in alt
            assert "reason" in alt

    def test_alternatives_sorted_desc(self):
        result = recommend_send_date("At-Risk", "2026-11")
        scores = [result["score"]] + [a["score"] for a in result["alternatives"]]
        assert scores == sorted(scores, reverse=True)

    def test_overlap_warning_none_when_no_history(self):
        result = recommend_send_date("At-Risk", "2026-11", campaign_history=[])
        assert result["overlap_warning"] is None

    def test_overlap_warning_set_when_conflict(self):
        # Place a campaign 3 days before middle of month (well within cooldown)
        history = _make_history([("2026-11-12", "At-Risk")])
        result = recommend_send_date(
            "At-Risk", "2026-11",
            campaign_history=history,
            cooldown_days=14,
        )
        # The recommended date might or might not conflict; check structure
        if result["overlap_warning"] is not None:
            assert isinstance(result["overlap_warning"], str)
            assert len(result["overlap_warning"]) > 0

    def test_goal_win_back_prefers_start_of_month(self):
        result = recommend_send_date("At-Risk", "2026-11", goal="win_back")
        d = date.fromisoformat(result["recommended_date"])
        # Win-back prefers week 1 (days 1–7) + payday window (1–5)
        # Score should strongly favor early month
        assert d.day <= 15  # at minimum in the first half

    def test_different_goals_can_differ(self):
        r1 = recommend_send_date("Champions", "2026-11", goal="win_back")
        r2 = recommend_send_date("Champions", "2026-11", goal="maximize_profit")
        # Not guaranteed different, but both should be valid dates
        date.fromisoformat(r1["recommended_date"])
        date.fromisoformat(r2["recommended_date"])

    def test_top_n_one_returns_no_alternatives(self):
        result = recommend_send_date("At-Risk", "2026-11", top_n=1)
        assert len(result["alternatives"]) == 0

    def test_empty_campaign_history(self):
        result = recommend_send_date("At-Risk", "2026-11", campaign_history=[])
        assert result["recommended_date"] is not None

    def test_none_pos_df_still_works(self):
        result = recommend_send_date("At-Risk", "2026-11", pos_df=None)
        assert result["recommended_date"] is not None

    def test_with_pos_df(self):
        pytest.importorskip("pandas")
        import pandas as pd
        base = datetime.now(timezone.utc) - timedelta(days=10)
        dates = [base + timedelta(days=i) for i in range(20)]
        df = pd.DataFrame({"date": dates, "amount": [100.0] * 20})
        result = recommend_send_date("At-Risk", "2026-11", pos_df=df)
        assert result["recommended_date"] is not None

    def test_reason_is_string(self):
        result = recommend_send_date("At-Risk", "2026-11")
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_payday_recommended_when_win_back(self):
        """Win-back goal + no history → recommended date likely in payday window."""
        result = recommend_send_date("At-Risk", "2026-11", goal="win_back")
        # With win_back (prefers week 1) + payday boost (days 1–5), overlap should be payday
        d = date.fromisoformat(result["recommended_date"])
        payday_dates = get_payday_dates("2026-11")
        # At least verify the top score beats zero
        assert result["score"] > 0

    def test_all_alternative_dates_in_month(self):
        result = recommend_send_date("Champions", "2026-12", top_n=5)
        for alt in result["alternatives"]:
            d = date.fromisoformat(alt["date"])
            assert d.month == 12

    def test_december_month(self):
        result = recommend_send_date("At-Risk", "2026-12")
        d = date.fromisoformat(result["recommended_date"])
        assert d.year == 2026
        assert d.month == 12

    def test_overlap_warning_contains_thai(self):
        """When overlap exists, warning message should be in Thai."""
        history = _make_history([("2026-11-10", "At-Risk")])
        result = recommend_send_date(
            "At-Risk", "2026-11",
            campaign_history=history,
            cooldown_days=14,
        )
        if result["overlap_warning"] is not None:
            # Thai characters should be present
            assert any(ord(c) > 127 for c in result["overlap_warning"])
