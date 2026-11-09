"""Tests for lead_scorer.py."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lead(**kwargs) -> dict:
    base = {
        "customer_id":          "C1",
        "tags":                 [],
        "has_premium_purchase": False,
        "total_purchases":      0,
        "last_activity_days":   None,
    }
    base.update(kwargs)
    return base


# ===========================================================================
# score_lead — tag signals
# ===========================================================================

class TestScoreLeadTagSignals:

    def test_asked_price_adds_40(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(tags=["asked_price"]))
        assert r["score"] == 40
        assert "asked_price" in r["signals"]

    def test_interested_premium_adds_20(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(tags=["interested_premium"]))
        assert r["score"] == 20

    def test_replied_adds_25(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(tags=["replied"]))
        assert r["score"] == 25

    def test_replied_with_suffix_matches(self):
        """replied_nov2026 should match the "replied" prefix rule."""
        from lead_scorer import score_lead
        r = score_lead(_lead(tags=["replied_nov2026"]))
        assert "replied" in r["signals"]

    def test_clicked_adds_15(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(tags=["clicked"]))
        assert r["score"] == 15

    def test_clicked_with_suffix_matches(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(tags=["clicked_promo_oct"]))
        assert "clicked" in r["signals"]

    def test_viewed_adds_5(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(tags=["viewed"]))
        assert r["score"] == 5

    def test_vip_adds_10(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(tags=["vip"]))
        assert r["score"] == 10

    def test_unknown_tag_ignored(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(tags=["some_random_tag"]))
        assert r["score"] == 0
        assert r["signals"] == []

    def test_multiple_tags_cumulative(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(tags=["asked_price", "replied", "clicked"]))
        # 40 + 25 + 15 = 80
        assert r["score"] == 80

    def test_tags_case_insensitive(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(tags=["ASKED_PRICE"]))
        assert "asked_price" in r["signals"]


# ===========================================================================
# score_lead — purchase signals
# ===========================================================================

class TestScoreLeadPurchaseSignals:

    def test_premium_purchase_adds_30(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(has_premium_purchase=True))
        assert r["score"] == 30
        assert "premium_purchase" in r["signals"]

    def test_no_premium_purchase_no_bonus(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(has_premium_purchase=False))
        assert r["score"] == 0

    def test_repeat_buyer_3_adds_10(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(total_purchases=3))
        assert "repeat_buyer" in r["signals"]
        assert r["score"] == 10

    def test_repeat_buyer_below_threshold_no_bonus(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(total_purchases=2))
        assert "repeat_buyer" not in r["signals"]

    def test_premium_plus_repeat(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(has_premium_purchase=True, total_purchases=5))
        assert r["score"] == 40   # 30 + 10


# ===========================================================================
# score_lead — recency bonus
# ===========================================================================

class TestScoreLeadRecency:

    def test_active_7d_adds_10(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(last_activity_days=5))
        assert "active_7d" in r["signals"]
        assert r["score"] == 10

    def test_exactly_7d_qualifies(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(last_activity_days=7))
        assert "active_7d" in r["signals"]

    def test_active_8_to_30d_adds_5(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(last_activity_days=20))
        assert "active_30d" in r["signals"]
        assert r["score"] == 5

    def test_exactly_30d_qualifies_30d_bonus(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(last_activity_days=30))
        assert "active_30d" in r["signals"]

    def test_over_30d_no_recency_bonus(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(last_activity_days=45))
        assert "active_7d" not in r["signals"]
        assert "active_30d" not in r["signals"]

    def test_none_recency_no_bonus(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(last_activity_days=None))
        assert r["score"] == 0


# ===========================================================================
# score_lead — tier assignment
# ===========================================================================

class TestScoreLeadTier:

    def test_score_60_is_hot(self):
        from lead_scorer import score_lead
        # asked_price(40) + replied(25) = 65 → hot
        r = score_lead(_lead(tags=["asked_price", "replied"]))
        assert r["tier"] == "hot"

    def test_score_35_is_warm(self):
        from lead_scorer import score_lead
        # clicked(15) + premium_purchase(30) = 45 → warm
        r = score_lead(_lead(tags=["clicked"], has_premium_purchase=True))
        assert r["tier"] == "warm"

    def test_score_10_is_cold(self):
        from lead_scorer import score_lead
        # vip(10) = 10 → cold
        r = score_lead(_lead(tags=["vip"]))
        assert r["tier"] == "cold"

    def test_score_below_10_is_unfit(self):
        from lead_scorer import score_lead
        r = score_lead(_lead())
        assert r["tier"] == "unfit"

    def test_score_exactly_60_is_hot(self):
        from lead_scorer import score_lead
        # asked_price(40) + interested_premium(20) = 60 → hot
        r = score_lead(_lead(tags=["asked_price", "interested_premium"]))
        assert r["score"] == 60
        assert r["tier"] == "hot"


# ===========================================================================
# score_lead — output structure
# ===========================================================================

class TestScoreLeadOutput:

    def test_all_keys_present(self):
        from lead_scorer import score_lead
        r = score_lead(_lead())
        for key in ("customer_id", "score", "tier", "signals", "scored_at"):
            assert key in r

    def test_customer_id_preserved(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(customer_id="CUST_999"))
        assert r["customer_id"] == "CUST_999"

    def test_scored_at_override(self):
        from lead_scorer import score_lead
        r = score_lead(_lead(scored_at="2026-11-01T00:00:00+00:00"))
        assert r["scored_at"] == "2026-11-01T00:00:00+00:00"

    def test_no_mutation_of_input(self):
        from lead_scorer import score_lead
        lead = _lead(tags=["clicked"])
        original_tags = list(lead["tags"])
        score_lead(lead)
        assert lead["tags"] == original_tags


# ===========================================================================
# score_leads_df
# ===========================================================================

class TestScoreLeadsDf:

    def _make_df(self, rows):
        return pd.DataFrame(rows)

    def test_adds_score_and_tier_columns(self):
        from lead_scorer import score_leads_df
        df = self._make_df([
            {"customer_id": "C1", "tags": ["asked_price"], "has_premium_purchase": False,
             "total_purchases": 1, "last_activity_days": 5},
        ])
        result = score_leads_df(df)
        assert "lead_score" in result.columns
        assert "lead_tier"  in result.columns

    def test_does_not_modify_original_df(self):
        from lead_scorer import score_leads_df
        df = self._make_df([{"customer_id": "C1", "tags": []}])
        score_leads_df(df)
        assert "lead_score" not in df.columns

    def test_multiple_rows_scored_independently(self):
        from lead_scorer import score_leads_df
        df = self._make_df([
            {"customer_id": "C1", "tags": ["asked_price", "replied"],
             "has_premium_purchase": False, "total_purchases": 0, "last_activity_days": None},
            {"customer_id": "C2", "tags": [],
             "has_premium_purchase": False, "total_purchases": 0, "last_activity_days": None},
        ])
        result = score_leads_df(df)
        assert result.loc[result["customer_id"] == "C1", "lead_score"].iloc[0] > 0
        assert result.loc[result["customer_id"] == "C2", "lead_score"].iloc[0] == 0

    def test_lead_signals_column_present(self):
        from lead_scorer import score_leads_df
        df = self._make_df([{"customer_id": "C1", "tags": ["vip"]}])
        result = score_leads_df(df)
        assert "lead_signals" in result.columns
        assert "vip" in result.loc[0, "lead_signals"]

    def test_missing_optional_columns_handled(self):
        """DataFrame with only customer_id should not raise."""
        from lead_scorer import score_leads_df
        df = self._make_df([{"customer_id": "C1"}])
        result = score_leads_df(df)
        assert result.loc[0, "lead_score"] == 0

    def test_custom_tags_col(self):
        from lead_scorer import score_leads_df
        df = self._make_df([{"customer_id": "C1", "line_tags": ["asked_price"]}])
        result = score_leads_df(df, tags_col="line_tags")
        assert result.loc[0, "lead_score"] == 40


# ===========================================================================
# get_priority_leads
# ===========================================================================

class TestGetPriorityLeads:

    def _results(self):
        from lead_scorer import score_lead
        leads = [
            _lead(customer_id="hot1",  tags=["asked_price", "replied"]),       # 65 hot
            _lead(customer_id="warm1", tags=["clicked"], has_premium_purchase=True),  # 45 warm
            _lead(customer_id="cold1", tags=["viewed"], last_activity_days=5),  # 15 cold
            _lead(customer_id="unfit1"),                                         # 0 unfit
        ]
        return [score_lead(l) for l in leads]

    def test_filter_by_hot_tier(self):
        from lead_scorer import get_priority_leads
        results = self._results()
        hot = get_priority_leads(results, tier="hot")
        assert all(r["tier"] == "hot" for r in hot)
        assert len(hot) == 1
        assert hot[0]["customer_id"] == "hot1"

    def test_filter_by_warm_tier(self):
        from lead_scorer import get_priority_leads
        results = self._results()
        warm = get_priority_leads(results, tier="warm")
        assert len(warm) == 1

    def test_no_tier_filter_returns_all(self):
        from lead_scorer import get_priority_leads
        results = self._results()
        all_leads = get_priority_leads(results, tier="")
        assert len(all_leads) == 4

    def test_sorted_descending_by_score(self):
        from lead_scorer import get_priority_leads
        results = self._results()
        all_leads = get_priority_leads(results, tier="")
        scores = [r["score"] for r in all_leads]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filter(self):
        from lead_scorer import get_priority_leads
        results = self._results()
        filtered = get_priority_leads(results, tier="", min_score=40)
        assert all(r["score"] >= 40 for r in filtered)

    def test_works_with_dataframe(self):
        from lead_scorer import score_leads_df, get_priority_leads
        df = pd.DataFrame([
            {"customer_id": "C1", "tags": ["asked_price", "replied"]},
            {"customer_id": "C2", "tags": []},
        ])
        scored = score_leads_df(df)
        hot = get_priority_leads(scored, tier="hot")
        assert len(hot) == 1


# ===========================================================================
# get_lead_report
# ===========================================================================

class TestGetLeadReport:

    def _scored_results(self):
        from lead_scorer import score_lead
        leads = [
            _lead(customer_id="A", tags=["asked_price", "replied"]),
            _lead(customer_id="B", tags=["clicked"], has_premium_purchase=True),
            _lead(customer_id="C", tags=["viewed"]),
            _lead(customer_id="D"),
        ]
        return [score_lead(l) for l in leads]

    def test_contains_tier_counts(self):
        from lead_scorer import get_lead_report
        report = get_lead_report(self._scored_results())
        assert "Hot" in report or "hot" in report
        assert "Warm" in report or "warm" in report

    def test_contains_total_leads(self):
        from lead_scorer import get_lead_report
        report = get_lead_report(self._scored_results())
        assert "4" in report

    def test_contains_thai_text(self):
        from lead_scorer import get_lead_report
        report = get_lead_report(self._scored_results())
        assert "Lead" in report
        assert "Premium" in report or "โทร" in report

    def test_top_leads_listed(self):
        from lead_scorer import get_lead_report
        report = get_lead_report(self._scored_results())
        assert "A" in report   # hot customer_id

    def test_empty_input_returns_message(self):
        from lead_scorer import get_lead_report
        report = get_lead_report([])
        assert "ไม่มีข้อมูล" in report

    def test_works_with_scored_dataframe(self):
        from lead_scorer import score_leads_df, get_lead_report
        df = pd.DataFrame([
            {"customer_id": "C1", "tags": ["asked_price"]},
            {"customer_id": "C2", "tags": []},
        ])
        scored = score_leads_df(df)
        report = get_lead_report(scored)
        assert "Lead" in report
