"""Tests for content_calendar.py — monthly content mix planner."""
from __future__ import annotations

import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kit():
    from content_generator import BrandKit
    return BrandKit(brand_name="คาเฟ่ทดสอบ", category="food_and_beverage")


@pytest.fixture
def mix_30_60_10():
    from content_calendar import ContentMixConfig
    return ContentMixConfig(knowledge=0.30, story=0.60, sales=0.10, pieces_per_month=12)


@pytest.fixture
def ml_actions():
    return [
        {"segment": "At-Risk",   "action": "Win-Back",  "promo_type": "discount_15", "top_product": "Latte"},
        {"segment": "Champions", "action": "VIP Reward","promo_type": "points_x2",   "top_product": "Croissant"},
    ]


# ===========================================================================
# ContentMixConfig
# ===========================================================================

class TestContentMixConfig:

    def test_valid_config_instantiates(self):
        from content_calendar import ContentMixConfig
        cfg = ContentMixConfig(knowledge=0.30, story=0.60, sales=0.10, pieces_per_month=12)
        assert cfg.knowledge == 0.30

    def test_ratios_not_summing_to_1_raises(self):
        from content_calendar import ContentMixConfig
        with pytest.raises(ValueError, match="sum to 1.0"):
            ContentMixConfig(knowledge=0.40, story=0.40, sales=0.10)

    def test_zero_pieces_raises(self):
        from content_calendar import ContentMixConfig
        with pytest.raises(ValueError, match="pieces_per_month"):
            ContentMixConfig(knowledge=0.30, story=0.60, sales=0.10, pieces_per_month=0)

    def test_tolerance_accepts_small_float_error(self):
        """0.1 + 0.2 + 0.7 may not be exactly 1.0 in float — should still pass."""
        from content_calendar import ContentMixConfig
        cfg = ContentMixConfig(knowledge=0.1, story=0.2, sales=0.7, pieces_per_month=10)
        assert cfg is not None

    def test_default_values(self):
        from content_calendar import ContentMixConfig
        cfg = ContentMixConfig()
        assert cfg.knowledge == 0.30
        assert cfg.story == 0.60
        assert cfg.sales == 0.10
        assert cfg.pieces_per_month == 12


# ===========================================================================
# plan_content_calendar — counts
# ===========================================================================

class TestCalendarCounts:

    def test_total_slots_equals_pieces_per_month(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        assert len(plan.slots) == mix_30_60_10.pieces_per_month

    def test_knowledge_count_correct(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        # round(12 * 0.30) = 4
        assert plan.knowledge_count == 4

    def test_sales_count_correct(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        # round(12 * 0.10) = 1
        assert plan.sales_count == 1

    def test_story_count_absorbs_remainder(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        assert plan.knowledge_count + plan.story_count + plan.sales_count == 12

    def test_slot_categories_match_counts(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        from collections import Counter
        cats = Counter(s.category for s in plan.slots)
        assert cats["knowledge"] == plan.knowledge_count
        assert cats["story"]     == plan.story_count
        assert cats["sales"]     == plan.sales_count

    def test_8_pieces_distributes_correctly(self, kit):
        from content_calendar import ContentMixConfig, plan_content_calendar
        mix = ContentMixConfig(knowledge=0.25, story=0.50, sales=0.25, pieces_per_month=8)
        plan = plan_content_calendar(mix, kit)
        assert len(plan.slots) == 8
        assert plan.knowledge_count + plan.story_count + plan.sales_count == 8


# ===========================================================================
# plan_content_calendar — ML actions + overflow
# ===========================================================================

class TestCalendarMLActions:

    def test_ml_actions_assigned_to_sales_slots(self, kit, mix_30_60_10, ml_actions):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit, campaign_actions=ml_actions)
        # sales_count=1, first action assigned
        sales_slots = [s for s in plan.slots if s.category == "sales"]
        assert len(sales_slots) == 1
        assert sales_slots[0].brief is not None
        assert sales_slots[0].brief["segment"] == "At-Risk"

    def test_overflow_when_actions_exceed_sales_quota(self, kit, mix_30_60_10, ml_actions):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit, campaign_actions=ml_actions)
        # sales_count=1 but 2 actions → 1 overflow
        assert len(plan.overflow_actions) == 1
        assert plan.overflow_actions[0]["segment"] == "Champions"

    def test_no_overflow_when_actions_within_quota(self, kit):
        from content_calendar import ContentMixConfig, plan_content_calendar
        mix = ContentMixConfig(knowledge=0.20, story=0.60, sales=0.20, pieces_per_month=10)
        actions = [{"segment": "At-Risk", "action": "Win-Back", "promo_type": "discount_15"}]
        plan = plan_content_calendar(mix, kit, campaign_actions=actions)
        assert plan.overflow_actions == []

    def test_no_actions_sales_slots_have_no_brief(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit, campaign_actions=[])
        sales_slots = [s for s in plan.slots if s.category == "sales"]
        for slot in sales_slots:
            assert slot.brief is None

    def test_brief_dict_has_required_fields(self, kit, mix_30_60_10, ml_actions):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit, campaign_actions=ml_actions)
        sales_slot = next(s for s in plan.slots if s.category == "sales")
        assert sales_slot.brief is not None
        for field in ("segment", "action", "promo_type", "channel"):
            assert field in sales_slot.brief


# ===========================================================================
# plan_content_calendar — slot structure
# ===========================================================================

class TestCalendarSlotStructure:

    def test_slot_numbers_are_sequential(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        numbers = [s.slot_number for s in plan.slots]
        assert numbers == list(range(1, 13))

    def test_weeks_are_1_to_4(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        weeks = {s.week for s in plan.slots}
        assert weeks <= {1, 2, 3, 4}
        assert 1 in weeks and 4 in weeks

    def test_all_slots_have_topic_suggestion(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        for slot in plan.slots:
            assert slot.topic_suggestion and len(slot.topic_suggestion) > 5

    def test_all_slots_have_day_of_week(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        valid_days = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
        for slot in plan.slots:
            assert slot.day_of_week in valid_days

    def test_content_types_valid(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        valid_types = {
            "new_menu", "service_highlight",
            "brand_story", "bestseller", "seasonal",
            "ml_promo",
        }
        plan = plan_content_calendar(mix_30_60_10, kit)
        for slot in plan.slots:
            assert slot.content_type in valid_types, f"unexpected: {slot.content_type}"

    def test_ml_promo_slots_have_sales_category(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        for slot in plan.slots:
            if slot.content_type == "ml_promo":
                assert slot.category == "sales"

    def test_knowledge_types_in_knowledge_category(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        for slot in plan.slots:
            if slot.content_type in ("new_menu", "service_highlight"):
                assert slot.category == "knowledge"

    def test_story_types_in_story_category(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        for slot in plan.slots:
            if slot.content_type in ("brand_story", "bestseller", "seasonal"):
                assert slot.category == "story"


# ===========================================================================
# plan_content_calendar — metadata
# ===========================================================================

class TestCalendarMetadata:

    def test_month_stored_in_plan(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit, month="2026-11")
        assert plan.month == "2026-11"

    def test_month_defaults_to_current(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        from datetime import datetime, timezone
        plan = plan_content_calendar(mix_30_60_10, kit)
        expected = datetime.now(timezone.utc).strftime("%Y-%m")
        assert plan.month == expected

    def test_mix_config_stored(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        assert plan.mix_config is mix_30_60_10

    def test_total_pieces_stored(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        assert plan.total_pieces == 12


# ===========================================================================
# summarise_calendar
# ===========================================================================

class TestSummariseCalendar:

    def test_summary_contains_month(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar, summarise_calendar
        plan = plan_content_calendar(mix_30_60_10, kit, month="2026-11")
        summary = summarise_calendar(plan)
        assert "2026-11" in summary

    def test_summary_contains_counts(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar, summarise_calendar
        plan = plan_content_calendar(mix_30_60_10, kit)
        summary = summarise_calendar(plan)
        assert "12" in summary   # total pieces

    def test_summary_mentions_overflow(self, kit, mix_30_60_10, ml_actions):
        from content_calendar import plan_content_calendar, summarise_calendar
        plan = plan_content_calendar(mix_30_60_10, kit, campaign_actions=ml_actions)
        summary = summarise_calendar(plan)
        assert "เกินโควต้า" in summary or "overflow" in summary.lower() or "เลื่อน" in summary

    def test_summary_no_overflow_message_when_clean(self, kit, mix_30_60_10):
        from content_calendar import plan_content_calendar, summarise_calendar
        plan = plan_content_calendar(mix_30_60_10, kit, campaign_actions=[])
        summary = summarise_calendar(plan)
        assert "เกินโควต้า" not in summary


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_all_sales_mix(self, kit):
        from content_calendar import ContentMixConfig, plan_content_calendar
        mix = ContentMixConfig(knowledge=0.0, story=0.0, sales=1.0, pieces_per_month=4)
        plan = plan_content_calendar(mix, kit)
        assert len(plan.slots) == 4
        assert all(s.category == "sales" for s in plan.slots)

    def test_no_sales_mix(self, kit):
        from content_calendar import ContentMixConfig, plan_content_calendar
        mix = ContentMixConfig(knowledge=0.40, story=0.60, sales=0.0, pieces_per_month=10)
        plan = plan_content_calendar(mix, kit)
        assert plan.sales_count == 0
        assert all(s.category != "sales" for s in plan.slots)

    def test_single_piece_plan(self, kit):
        from content_calendar import ContentMixConfig, plan_content_calendar
        mix = ContentMixConfig(knowledge=0.0, story=1.0, sales=0.0, pieces_per_month=1)
        plan = plan_content_calendar(mix, kit)
        assert len(plan.slots) == 1

    def test_many_actions_all_overflow_when_zero_sales(self, kit, ml_actions):
        from content_calendar import ContentMixConfig, plan_content_calendar
        mix = ContentMixConfig(knowledge=0.50, story=0.50, sales=0.0, pieces_per_month=4)
        plan = plan_content_calendar(mix, kit, campaign_actions=ml_actions)
        assert len(plan.overflow_actions) == len(ml_actions)
