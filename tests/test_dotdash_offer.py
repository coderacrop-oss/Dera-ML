"""infer_dotdash_offer: the public, key-driven way to get the DotDash offer for a decision."""
import logging

import pytest

from action_rules import ACTION_KEYS, get_dotdash_offer_by_id, infer_dotdash_offer

BASE_FIELDS = {"offer_type", "min_spend", "expiry_days"}


@pytest.mark.parametrize("key", sorted(ACTION_KEYS))
def test_every_key_yields_a_complete_offer(key):
    offer = infer_dotdash_offer(key)
    assert offer["action_key"] == key
    assert BASE_FIELDS <= set(offer)
    if offer["offer_type"] == "points":
        assert offer["points_amount"] > 0
    else:
        assert offer["offer_type"] == "coupon"
        assert offer["coupon_type"] in ("fixed", "percent") and offer["coupon_value"] > 0


def test_unknown_key_raises_instead_of_guessing():
    with pytest.raises(ValueError, match="no_such_rule"):
        infer_dotdash_offer("no_such_rule")
    with pytest.raises(ValueError):
        get_dotdash_offer_by_id("no_such_rule")


def test_key_wins_over_whatever_the_action_text_says():
    offer = infer_dotdash_offer("st_vip_reward_offpeak", "some renamed action")
    assert offer["offer_type"] == "points" and offer["points_amount"] == 100


@pytest.mark.parametrize("text", ["Do Nothing (Lost Cause)", "Zero-Cost: เราคิดถึงคุณ", ""])
def test_no_key_and_no_promotion_is_a_zero_point_offer(text):
    assert infer_dotdash_offer(None, text) == {"offer_type": "points", "points_amount": 0,
                                                "min_spend": 0, "expiry_days": 0, "action_key": None}


def test_legacy_text_path_still_answers_but_warns(caplog):
    with caplog.at_level(logging.WARNING):
        offer = infer_dotdash_offer(None, "VIP Reward: แจกแต้ม X2")
    assert offer["points_amount"] == 100 and offer["action_key"] is None
    assert any("action_key" in r.message for r in caplog.records)


def test_planner_keeps_its_private_alias():
    import campaign_planner as cp
    assert cp._infer_dotdash_offer("Loyal Offer: ชุด A")["coupon_value"] == 5.0
