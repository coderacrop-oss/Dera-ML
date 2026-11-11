"""Every decision the engine takes carries a stable action_key (the rule id), so consumers never parse action names."""
import pandas as pd
import pytest

import uplift_engine as l4
from action_rules import (ACTION_KEYS, LOW_BASKET_RULES, PERSUADABLE_RULES, SURE_THING_RULES,
                          get_dotdash_offer_by_id)
from business_config import CONFIG
from fixtures_pos import make_pos_csv


@pytest.fixture(scope="module")
def decided(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ak")
    csv = make_pos_csv(tmp, n_customers=400, seed=7)
    ctx = l4.build_serving_context(csv, None)
    model = l4._train_xlearner(ctx.raw_df[ctx.raw_df["was_treated"] == 1],
                               ctx.raw_df[ctx.raw_df["was_treated"] == 0], ctx.features)
    _total, frame = l4.decide_for_customers(model, ctx)
    return frame


def test_action_keys_unique_and_cover_all_rule_lists():
    ids = [r["id"] for r in SURE_THING_RULES + PERSUADABLE_RULES + LOW_BASKET_RULES]
    assert len(ids) == len(set(ids)), "duplicate rule id"
    assert set(ACTION_KEYS) == set(ids)


def test_every_sent_row_has_a_known_action_key(decided):
    sent = decided[decided["should_send"] == 1]
    assert len(sent) > 0, "fixture must produce at least one sent customer"
    assert sent["action_key"].notna().all()
    assert sent["action_key"].isin(list(ACTION_KEYS)).all()


def test_action_key_matches_the_action_text_of_its_rule(decided):
    sent = decided[decided["should_send"] == 1]
    for _, row in sent.iterrows():
        prefix = ACTION_KEYS[row["action_key"]]["action"].split("{")[0]
        assert row["recommended_action"].startswith(prefix), (row["action_key"], row["recommended_action"])


def test_do_nothing_rows_have_no_key(decided):
    dn = decided[decided["recommended_action"].str.startswith(("Do Nothing", "Zero-Cost"))]
    assert len(dn) > 0
    assert dn["action_key"].isna().all()


def test_milestone_offer_points_follow_config(monkeypatch):
    monkeypatch.setitem(CONFIG, "milestone_points", 70)
    assert get_dotdash_offer_by_id("low_milestone")["points_amount"] == 70


def test_every_low_basket_rule_has_a_dotdash_offer():
    for rule in LOW_BASKET_RULES:
        offer = get_dotdash_offer_by_id(rule["id"])
        assert offer["offer_type"] in ("points", "coupon")


def test_low_basket_rule_texts_are_exactly_what_the_engine_always_sent():
    """The engine now renders these from the rules; the customer-facing text must not change."""
    r = {x["id"]: x["action"] for x in LOW_BASKET_RULES}
    assert r["low_milestone"].format(basket=150.0, pts=50) == "Milestone Reward: สั่งครบ 150 บาท รับ 50 แต้ม"
    assert r["low_welcome"] == "Welcome: สั่งครั้งที่ 2 รับส่วนลด 10%"
    assert r["low_abouttosleep_miss_you"] == "เราคิดถึงคุณ: กลับมาหาเราก่อนที่จะไม่ได้เจอกัน"
    assert r["low_hibernating_nudge"] == "Reactivation Nudge: กลับมาช้อปรับส่วนลดพิเศษ"
    assert r["low_churn_winback"] == "Early Win-Back: กลับมาหาเรา รับส่วนลดพิเศษ"
    assert r["low_bundle_upsell"].format(bundle="ลาเต้ + เค้ก") == "Bundle Upsell: เพิ่ม ลาเต้ + เค้ก"


def test_a_low_basket_decision_is_keyed(decided):
    """The fixture's data reaches the low-basket path (adaptive threshold) — that row is keyed too."""
    low = decided[decided["action_key"].astype("string").str.startswith("low_", na=False)]
    assert len(low) > 0
    assert (low["should_send"] == 1).all()
