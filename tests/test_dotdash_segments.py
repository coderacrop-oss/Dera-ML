"""dotdash_segments: one group per (segment, action_key) so 1 group = 1 reward (DotDash item 1)."""
import logging

import pandas as pd
import pytest

from campaign_planner import _build_dotdash_segments

VIP = "VIP Reward: แจกแต้มพิเศษ 100 แต้ม"
NURTURE = "Loyalty Nurture: แต้มสะสม"
WINBACK = "VIP Win-Back: กลับมาเป็น VIP อีกครั้ง"


def _frame(rows):
    return pd.DataFrame(rows, columns=["customer_id", "segment_name", "recommended_action",
                                       "action_key", "expected_profit", "promo_cost"])


@pytest.fixture
def mixed():
    return _frame([
        ("a1", "Champions", VIP, "st_vip_reward_offpeak", 10.0, 2.0),
        ("a2", "Champions", VIP, "st_vip_reward_offpeak", 12.0, 2.0),
        ("a3", "Champions", NURTURE, "st_loyalty_nurture", 3.0, 1.0),     # same segment, other action
        ("b1", "At-Risk", WINBACK, "st_cantlose_winback", -4.0, 6.0),
        ("b2", "At-Risk", NURTURE, "st_loyalty_nurture", 1.0, 1.0),
    ])


def test_a_segment_with_two_actions_becomes_two_groups(mixed):
    groups = _build_dotdash_segments(mixed)
    assert sorted((g["segment_name"], g["action_key"]) for g in groups) == [
        ("At-Risk", "st_cantlose_winback"), ("At-Risk", "st_loyalty_nurture"),
        ("Champions", "st_loyalty_nurture"), ("Champions", "st_vip_reward_offpeak")]


def test_nobody_is_lost_or_duplicated(mixed):
    groups = _build_dotdash_segments(mixed)
    ids = [c for g in groups for c in g["customer_ids"]]
    assert sorted(ids) == sorted(mixed["customer_id"]) and len(ids) == len(set(ids))
    assert sum(g["customer_count"] for g in groups) == len(mixed)


def test_customer_gets_own_offer(mixed):
    by_cust = mixed.set_index("customer_id")["action_key"]
    for g in _build_dotdash_segments(mixed):
        assert g["dotdash_offer"]["action_key"] == g["action_key"]
        assert all(by_cust[c] == g["action_key"] for c in g["customer_ids"])


def test_group_sums_and_fields(mixed):
    g = next(x for x in _build_dotdash_segments(mixed)
             if (x["segment_name"], x["action_key"]) == ("Champions", "st_vip_reward_offpeak"))
    assert g["customer_count"] == 2 and g["expected_profit"] == 22.0 and g["actual_spend"] == 4.0
    assert g["recommended_action"] == VIP
    assert g["dotdash_offer"]["points_amount"] == 100


def test_groups_are_sorted_by_expected_profit_desc(mixed):
    profits = [g["expected_profit"] for g in _build_dotdash_segments(mixed)]
    assert profits == sorted(profits, reverse=True)


def test_empty_selection_gives_no_groups():
    assert _build_dotdash_segments(_frame([])) == []


def test_single_customer_single_group():
    groups = _build_dotdash_segments(_frame([("z", "Loyal", VIP, "loyal_vip_reward", 1.0, 1.0)]))
    assert len(groups) == 1 and groups[0]["customer_ids"] == ["z"]


def test_rows_without_a_key_fall_back_to_action_text_with_a_warning(caplog):
    legacy = _frame([("x1", "Loyal", VIP, None, 1.0, 1.0), ("x2", "Loyal", NURTURE, None, 1.0, 1.0)])
    with caplog.at_level(logging.WARNING):
        groups = _build_dotdash_segments(legacy)
    assert len(groups) == 2                       # still split by action
    assert all(g["action_key"] is None and g["dotdash_offer"]["action_key"] is None for g in groups)
    assert any("action_key" in r.message for r in caplog.records)


def test_frame_without_the_action_key_column_still_works():
    legacy = _frame([("x1", "Loyal", VIP, None, 1.0, 1.0)]).drop(columns=["action_key"])
    assert len(_build_dotdash_segments(legacy)) == 1


def test_planner_run_end_to_end_every_selected_customer_is_in_exactly_one_group(tmp_path, monkeypatch):
    """Real planner on a generated POS file (no champion registry -> the rule-based fallback)."""
    import campaign_planner as cp
    from fixtures_pos import make_pos_csv
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cp, "REGISTRY_PATH", str(tmp_path / "models" / "registry"))
    csv = make_pos_csv(tmp_path, n_customers=150, seed=4)
    report = cp.run(budget=10_000_000.0, pos_file=csv)
    groups = report["dotdash_segments"]
    assert report["customers_selected"] > 0 and groups
    ids = [c for g in groups for c in g["customer_ids"]]
    assert len(ids) == len(set(ids)) == report["customers_selected"]
    assert all(g["action_key"] and g["dotdash_offer"]["action_key"] == g["action_key"] for g in groups)
