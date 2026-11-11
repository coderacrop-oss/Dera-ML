"""bundle_items / bundle_coverage / bundle_low_coverage per (segment x action) group (DotDash item 5)."""
import pandas as pd
import pytest

from bundle_pairs import group_bundle, personal_pairs
from campaign_planner import _build_dotdash_segments


def _raw(rows):
    return pd.DataFrame(rows, columns=["customer_id", "transaction_id", "product_name"])


def test_personal_pair_is_the_most_frequent_pair_across_the_customers_bills():
    raw = _raw([("c1", 1, "A"), ("c1", 1, "B"), ("c1", 2, "A"), ("c1", 2, "B"), ("c1", 3, "A"), ("c1", 3, "C"),
                ("c2", 4, "A"), ("c2", 5, "B")])            # c2 never bought two items together
    assert personal_pairs(raw) == {"c1": ("A", "B")}


def test_personal_pair_tie_is_deterministic_smallest_pair():
    raw = _raw([("c1", 1, "C"), ("c1", 1, "D"), ("c1", 2, "A"), ("c1", 2, "B")])
    assert personal_pairs(raw) == {"c1": ("A", "B")}


def test_without_product_names_there_are_no_pairs_and_no_crash():
    assert personal_pairs(pd.DataFrame({"customer_id": ["c1"], "transaction_id": [1]})) == {}
    assert personal_pairs(_raw([])) == {}


def _ids(n):
    return [f"c{i}" for i in range(n)]


def test_group_bundle_most_common_pair_and_its_coverage():
    pairs = {f"c{i}": ("A", "B") for i in range(6)} | {f"c{i}": ("C", "D") for i in range(6, 9)}
    assert group_bundle(_ids(10), pairs) == (["A", "B"], 0.6)    # c9 has no pair but counts in the 10


def test_group_bundle_nobody_has_a_pair():
    assert group_bundle(_ids(4), {}) == (None, 0.0)


def test_group_bundle_tie_is_deterministic():
    pairs = {"c0": ("A", "C"), "c1": ("A", "B")}
    assert group_bundle(["c0", "c1"], pairs) == (["A", "B"], 0.5)


def _frame(rows):
    return pd.DataFrame(rows, columns=["customer_id", "segment_name", "recommended_action",
                                       "action_key", "expected_profit", "promo_cost"])


def test_coverage_is_computed_per_segment_and_action_group_not_per_segment():
    frame = _frame([(f"a{i}", "Loyal", "VIP", "loyal_vip_reward", 1.0, 1.0) for i in range(4)]
                   + [(f"b{i}", "Loyal", "Nurture", "st_loyalty_nurture", 1.0, 1.0) for i in range(4)])
    pairs = {f"a{i}": ("A", "B") for i in range(4)} | {f"b{i}": ("C", "D") for i in range(4)}
    groups = {g["action_key"]: g["dotdash_offer"] for g in _build_dotdash_segments(frame, pairs)}
    assert groups["loyal_vip_reward"]["bundle_items"] == ["A", "B"]
    assert groups["loyal_vip_reward"]["bundle_coverage"] == 1.0       # a segment-level figure would be 0.5
    assert groups["st_loyalty_nurture"]["bundle_items"] == ["C", "D"]


def test_low_coverage_flag_follows_the_config_threshold(monkeypatch):
    from business_config import CONFIG
    frame = _frame([(f"a{i}", "Loyal", "VIP", "loyal_vip_reward", 1.0, 1.0) for i in range(10)])
    pairs = {f"a{i}": ("A", "B") for i in range(4)}                   # coverage 0.4
    offer = _build_dotdash_segments(frame, pairs)[0]["dotdash_offer"]
    assert offer["bundle_coverage"] == 0.4 and offer["bundle_low_coverage"] is True
    monkeypatch.setitem(CONFIG, "bundle_low_coverage_threshold", 0.3)
    assert _build_dotdash_segments(frame, pairs)[0]["dotdash_offer"]["bundle_low_coverage"] is False


def test_group_without_any_pair_is_reported_not_crashed():
    frame = _frame([("a0", "Loyal", "VIP", "loyal_vip_reward", 1.0, 1.0)])
    offer = _build_dotdash_segments(frame, {})[0]["dotdash_offer"]
    assert offer["bundle_items"] is None and offer["bundle_coverage"] == 0.0 and offer["bundle_low_coverage"] is True


def test_planner_run_adds_bundle_fields_to_every_group(tmp_path, monkeypatch):
    import campaign_planner as cp
    from fixtures_pos import make_pos_csv
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cp, "REGISTRY_PATH", str(tmp_path / "models" / "registry"))
    report = cp.run(budget=10_000_000.0, pos_file=make_pos_csv(tmp_path, n_customers=150, seed=4))
    for g in report["dotdash_segments"]:
        o = g["dotdash_offer"]
        assert 0.0 <= o["bundle_coverage"] <= 1.0 and isinstance(o["bundle_low_coverage"], bool)
        assert o["bundle_items"] is None or len(o["bundle_items"]) == 2
    assert any(g["dotdash_offer"]["bundle_items"] for g in report["dotdash_segments"])
