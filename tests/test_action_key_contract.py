"""action_key is a promise to DotDash/Dera: released keys never disappear, the doc table is the code, /segments gets key counts."""
import json
import os
import re

import pandas as pd

from action_rules import ACTION_KEYS
from crm_export import write_segments_latest

ROOT = os.path.join(os.path.dirname(__file__), "..")
SNAPSHOT = os.path.join(os.path.dirname(__file__), "golden", "action_keys_v1.json")
TABLE = os.path.join(ROOT, "docs", "dotdash-action-keys.md")


def test_released_keys_never_removed_or_renamed():
    """Keys may be ADDED (and action names changed); a key that shipped must keep existing."""
    released = json.load(open(SNAPSHOT, encoding="utf-8"))
    missing = [k for k in released if k not in ACTION_KEYS]
    assert missing == [], f"released action_key removed/renamed: {missing}"


def _table_rows():
    rows = []
    for line in open(TABLE, encoding="utf-8"):
        m = re.match(r"^\| (.+?) \| `([a-z0-9_]+)` \| (.+?) \|$", line.rstrip("\n"))
        if m:
            rows.append(m.groups())
    return rows


def test_doc_table_is_exactly_the_code():
    rows = _table_rows()
    keys = [k for _old, k, _new in rows]
    assert sorted(keys) == sorted(ACTION_KEYS) and len(keys) == len(set(keys))
    for _old, key, new in rows:
        assert new == ACTION_KEYS[key]["action"], key


def test_doc_table_lists_the_old_name_of_every_renamed_action():
    old_by_key = {k: old for old, k, _new in _table_rows()}
    for key in ("st_vip_reward_offpeak", "loyal_vip_reward", "champ_vip_reward",
                "mid_vip_reward_offpeak", "mid_vip_reward_peak"):
        assert old_by_key[key] == "VIP Reward: แจกแต้ม X2"


def _layer4(rows):
    return pd.DataFrame(rows, columns=["customer_id", "cluster", "recommended_action", "action_key",
                                       "expected_profit", "should_send", "promo_cost", "send_window"])


def _merge(tmp_path, monkeypatch, frame):
    import campaign_pipeline as cp
    monkeypatch.chdir(tmp_path)
    payload = {"target_segments": [{"cluster_id": 0, "segment_name": "Champions", "customer_count": len(frame),
                                    "unit_economics": {}}]}
    return cp.merge_layer4_into_payload(payload, frame, None)["target_segments"][0]


def test_merge_adds_key_counts_that_sum_to_the_action_counts(tmp_path, monkeypatch):
    frame = _layer4([
        ("a", 0, "VIP Reward: แจกแต้มพิเศษ 100 แต้ม", "st_vip_reward_offpeak", 5.0, 1, 2.0, "Peak Window"),
        ("b", 0, "VIP Reward: แจกแต้มพิเศษ 100 แต้ม", "st_vip_reward_offpeak", 4.0, 1, 2.0, "Peak Window"),
        ("c", 0, "Loyalty Nurture: แต้มสะสม", "st_loyalty_nurture", 1.0, 1, 1.0, "Peak Window"),
        ("d", 0, "Do Nothing (Lost Cause)", None, 0.0, 0, 0.0, "Peak Window"),
    ])
    seg = _merge(tmp_path, monkeypatch, frame)
    assert seg["l4_action_key_breakdown"] == {"st_vip_reward_offpeak": 2, "st_loyalty_nurture": 1, "none": 1}
    assert sum(seg["l4_action_key_breakdown"].values()) == sum(seg["l4_action_breakdown"].values()) == 4


def test_existing_action_breakdown_keeps_its_shape(tmp_path, monkeypatch):
    frame = _layer4([("a", 0, "Loyalty Nurture: แต้มสะสม", "st_loyalty_nurture", 1.0, 1, 1.0, "Peak Window")])
    seg = _merge(tmp_path, monkeypatch, frame)
    assert seg["l4_action_breakdown"] == {"Loyalty Nurture: แต้มสะสม": 1}      # name -> count, as before


def test_segments_latest_exports_action_key_breakdown_next_to_the_old_field(tmp_path):
    payload = {"gate_status": "PASSED", "target_segments": [
        {"cluster_id": 0, "segment_name": "Champions", "customer_count": 3,
         "l4_action_breakdown": {"A": 3}, "l4_action_key_breakdown": {"k": 3}, "unit_economics": {}},
        {"cluster_id": 1, "segment_name": "Other", "customer_count": 1, "unit_economics": {}}]}
    s0, s1 = json.load(open(write_segments_latest(payload, str(tmp_path)), encoding="utf-8"))["segments"]
    assert s0["action_breakdown"] == {"A": 3} and s0["action_key_breakdown"] == {"k": 3}
    assert s1["action_key_breakdown"] is None
