"""Customer-facing action names/messages must say what the offer really is (DotDash item 3)."""
import re

import campaign_scheduler as sch
from action_rules import ACTION_KEYS, get_dotdash_offer

MULTIPLIER = re.compile(r"(?i)[x×]\s?\d|\d\s?[x×]")
# Known and deliberately left for a follow-up: renaming needs its own approved wording.
PENDING_MULTIPLIER_NAMES = {"VIP Win-Back: {bundle} + แต้มพิเศษ X3"}
# Pre-existing template key that no rule produces; reported, not removed here.
STALE_TEMPLATE_KEYS = {"Push Promotion: เซตคู่คุ้มค่า (Bundle)"}

NEW_100 = "VIP Reward: แจกแต้มพิเศษ 100 แต้ม"
NEW_80 = "VIP Reward: แจกแต้มพิเศษ 80 แต้ม"


def test_no_action_name_claims_a_points_multiplier():
    bad = [r["action"] for r in ACTION_KEYS.values()
           if MULTIPLIER.search(r["action"]) and r["action"] not in PENDING_MULTIPLIER_NAMES]
    assert bad == []


def test_no_message_template_claims_a_points_multiplier():
    bad = [k for k, v in sch.MESSAGE_TEMPLATES.items() if MULTIPLIER.search(k) or MULTIPLIER.search(v)]
    assert bad == []


def test_points_stated_in_an_action_name_equal_the_offer():
    for key, rule in ACTION_KEYS.items():
        m = re.search(r"(\d+) แต้ม", rule["action"])
        if m and rule["action"].startswith("VIP Reward"):
            assert get_dotdash_offer(rule)["points_amount"] == int(m.group(1)), key


def test_vip_reward_rules_use_the_new_names():
    names = {k: r["action"] for k, r in ACTION_KEYS.items() if r["action"].startswith("VIP Reward")}
    assert names == {
        "st_vip_reward_offpeak": NEW_100, "loyal_vip_reward": NEW_100, "champ_vip_reward": NEW_100,
        "mid_vip_reward_offpeak": NEW_100, "mid_vip_reward_peak": NEW_80,
    }


def test_scheduler_knows_the_renamed_actions():
    for name in (NEW_100, NEW_80):
        assert sch.WAVE_MAP[name] == 3
        assert name in sch.MESSAGE_TEMPLATES


def test_scheduler_keys_are_real_rule_actions():
    actions = {r["action"] for r in ACTION_KEYS.values()}
    assert [k for k in sch.WAVE_MAP if k not in actions] == []
    assert [k for k in sch.MESSAGE_TEMPLATES
            if k != "_default" and k not in actions and k not in STALE_TEMPLATE_KEYS] == []
