"""action_group (what kind of action), profit_basis (why a row is sent), schema_version (output format number)."""
import json
import os
import pickle

import pandas as pd
import pytest

import campaign_pipeline as cp
import uplift_engine as l4
from action_rules import ACTION_GROUPS, ACTION_GROUP_VALUES, ACTION_KEYS, infer_dotdash_offer
from business_config import OUTPUT_SCHEMA_VERSION
from fixtures_pos import make_pos_csv
from model_registry import ModelRegistry

META = {"approved_by": "t", "approved_at": "x", "provenance": "proxy-labels", "metrics": None, "created_at": "x"}
PROFIT_BASES = {"positive_incremental", "sure_thing_loyalty", "rescue_clv"}


def test_every_action_key_has_one_group_from_the_fixed_list():
    assert set(ACTION_GROUPS) == set(ACTION_KEYS)
    assert set(ACTION_GROUPS.values()) <= set(ACTION_GROUP_VALUES)
    assert ACTION_GROUP_VALUES == ("win_back", "retain", "loyalty_reward", "onboarding", "upsell", "other")
    assert all(ACTION_KEYS[k]["action_group"] == g for k, g in ACTION_GROUPS.items())


@pytest.mark.parametrize("key,group", [("st_cantlose_winback", "win_back"), ("st_newcust_welcome", "onboarding"),
                                       ("st_vip_reward_offpeak", "loyalty_reward"), ("atrisk_standard_peak", "retain"),
                                       ("mid_bundle_upsell_peak", "upsell"), ("low_milestone", "loyalty_reward")])
def test_group_examples_follow_the_action_meaning(key, group):
    assert ACTION_GROUPS[key] == group


def test_dotdash_offer_carries_the_group_next_to_the_key():
    assert infer_dotdash_offer("st_cantlose_winback")["action_group"] == "win_back"
    assert infer_dotdash_offer(None, "Do Nothing (Lost Cause)")["action_group"] is None


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = make_pos_csv(tmp_path, n_customers=120, seed=0)
    ctx = l4.build_serving_context(csv, None)
    model = l4._train_xlearner(ctx.raw_df[ctx.raw_df["was_treated"] == 1],
                               ctx.raw_df[ctx.raw_df["was_treated"] == 0], ctx.features)
    ModelRegistry(str(tmp_path / "models")).promote(pickle.dumps(model), None, META)
    return dict(csv=csv, models=str(tmp_path / "models"), ctx=ctx, model=model)


def test_engine_rows_have_group_and_a_profit_basis_that_matches_their_profit(env):
    _t, frame = l4.decide_for_customers(env["model"], env["ctx"])
    sent = frame[frame["should_send"] == 1]
    assert sent["action_group"].isin(ACTION_GROUP_VALUES).all()
    assert sent["profit_basis"].isin(PROFIT_BASES).all()
    assert (sent["expected_profit"] < 0).any(), "fixture must contain a sent row that loses money"
    losing = sent[sent["expected_profit"] < 0]
    assert losing["profit_basis"].isin({"sure_thing_loyalty", "rescue_clv"}).all()      # never 'positive_incremental'
    gaining = sent[sent["profit_basis"] == "positive_incremental"]
    assert (gaining["expected_profit"] > 0).all()
    nothing = frame[frame["action_key"].isna()]
    assert nothing["action_group"].isna().all() and nothing["profit_basis"].isna().all()


def test_final_csv_and_broadcast_list_carry_schema_version_and_the_new_columns_last(env):
    cp.run_layer4(env["csv"], None, models_root=env["models"])
    out = pd.read_csv("data/processed/final_campaign_engine_output.csv")
    assert list(out.columns)[-5:] == ["action_key", "cost_source", "action_group", "profit_basis", "schema_version"]
    assert (out["schema_version"].astype(str) == OUTPUT_SCHEMA_VERSION).all()


def test_operator_rows_are_grouped_other_and_have_no_profit_basis(env):
    from promo_advisor import PromoSpec
    cp.run_layer4(env["csv"], None, models_root=env["models"],
                  promo_spec=PromoSpec(name="Shop promo", discount_rate=0.02, promo_type="discount", force=True))
    out = pd.read_csv("data/processed/final_campaign_engine_output.csv")
    op = out[out["final_source"] == "operator"]
    assert len(op) > 0 and (op["action_group"] == "other").all() and op["profit_basis"].isna().all()


def test_planner_report_and_segments_json_name_their_schema_version(tmp_path, monkeypatch):
    import campaign_planner as pl
    from crm_export import write_segments_latest
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pl, "REGISTRY_PATH", str(tmp_path / "models" / "registry"))
    report = pl.run(budget=10_000_000.0, pos_file=make_pos_csv(tmp_path, n_customers=150, seed=4))
    assert report["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert all(g["dotdash_offer"]["action_group"] in ACTION_GROUP_VALUES for g in report["dotdash_segments"])
    path = write_segments_latest({"gate_status": "PASSED", "target_segments": []}, str(tmp_path))
    assert json.load(open(path, encoding="utf-8"))["schema_version"] == OUTPUT_SCHEMA_VERSION
