"""final_campaign_engine_output.csv / broadcast_list.csv: action_key + cost_source added, nothing else moves (Dera)."""
import json
import os
import pickle

import pandas as pd
import pytest

import campaign_pipeline as cp
import uplift_engine as l4
from action_rules import ACTION_KEYS
from fixtures_pos import make_pos_csv
from model_registry import ModelRegistry
from promo_advisor import PromoSpec

META = {"approved_by": "t", "approved_at": "x", "provenance": "proxy-labels", "metrics": None, "created_at": "x"}
BASELINE_COLS = json.load(open(os.path.join(os.path.dirname(__file__), "golden", "final_output_columns_v0.json")))
OUT = "data/processed/final_campaign_engine_output.csv"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = make_pos_csv(tmp_path, n_customers=120, seed=0)
    ctx = l4.build_serving_context(csv, None)
    model = l4._train_xlearner(ctx.raw_df[ctx.raw_df["was_treated"] == 1],
                               ctx.raw_df[ctx.raw_df["was_treated"] == 0], ctx.features)
    ModelRegistry(str(tmp_path / "models")).promote(pickle.dumps(model), None, META)
    return dict(tmp=tmp_path, csv=csv, models=str(tmp_path / "models"), ctx=ctx, model=model)


def _run(env, **kw):
    returned = cp.run_layer4(env["csv"], None, models_root=env["models"], **kw)
    return returned, pd.read_csv(OUT)


def test_old_columns_keep_their_names_and_order_and_new_ones_come_last(env):
    _ret, out = _run(env)
    assert list(out.columns) == BASELINE_COLS + ["action_key", "cost_source"]


def test_action_key_matches_the_action_written_on_the_same_row(env):
    """run_layer4 decides with its own context (Layer 2 segment names), so compare within the CSV itself."""
    _ret, out = _run(env)
    sent = out[out["should_send"] == 1]
    assert len(sent) > 0
    for _, row in sent.iterrows():
        prefix = ACTION_KEYS[row["action_key"]]["action"].split("{")[0]
        assert row["recommended_action"].startswith(prefix), (row["action_key"], row["recommended_action"])
    assert (sent["promo_cost"] > 0).any() and sent["promo_cost"].dtype == float      # promo_cost is still baht per customer


def test_every_sent_row_has_a_known_key_and_an_estimated_cost(env):
    _ret, out = _run(env)
    sent = out[out["should_send"] == 1]
    assert len(sent) > 0
    assert sent["action_key"].isin(list(ACTION_KEYS)).all()
    assert (sent["cost_source"] == "estimated").all()


def test_rows_with_no_promotion_have_no_key_and_cost_source_none(env):
    _ret, out = _run(env)
    nothing = out[out["recommended_action"].fillna("").str.startswith(("Do Nothing", "Zero-Cost"))]
    assert len(nothing) > 0
    assert nothing["action_key"].isna().all() and (nothing["cost_source"] == "none").all()


def test_operator_promo_replaces_the_key_and_the_cost_source(env):
    spec = PromoSpec(name="Shop promo", discount_rate=0.02, promo_type="discount", force=True)
    _ret, out = _run(env, promo_spec=spec)
    assert {"ml", "operator", "none"} <= set(out["final_source"])            # fixture exercises all three
    op = out[out["final_source"] == "operator"]
    assert (op["action_key"] == "operator_promo").all() and (op["cost_source"] == "operator").all()
    ml = out[(out["final_source"] == "ml") & (out["should_send"] == 1)]
    assert ml["action_key"].isin(list(ACTION_KEYS)).all() and (ml["cost_source"] == "estimated").all()
    none = out[out["final_source"] == "none"]
    assert none["action_key"].isna().all() and (none["cost_source"] == "none").all()
    assert list(out.columns)[-2:] == ["action_key", "cost_source"]


def test_broadcast_list_carries_the_new_columns_after_the_old_ones(env):
    cp.run_layer4(env["csv"], None, models_root=env["models"])
    layer4 = pd.read_csv(OUT)
    payload = {"target_segments": [{"cluster_id": c, "segment_name": f"S{c}", "customer_count": 1,
                                    "unit_economics": {}} for c in sorted(layer4["kmeans_cluster"].unique())]}
    layer4 = layer4.rename(columns={"kmeans_cluster": "cluster"})
    cp.merge_layer4_into_payload(payload, layer4, None)
    b = pd.read_csv("data/processed/broadcast_list.csv")
    assert list(b.columns)[-2:] == ["action_key", "cost_source"]
    assert list(b.columns)[:11] == ["customer_id", "segment_name", "recommended_action", "offer", "expected_profit",
                                    "promo_cost", "send_date", "send_window", "send_time_hint", "channel", "should_send"]
    assert b["action_key"].isin(list(ACTION_KEYS) + ["operator_promo"]).all()


def test_layer4_view_hands_the_new_columns_to_the_payload_merge():
    """merge_layer4_into_payload builds broadcast_list/segments from this view — it must carry them."""
    detail = pd.DataFrame({"customer_id": ["a"], "cluster": [0], "recommended_action": ["x"], "expected_profit": [1.0],
                           "guardrail_blocked": [0], "should_send": [1], "send_window": ["Peak Window"],
                           "treatment_type": ["none"], "action_key": ["st_loyalty_nurture"], "cost_source": ["estimated"],
                           "unrelated": [9]})
    view = cp._layer4_view(detail)
    assert list(view.columns) == ["customer_id", "cluster", "recommended_action", "expected_profit", "guardrail_blocked",
                                  "should_send", "send_window", "treatment_type", "action_key", "cost_source"]


def test_both_run_layer4_and_the_cli_main_build_the_view_with_the_same_helper():
    """The __main__ path is not exercised by unit tests, so pin that it cannot drift from run_layer4 again."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "campaign_pipeline.py"), encoding="utf-8").read()
    assert "detail[keep_cols]" not in src.replace("return detail[keep_cols]", "", 1)   # only the helper builds it
    assert src.count("return _layer4_view(detail)") + src.count("= _layer4_view(detail)") == 2
