"""Characterization + unit tests for the Layer 4 split (context / decisions / gate)."""
import os

import pandas as pd
import pytest

import uplift_engine as l4
from fixtures_pos import make_pos_csv

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "layer4_basic.csv")
COLS = ["customer_id", "recommended_action", "expected_profit", "should_send"]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg = tmp_path / "models" / "registry"
    monkeypatch.setattr(l4, "REGISTRY_PATH", str(reg))
    return tmp_path


def _sorted(df):
    return df[COLS].sort_values("customer_id").reset_index(drop=True)


def _trained(ctx):
    treat = ctx.raw_df[ctx.raw_df["was_treated"] == 1]
    ctrl = ctx.raw_df[ctx.raw_df["was_treated"] == 0]
    return l4._train_xlearner(treat, ctrl, ctx.features)


def test_decide_for_customers_matches_golden(workdir):
    csv = make_pos_csv(workdir, n_customers=120, seed=0)
    ctx = l4.build_serving_context(csv, None)
    profit, detail = l4.decide_for_customers(_trained(ctx), ctx)
    golden = pd.read_csv(GOLDEN, dtype={"customer_id": str})
    pd.testing.assert_frame_equal(_sorted(detail), golden, check_dtype=False, rtol=1e-6)
    assert profit == pytest.approx(detail["expected_profit"].sum())


def test_decide_for_customers_filters_ids(workdir):
    ctx = l4.build_serving_context(make_pos_csv(workdir), None)
    _, detail = l4.decide_for_customers(_trained(ctx), ctx, {"c001"})
    assert list(detail["customer_id"]) == ["c001"]


def test_decide_for_customers_with_no_matching_ids_is_empty_not_a_crash(workdir):
    ctx = l4.build_serving_context(make_pos_csv(workdir), None)
    profit, detail = l4.decide_for_customers(_trained(ctx), ctx, {"nobody"})
    assert profit == 0.0 and len(detail) == 0 and "should_send" in detail.columns


def test_trained_model_has_feature_names(workdir):
    ctx = l4.build_serving_context(make_pos_csv(workdir), None)
    assert _trained(ctx).feature_names_ == ctx.features


def test_context_label_mode(workdir):
    assert l4.build_serving_context(make_pos_csv(workdir), None).label_mode == "proxy"
    csv = make_pos_csv(workdir, with_labels=True, name="lab.csv")
    assert l4.build_serving_context(csv, csv).label_mode == "real"


G = l4.gate_decision


def test_gate_passes_with_growth_above_target():
    r = G(105.0, 100.0, 10.0, False, 50.0)
    assert r.passed and r.growth_pct == pytest.approx(5.0)


def test_gate_fails_below_target():
    r = G(101.0, 100.0, 10.0, False, 50.0)
    assert not r.passed and "ยังไม่ถึงเกณฑ์" in r.reason


def test_gate_fails_when_worse():
    assert not G(98.0, 100.0, 10.0, False, 50.0).passed


def test_gate_fails_when_virtually_identical():
    assert not G(100.2, 100.0, 10.0, False, 50.0).passed


def test_gate_passes_in_recession_with_defensive_model():
    assert G(90.0, 100.0, 60.0, True, 50.0).passed


def test_gate_first_champion_passes():
    r = G(10.0, None, 0.0, False, 50.0)
    assert r.passed and r.growth_pct is None and "Champion ตัวแรก" in r.reason


def test_gate_zero_champion_profit_does_not_crash_and_fails_safe():
    r = G(10.0, 0.0, 0.0, False, 50.0)
    assert r.growth_pct is None and not r.passed


def test_gate_negative_champion_profit_does_not_crash():
    r = G(-10.0, -50.0, 0.0, False, 50.0)
    assert r.growth_pct is None and not r.passed


def test_market_is_recession_flags_a_price_collapse():
    ts = pd.date_range("2026-01-01", periods=100, freq="D")
    price = [100.0] * 75 + [10.0] * 25
    rec, current = l4.market_is_recession(pd.DataFrame({"timestamp": ts, "total_price": price}))
    assert rec is True and current == pytest.approx(10.0)


def test_decisions_record_the_route_and_uplift_group_each_customer_went_through(workdir):
    ctx = l4.build_serving_context(make_pos_csv(workdir, n_customers=120, seed=0), None)
    _, detail = l4.decide_for_customers(_trained(ctx), ctx)
    assert set(detail["basket_route"]) <= {"out_of_stock", "low", "normal"}
    groups = {"persuadable", "sure_thing", "do_not_disturb", "lost_cause"}
    # only customers who got to the uplift step have a group; stock and low-basket stops have none
    assert set(detail.loc[detail["basket_route"] != "normal", "uplift_group"].dropna()) == set()
    assert set(detail.loc[detail["basket_route"] == "normal", "uplift_group"]) <= groups
    # the route says exactly what the decision did
    assert (detail.loc[detail["basket_route"] == "out_of_stock", "recommended_action"] == "Do Nothing (Out of Stock)").all()
    assert (detail.loc[detail["uplift_group"] == "lost_cause", "recommended_action"] == "Do Nothing (Lost Cause)").all()

