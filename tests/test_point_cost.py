"""Optional per-shop point value: cost of points offers = points x point_value_thb x redemption_rate (DotDash item 4)."""
import pandas as pd
import pytest
from pydantic import ValidationError

import uplift_engine as l4
from fixtures_pos import make_pos_csv


def _vip_decision(point_cost_per_point=None):
    """A sure-thing Champion off-peak -> rule st_vip_reward_offpeak (100 points, rate vip_points = 2%)."""
    return l4._select_action(group="sure_thing", avg_spent=120.0, cogs=0.4, uplift=0.05, prob_control=0.6,
                             rfm_seg=10, is_spike=0.0, time_context="Off-Peak",
                             cluster_segment_name="Champions", point_cost_per_point=point_cost_per_point)


def test_without_override_the_cost_is_the_two_percent_estimate():
    action, _profit, cost, _window, key = _vip_decision()
    assert key == "st_vip_reward_offpeak" and cost == pytest.approx(120.0 * 0.02, abs=0.01)


def test_override_prices_points_from_the_shop_value():
    _a, _p, cost, _w, key = _vip_decision(point_cost_per_point=0.5 * 0.4)    # 0.5 THB/pt, 40% redeemed
    assert key == "st_vip_reward_offpeak" and cost == pytest.approx(100 * 0.5 * 0.4)


def test_a_point_value_that_makes_the_offer_unaffordable_removes_it():
    action, _p, cost, _w, key = _vip_decision(point_cost_per_point=1.0)      # 100 THB cost > baseline profit
    assert action == "" and key is None and cost == 0.0


def test_override_does_not_touch_coupon_offers():
    kw = dict(group="persuadable", avg_spent=120.0, cogs=0.4, uplift=0.10, prob_control=0.2, rfm_seg=5,
              is_spike=0.0, time_context="Off-Peak", cluster_segment_name="Promising")
    assert l4._select_action(**kw)[2:] == l4._select_action(**kw, point_cost_per_point=0.3)[2:]


@pytest.fixture
def ctx_and_model(tmp_path):
    ctx = l4.build_serving_context(make_pos_csv(tmp_path, n_customers=200, seed=3), None)
    model = l4._train_xlearner(ctx.raw_df[ctx.raw_df["was_treated"] == 1],
                               ctx.raw_df[ctx.raw_df["was_treated"] == 0], ctx.features)
    return ctx, model


def test_no_override_on_the_context_changes_nothing(ctx_and_model):
    ctx, model = ctx_and_model
    _t, base = l4.decide_for_customers(model, ctx)
    ctx.point_cost_per_point = None
    _t, again = l4.decide_for_customers(model, ctx)
    pd.testing.assert_frame_equal(base, again)


def test_override_on_the_context_reprices_only_points_rows(ctx_and_model):
    ctx, model = ctx_and_model
    _t, base = l4.decide_for_customers(model, ctx)
    ctx.point_cost_per_point = 0.01
    _t, cheap = l4.decide_for_customers(model, ctx)
    pts = base["action_key"].isin(["st_vip_reward_offpeak", "st_loyalty_nurture", "st_vip_early_access_peak"])
    assert pts.any()
    both = base.merge(cheap, on="customer_id", suffixes=("_b", "_c"))
    same_key = both[both["action_key_b"] == both["action_key_c"]]
    pts_rows = same_key[same_key["action_key_b"].isin(["st_vip_reward_offpeak", "st_loyalty_nurture"])]
    assert len(pts_rows) > 0 and (pts_rows["promo_cost_c"] != pts_rows["promo_cost_b"]).any()


@pytest.mark.parametrize("kw", [
    {"point_value_thb": 0}, {"point_value_thb": -1.0},
    {"point_value_thb": 0.5, "redemption_rate": 0}, {"point_value_thb": 0.5, "redemption_rate": 1.5},
    {"redemption_rate": 0.5},                                   # a rate without a value is meaningless
])
def test_request_model_rejects_bad_values(kw):
    from crm_api.campaign import PlanRequest
    with pytest.raises(ValidationError):
        PlanRequest(store_id="s", pos_data="x", budget=100.0, **kw)


def test_planner_rejects_bad_values_too(tmp_path):
    import campaign_planner as cp
    with pytest.raises(ValueError):
        cp.run(budget=100.0, pos_file="unused.csv", point_value_thb=0.0)
    with pytest.raises(ValueError):
        cp.run(budget=100.0, pos_file="unused.csv", point_value_thb=1.0, redemption_rate=1.5)
    with pytest.raises(ValueError):
        cp.run(budget=100.0, pos_file="unused.csv", redemption_rate=0.5)


def _run(tmp_path, monkeypatch, **kw):
    import campaign_planner as cp
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cp, "REGISTRY_PATH", str(tmp_path / "models" / "registry"))
    return cp.run(budget=10_000_000.0, pos_file=make_pos_csv(tmp_path, n_customers=150, seed=4), **kw)


def test_cost_source_is_estimated_everywhere_without_an_override(tmp_path, monkeypatch):
    groups = _run(tmp_path, monkeypatch)["dotdash_segments"]
    assert groups and {g["cost_source"] for g in groups} == {"estimated"}


def test_cost_source_marks_only_points_groups_as_shop_provided(tmp_path, monkeypatch):
    groups = _run(tmp_path, monkeypatch, point_value_thb=0.01, redemption_rate=0.5)["dotdash_segments"]
    for g in groups:
        expected = "shop_provided" if g["dotdash_offer"]["offer_type"] == "points" else "estimated"
        assert g["cost_source"] == expected
    assert any(g["cost_source"] == "shop_provided" for g in groups)
