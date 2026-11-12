"""tests/crm_api/test_record_outcome_summary.py — POST /v1/campaign/record-outcome carries plan_result (summary, next_actions, model_accuracy, top1)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from conftest import HDR, KEY, CAL_TOKEN
from crm_api import store as st
from crm_api.app import create_app
from crm_api.settings import Settings

URL = "/v1/campaign/record-outcome"


@pytest.fixture
def env(tmp_path):
    client = TestClient(create_app(Settings(data_dir=tmp_path, api_key=KEY, calendar_token=CAL_TOKEN)))
    return client, str(tmp_path)


def _plan(data_dir, plan_id="plan_1", expected=1000.0, executed_days_ago=None, top1=None):
    rec = {"plan_id": plan_id, "store_id": "s1", "expected_profit": expected}
    if executed_days_ago is not None:
        rec["executed_at"] = (datetime.now(timezone.utc) - timedelta(days=executed_days_ago)).isoformat()
    if top1:
        rec["top1_counterfactual"] = top1
    st.append_plan(data_dir, rec)


def _post(client, **o):
    body = {"plan_id": "plan_1", "actual_revenue": 900.0, "actual_customers_responded": 40, "measurement_period_days": 30, **o}
    return client.post(URL, json=body, headers=HDR)


def test_a_close_result_keeps_going_and_the_old_fields_are_unchanged(env):
    client, d = env; _plan(d)
    r = _post(client); assert r.status_code == 200
    j = r.json(); assert j["outcome_recorded"] is True and j["calibration_ratio"] == 0.9 and "calibration_stats" in j
    pr = j["plan_result"]
    assert [a["code"] for a in pr["next_actions"]] == ["KEEP_GOING"] and pr["model_accuracy"]["error_pct"] == -10.0 and "คาดกำไร 1,000 บาท" in pr["summary_th"]
    assert pr["top1_counterfactual"] is None


def test_a_big_miss_asks_for_a_retrain(env):
    client, d = env; _plan(d)
    pr = _post(client, actual_revenue=200.0).json()["plan_result"]
    assert pr["model_accuracy"]["retrain_recommended"] is True and "REQUEST_RETRAIN" in [a["code"] for a in pr["next_actions"]]


def test_a_plan_still_inside_its_window_says_to_wait(env):
    client, d = env; _plan(d, executed_days_ago=5)
    assert "WAIT_FOR_WINDOW" in [a["code"] for a in _post(client).json()["plan_result"]["next_actions"]]
    _plan(d, plan_id="plan_2", executed_days_ago=40)
    assert "WAIT_FOR_WINDOW" not in [a["code"] for a in _post(client, plan_id="plan_2").json()["plan_result"]["next_actions"]]


def test_the_plans_own_top1_figure_is_passed_on(env):
    client, d = env; t = {"segment": "S", "action": "A", "expected_profit": 600.0}
    _plan(d, top1=t)
    assert _post(client).json()["plan_result"]["top1_counterfactual"] == t


def test_an_unknown_plan_is_still_recorded_and_says_it_cannot_judge(env):
    client, _ = env
    pr = _post(client, plan_id="nope").json()["plan_result"]
    assert pr["model_accuracy"]["error_pct"] is None and "ยังประมาณผลจริงไม่ได้" in pr["summary_th"] or pr["model_accuracy"]["predicted_profit"] == 0.0
