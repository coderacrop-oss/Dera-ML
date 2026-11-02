"""Compare API output with the REAL files the pipeline wrote (skipped if none exist)."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from crm_api.app import create_app
from crm_api.loaders import list_months
from crm_api.settings import Settings

REAL = Path(__file__).resolve().parents[2] / "data" / "processed"
HDR = {"X-API-Key": "k"}
MONTHS = list_months(REAL) if REAL.exists() else []
pytestmark = pytest.mark.skipif(not MONTHS, reason="no pipeline output (run campaign_scheduler.py first)")


@pytest.fixture(scope="module")
def real():
    month = MONTHS[-1]
    client = TestClient(create_app(Settings(data_dir=REAL, api_key="k")))
    sched = pd.read_csv(REAL / f"campaign_schedule_{month}.csv")
    summary = json.loads((REAL / f"campaign_summary_{month}.json").read_text(encoding="utf-8"))
    return month, client, sched, summary


def test_totals_and_waves_equal_source(real):
    month, client, sched, summary = real
    body = client.get(f"/v1/plans/{month}", headers=HDR).json()
    assert body["totals"]["scheduled"] == summary["total_scheduled"] == len(sched)
    assert body["totals"]["holdout"] == summary["total_holdout"]
    assert body["totals"]["expected_profit"] == pytest.approx(summary["total_expected_profit"], abs=0.05)
    wb = summary["wave_breakdown"]
    got = {w["wave"]: w["count"] for w in body["waves"]}
    assert got == {1: wb["wave_1_off_peak_traffic"], 2: wb["wave_2_peak_upsell"], 3: wb["wave_3_vip_early"]}


def test_events_cover_every_scheduled_customer_once(real):
    month, client, sched, _ = real
    ev = client.get(f"/v1/plans/{month}/events", headers=HDR).json()["events"]
    assert sum(e["customer_count"] for e in ev) == len(sched)
    daily = {}
    for e in ev:
        daily[e["start"][:10]] = daily.get(e["start"][:10], 0) + e["customer_count"]
    assert daily == {str(k): v for k, v in sched.assign(d=sched.send_datetime.str[:10]).groupby("d").size().items()}


def test_sampled_customers_match_csv_rows(real):
    month, client, sched, _ = real
    for _, row in sched.sample(min(10, len(sched)), random_state=0).iterrows():
        b = client.get(f"/v1/plans/{month}/customers/{row.customer_id}", headers=HDR).json()
        assert b["wave"] == int(row.wave) and b["recommended_action"] == row.recommended_action
        assert b["send_datetime"][:16].replace("T", " ") == str(row.send_datetime)[:16]


def test_ics_event_count_equals_events(real):
    month, client, _, _ = real
    n = len(client.get(f"/v1/plans/{month}/events", headers=HDR).json()["events"])
    assert client.get(f"/v1/plans/{month}/calendar.ics", headers=HDR).text.count("BEGIN:VEVENT") == n
