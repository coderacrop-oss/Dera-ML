"""tests/crm_api/test_send_date_endpoint.py — Tests for GET /v1/send-date/recommend"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import HDR, KEY, CAL_TOKEN
from crm_api.app import create_app
from crm_api.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pos_csv(data_dir: Path, store_id: str, n_customers: int = 60) -> Path:
    """Write a synthetic POS CSV to the store's pos path."""
    store_dir = data_dir / "stores" / store_id
    store_dir.mkdir(parents=True, exist_ok=True)
    pos_path = store_dir / "latest_pos.csv"

    rows = []
    base = datetime(2026, 1, 1)
    for cid in range(1, n_customers + 1):
        for j in range(5):
            txn_date = base + timedelta(days=cid * 3 + j * 20)
            rows.append({
                "customer_id":      str(cid),
                "transaction_date": txn_date.strftime("%Y-%m-%d"),
                "amount":           round(100.0 + cid * 1.5 + j * 10, 2),
            })
    pd.DataFrame(rows).to_csv(pos_path, index=False)
    return pos_path


def _make_client(data_dir: Path, store_id: str, with_pos: bool = True) -> TestClient:
    if with_pos:
        _write_pos_csv(data_dir, store_id)
    return TestClient(create_app(Settings(data_dir=data_dir, api_key=KEY, calendar_token=CAL_TOKEN)))


STORE_ID = "test_store_sd"
BASE_URL = "/v1/send-date/recommend"
BASE_PARAMS = {
    "store_id": STORE_ID,
    "segment":  "At-Risk",
    "month":    "2026-11",
}


@pytest.fixture
def sd_client(tmp_path):
    return _make_client(tmp_path, STORE_ID)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_requires_api_key(self, tmp_path):
        client = _make_client(tmp_path, STORE_ID)
        r = client.get(BASE_URL, params=BASE_PARAMS)
        assert r.status_code == 401

    def test_wrong_key_rejected(self, tmp_path):
        client = _make_client(tmp_path, STORE_ID)
        r = client.get(BASE_URL, params=BASE_PARAMS, headers={"X-API-Key": "bad"})
        assert r.status_code == 401

    def test_correct_key_accepted(self, sd_client):
        r = sd_client.get(BASE_URL, params=BASE_PARAMS, headers=HDR)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 404 — store not found
# ---------------------------------------------------------------------------

class TestStoreNotFound:
    def test_missing_pos_returns_404(self, tmp_path):
        # No POS file written
        client = _make_client(tmp_path, STORE_ID, with_pos=False)
        r = client.get(BASE_URL, params=BASE_PARAMS, headers=HDR)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "store_not_found"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_bad_month_format_returns_400(self, sd_client):
        params = {**BASE_PARAMS, "month": "11-2026"}
        r = sd_client.get(BASE_URL, params=params, headers=HDR)
        assert r.status_code == 400

    def test_invalid_month_number_returns_400(self, sd_client):
        params = {**BASE_PARAMS, "month": "2026-13"}
        r = sd_client.get(BASE_URL, params=params, headers=HDR)
        assert r.status_code == 400

    def test_invalid_goal_returns_400(self, sd_client):
        params = {**BASE_PARAMS, "goal": "destroy_everything"}
        r = sd_client.get(BASE_URL, params=params, headers=HDR)
        assert r.status_code == 400

    def test_negative_cooldown_returns_400(self, sd_client):
        params = {**BASE_PARAMS, "cooldown_days": -1}
        r = sd_client.get(BASE_URL, params=params, headers=HDR)
        assert r.status_code == 400

    def test_top_n_zero_returns_400(self, sd_client):
        params = {**BASE_PARAMS, "top_n": 0}
        r = sd_client.get(BASE_URL, params=params, headers=HDR)
        assert r.status_code == 400

    def test_top_n_11_returns_400(self, sd_client):
        params = {**BASE_PARAMS, "top_n": 11}
        r = sd_client.get(BASE_URL, params=params, headers=HDR)
        assert r.status_code == 400

    def test_missing_store_id_returns_422(self, sd_client):
        r = sd_client.get(BASE_URL, params={"segment": "At-Risk", "month": "2026-11"}, headers=HDR)
        assert r.status_code == 422

    def test_missing_segment_returns_422(self, sd_client):
        r = sd_client.get(BASE_URL, params={"store_id": STORE_ID, "month": "2026-11"}, headers=HDR)
        assert r.status_code == 422

    def test_missing_month_returns_422(self, sd_client):
        r = sd_client.get(BASE_URL, params={"store_id": STORE_ID, "segment": "At-Risk"}, headers=HDR)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Successful response
# ---------------------------------------------------------------------------

class TestSuccessResponse:
    def _get(self, sd_client, **extra_params):
        params = {**BASE_PARAMS, **extra_params}
        return sd_client.get(BASE_URL, params=params, headers=HDR)

    def test_200_status(self, sd_client):
        assert self._get(sd_client).status_code == 200

    def test_response_has_required_keys(self, sd_client):
        data = self._get(sd_client).json()
        for key in ("store_id", "segment", "month", "goal",
                    "recommended_date", "score", "reason",
                    "payday_window", "overlap_warning",
                    "alternatives", "payday_dates"):
            assert key in data, f"missing key: {key}"

    def test_store_id_echoed(self, sd_client):
        data = self._get(sd_client).json()
        assert data["store_id"] == STORE_ID

    def test_segment_echoed(self, sd_client):
        data = self._get(sd_client).json()
        assert data["segment"] == "At-Risk"

    def test_month_echoed(self, sd_client):
        data = self._get(sd_client).json()
        assert data["month"] == "2026-11"

    def test_recommended_date_in_month(self, sd_client):
        data = self._get(sd_client).json()
        d = data["recommended_date"]
        assert d.startswith("2026-11-"), f"expected November date, got {d}"

    def test_score_is_non_negative(self, sd_client):
        data = self._get(sd_client).json()
        assert data["score"] >= 0.0

    def test_payday_window_is_bool(self, sd_client):
        data = self._get(sd_client).json()
        assert isinstance(data["payday_window"], bool)

    def test_alternatives_is_list(self, sd_client):
        data = self._get(sd_client).json()
        assert isinstance(data["alternatives"], list)

    def test_alternatives_default_top_n(self, sd_client):
        # top_n default = 3 → best + 2 alts
        data = self._get(sd_client).json()
        assert len(data["alternatives"]) == 2

    def test_alternatives_have_keys(self, sd_client):
        data = self._get(sd_client).json()
        for alt in data["alternatives"]:
            assert "date" in alt
            assert "score" in alt
            assert "reason" in alt

    def test_payday_dates_is_list(self, sd_client):
        data = self._get(sd_client).json()
        assert isinstance(data["payday_dates"], list)
        assert len(data["payday_dates"]) > 0

    def test_payday_dates_in_november(self, sd_client):
        data = self._get(sd_client).json()
        for d in data["payday_dates"]:
            assert d.startswith("2026-11-")

    def test_overlap_warning_none_no_history(self, sd_client):
        # No campaign history → no overlap warning
        data = self._get(sd_client).json()
        assert data["overlap_warning"] is None

    def test_reason_is_string(self, sd_client):
        data = self._get(sd_client).json()
        assert isinstance(data["reason"], str)
        assert len(data["reason"]) > 0

    def test_goal_echoed(self, sd_client):
        data = self._get(sd_client, goal="win_back").json()
        assert data["goal"] == "win_back"

    def test_all_valid_goals_accepted(self, sd_client):
        for goal in ("maximize_profit", "win_back", "retain_loyal", "maximize_reach"):
            r = self._get(sd_client, goal=goal)
            assert r.status_code == 200, f"goal={goal} failed: {r.json()}"

    def test_top_n_one_returns_no_alternatives(self, sd_client):
        data = self._get(sd_client, top_n=1).json()
        assert len(data["alternatives"]) == 0

    def test_top_n_five_returns_four_alternatives(self, sd_client):
        data = self._get(sd_client, top_n=5).json()
        assert len(data["alternatives"]) == 4

    def test_different_months_work(self, sd_client):
        for month in ("2026-11", "2026-12", "2027-01"):
            r = self._get(sd_client, month=month)
            assert r.status_code == 200
            assert r.json()["month"] == month

    def test_all_ibm_segments_accepted(self, sd_client):
        segments = [
            "Champions", "Loyal Customers", "Potential Loyalist", "Promising",
            "Need Attention", "About to Sleep", "At-Risk", "Can't Lose Them",
            "Hibernating", "Lost",
        ]
        for seg in segments:
            r = sd_client.get(BASE_URL, params={
                "store_id": STORE_ID, "segment": seg, "month": "2026-11"
            }, headers=HDR)
            assert r.status_code == 200, f"segment={seg!r} failed: {r.text}"

    def test_cooldown_zero_accepted(self, sd_client):
        r = self._get(sd_client, cooldown_days=0)
        assert r.status_code == 200

    def test_cooldown_365_accepted(self, sd_client):
        r = self._get(sd_client, cooldown_days=365)
        assert r.status_code == 200
