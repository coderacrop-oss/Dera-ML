"""tests/crm_api/test_retrain_request.py — Tests for POST /v1/model/retrain-request"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from conftest import HDR, KEY, CAL_TOKEN
from crm_api.app import create_app
from crm_api.settings import Settings

URL = "/v1/model/retrain-request"

VALID_BODY = {
    "store_id": "shop_001",
    "reason":   "manual_request",
    "psm_mode": "auto",
}


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(Settings(data_dir=tmp_path, api_key=KEY, calendar_token=CAL_TOKEN)))


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_requires_api_key(self, client):
        r = client.post(URL, json=VALID_BODY)
        assert r.status_code == 401

    def test_wrong_key_rejected(self, client):
        r = client.post(URL, json=VALID_BODY, headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_correct_key_accepted(self, client):
        r = client.post(URL, json=VALID_BODY, headers=HDR)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_reason_returns_400(self, client):
        body = {**VALID_BODY, "reason": "hack_the_model"}
        r = client.post(URL, json=body, headers=HDR)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_input"

    def test_invalid_psm_mode_returns_400(self, client):
        body = {**VALID_BODY, "psm_mode": "magic"}
        r = client.post(URL, json=body, headers=HDR)
        assert r.status_code == 400

    def test_missing_store_id_returns_422(self, client):
        r = client.post(URL, json={"reason": "manual_request"}, headers=HDR)
        assert r.status_code == 422

    def test_all_valid_reasons_accepted(self, client):
        for reason in ("manual_request", "accuracy_concern", "new_data"):
            body = {**VALID_BODY, "reason": reason}
            r = client.post(URL, json=body, headers=HDR)
            assert r.status_code == 200, f"reason={reason!r} failed: {r.json()}"

    def test_all_valid_psm_modes_accepted(self, client):
        for mode in ("auto", "discount", "campaign_list", "proxy"):
            body = {**VALID_BODY, "psm_mode": mode}
            r = client.post(URL, json=body, headers=HDR)
            assert r.status_code == 200, f"psm_mode={mode!r} failed: {r.json()}"


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------

class TestResponse:
    def _post(self, client, **overrides):
        return client.post(URL, json={**VALID_BODY, **overrides}, headers=HDR)

    def test_returns_request_id(self, client):
        data = self._post(client).json()
        assert "request_id" in data
        assert data["request_id"].startswith("retrain_req_")

    def test_request_id_unique(self, client):
        id1 = self._post(client).json()["request_id"]
        id2 = self._post(client).json()["request_id"]
        assert id1 != id2

    def test_store_id_echoed(self, client):
        data = self._post(client).json()
        assert data["store_id"] == "shop_001"

    def test_queued_at_present(self, client):
        data = self._post(client).json()
        assert "queued_at" in data
        from datetime import datetime
        datetime.fromisoformat(data["queued_at"].replace("Z", "+00:00"))

    def test_status_queued(self, client):
        data = self._post(client).json()
        assert data["status"] == "queued"

    def test_reason_echoed(self, client):
        data = self._post(client, reason="accuracy_concern").json()
        assert data["reason"] == "accuracy_concern"

    def test_psm_mode_echoed(self, client):
        data = self._post(client, psm_mode="discount").json()
        assert data["psm_mode"] == "discount"

    def test_estimated_completion_is_thai(self, client):
        data = self._post(client).json()
        assert "estimated_completion" in data
        assert any(ord(c) > 127 for c in data["estimated_completion"])

    def test_defaults_applied(self, client):
        # Send only required field
        r = client.post(URL, json={"store_id": "shop_x"}, headers=HDR)
        assert r.status_code == 200
        data = r.json()
        assert data["reason"]   == "manual_request"
        assert data["psm_mode"] == "auto"

    def test_requested_by_optional(self, client):
        body = {**VALID_BODY, "requested_by": "user@dotdash.app"}
        r = client.post(URL, json=body, headers=HDR)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Persistence — requests saved to JSONL
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_request_saved_to_file(self, tmp_path):
        client = TestClient(create_app(Settings(data_dir=tmp_path, api_key=KEY, calendar_token=CAL_TOKEN)))
        r = client.post(URL, json=VALID_BODY, headers=HDR)
        assert r.status_code == 200

        req_file = tmp_path / "retrain_requests.jsonl"
        assert req_file.exists()
        lines = [json.loads(l) for l in req_file.read_text().strip().splitlines()]
        assert len(lines) == 1
        assert lines[0]["store_id"] == "shop_001"
        assert lines[0]["status"] == "queued"

    def test_multiple_requests_appended(self, tmp_path):
        client = TestClient(create_app(Settings(data_dir=tmp_path, api_key=KEY, calendar_token=CAL_TOKEN)))
        for _ in range(3):
            client.post(URL, json=VALID_BODY, headers=HDR)

        req_file = tmp_path / "retrain_requests.jsonl"
        lines = [json.loads(l) for l in req_file.read_text().strip().splitlines()]
        assert len(lines) == 3

    def test_saved_record_has_all_fields(self, tmp_path):
        client = TestClient(create_app(Settings(data_dir=tmp_path, api_key=KEY, calendar_token=CAL_TOKEN)))
        body = {**VALID_BODY, "requested_by": "nina@codera.ai"}
        r = client.post(URL, json=body, headers=HDR)
        req_id = r.json()["request_id"]

        req_file = tmp_path / "retrain_requests.jsonl"
        record = json.loads(req_file.read_text().strip())
        assert record["request_id"] == req_id
        assert record["requested_by"] == "nina@codera.ai"
        assert record["queued_at"] != ""

    def test_different_stores_saved_separately(self, tmp_path):
        client = TestClient(create_app(Settings(data_dir=tmp_path, api_key=KEY, calendar_token=CAL_TOKEN)))
        client.post(URL, json={**VALID_BODY, "store_id": "store_a"}, headers=HDR)
        client.post(URL, json={**VALID_BODY, "store_id": "store_b"}, headers=HDR)

        req_file = tmp_path / "retrain_requests.jsonl"
        lines = [json.loads(l) for l in req_file.read_text().strip().splitlines()]
        store_ids = {l["store_id"] for l in lines}
        assert store_ids == {"store_a", "store_b"}


# ---------------------------------------------------------------------------
# store.py helpers
# ---------------------------------------------------------------------------

class TestStoreHelpers:
    def test_get_retrain_requests_empty(self, tmp_path):
        from crm_api import store as _store
        result = _store.get_retrain_requests(str(tmp_path))
        assert result == []

    def test_get_retrain_requests_returns_saved(self, tmp_path):
        from crm_api import store as _store
        _store.append_retrain_request(str(tmp_path), {
            "request_id": "req_1", "store_id": "s1", "status": "queued"
        })
        result = _store.get_retrain_requests(str(tmp_path))
        assert len(result) == 1
        assert result[0]["request_id"] == "req_1"

    def test_get_retrain_requests_filter_by_store(self, tmp_path):
        from crm_api import store as _store
        _store.append_retrain_request(str(tmp_path), {"store_id": "s1", "status": "queued"})
        _store.append_retrain_request(str(tmp_path), {"store_id": "s2", "status": "queued"})
        result = _store.get_retrain_requests(str(tmp_path), store_id="s1")
        assert len(result) == 1
        assert result[0]["store_id"] == "s1"

    def test_get_retrain_requests_filter_by_status(self, tmp_path):
        from crm_api import store as _store
        _store.append_retrain_request(str(tmp_path), {"store_id": "s1", "status": "queued"})
        _store.append_retrain_request(str(tmp_path), {"store_id": "s1", "status": "processing"})
        queued = _store.get_retrain_requests(str(tmp_path), status="queued")
        assert len(queued) == 1
        assert queued[0]["status"] == "queued"
