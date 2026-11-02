import pytest

from crm_api.errors import ApiError, valid_month
from crm_api.settings import Settings
from conftest import HDR


def test_health_open_no_key(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "months_available": ["2026-12"]}


def test_v1_months_requires_key(client):
    for headers in ({}, {"X-API-Key": "wrong"}):
        r = client.get("/v1/months", headers=headers)
        assert r.status_code == 401
        assert set(r.json()["error"]) == {"code", "message"}
    r = client.get("/v1/months", headers=HDR)
    assert r.status_code == 200 and r.json() == {"months": ["2026-12"]}


def test_month_needs_both_summary_and_schedule(data_dir, client):
    (data_dir / "campaign_schedule_2026-12.csv").unlink()
    assert client.get("/v1/months", headers=HDR).json() == {"months": []}


def test_settings_refuse_without_api_key_env(monkeypatch):
    monkeypatch.delenv("CRM_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        Settings.from_env()


@pytest.mark.parametrize("bad", ["2026-13", "2026-1", "../x", "2026-12/../x", ""])
def test_valid_month_rejects(bad):
    with pytest.raises(ApiError) as e:
        valid_month(bad)
    assert e.value.status == 422


def test_valid_month_accepts():
    assert valid_month("2026-12") == "2026-12"
