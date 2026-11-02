import pandas as pd

from conftest import HDR


def _sched(data_dir):
    return pd.read_csv(data_dir / "campaign_schedule_2026-12.csv")


def test_catalog_endpoints_require_key(client):
    for path in ("segments", "promotions", "customers/C001"):
        assert client.get(f"/v1/plans/2026-12/{path}").status_code == 401


def test_promotions_counts_sum_to_scheduled(client, data_dir):
    sched = _sched(data_dir)
    promos = client.get("/v1/plans/2026-12/promotions", headers=HDR).json()["promotions"]
    assert sum(p["customer_count"] for p in promos) == len(sched)
    assert {p["name"] for p in promos} == set(sched.recommended_action)
    p1 = next(p for p in promos if p["wave"] == 1)
    assert set(p1) >= {"name", "wave", "channel", "customer_count", "expected_profit", "example_message"}
    assert p1["expected_profit"] == round(sched[sched.wave == 1].expected_profit.sum(), 2)
    assert p1["example_message"]


def test_customer_found(client, data_dir):
    row = _sched(data_dir).iloc[0]
    r = client.get(f"/v1/plans/2026-12/customers/{row.customer_id}", headers=HDR)
    assert r.status_code == 200
    b = r.json()
    assert b["customer_id"] == row.customer_id and b["wave"] == int(row.wave)
    assert b["message"] == row.message and b["recommended_action"] == row.recommended_action
    assert b["send_datetime"].startswith("2026-12-01") and b["send_datetime"].endswith("+07:00")


def test_customer_holdout_and_unknown_are_indistinguishable(client):
    holdout = client.get("/v1/plans/2026-12/customers/H001", headers=HDR)
    unknown = client.get("/v1/plans/2026-12/customers/NOPE", headers=HDR)
    assert holdout.status_code == unknown.status_code == 404
    assert holdout.json() == unknown.json()
    assert holdout.json()["error"]["code"] == "customer_not_found"


def test_customer_id_with_slash_or_unicode_is_404_not_500(client):
    for cid in ("a%2Fb", "ลูกค้า", "..%2F..%2Fetc%2Fpasswd", "a b"):
        r = client.get(f"/v1/plans/2026-12/customers/{cid}", headers=HDR)
        assert r.status_code == 404, cid
        assert set(r.json()["error"]) == {"code", "message"}


def test_segments_values_match_source_json(client):
    segs = client.get("/v1/plans/2026-12/segments", headers=HDR).json()["segments"]
    assert segs[0]["name"] == "1. Champions" and segs[0]["customers"] == 516
    assert segs[0]["unit_economics"]["total_projected_profit"] == 106433.28
    assert segs[0]["action_breakdown"] == {"VIP Early Access": 12}


def test_segments_absent_file_returns_reason(data_dir, client):
    (data_dir / "segments_2026-12.json").unlink()
    r = client.get("/v1/plans/2026-12/segments", headers=HDR)
    assert r.status_code == 200
    assert r.json() == {"month": "2026-12", "segments": None, "reason": "segments file not generated"}
