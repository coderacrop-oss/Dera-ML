import json

import pandas as pd

from conftest import HDR, make_schedule_rows, write_fixture_month


def _schedule(data_dir):
    return pd.read_csv(data_dir / "campaign_schedule_2026-12.csv")


def test_plan_requires_key_and_validates_month(client):
    assert client.get("/v1/plans/2026-12").status_code == 401
    for bad in ("2026-13", "2026-1"):
        r = client.get(f"/v1/plans/{bad}", headers=HDR)
        assert r.status_code == 422 and set(r.json()["error"]) == {"code", "message"}
    assert client.get("/v1/plans/..%2Fx", headers=HDR).status_code in (404, 422)


def test_unknown_month_404(client):
    r = client.get("/v1/plans/2027-01", headers=HDR)
    assert r.status_code == 404 and r.json()["error"]["code"] == "month_not_found"


def test_plan_totals_match_source_files(client, data_dir):
    sched = _schedule(data_dir)
    body = client.get("/v1/plans/2026-12", headers=HDR).json()
    assert body["month"] == "2026-12" and body["timezone"] == "Asia/Bangkok" and body["currency"] == "THB"
    assert body["gate_status"] == "PASSED"
    t = body["totals"]
    assert t["scheduled"] == len(sched) and t["holdout"] == 2
    assert t["expected_profit"] == round(sched.expected_profit.sum(), 2)
    counts = sched.groupby("wave").size().to_dict()
    assert {w["wave"]: w["count"] for w in body["waves"]} == {1: counts[1], 2: counts[2], 3: counts[3]}
    assert [w["weeks"] for w in body["waves"]] == [[1, 3], [2, 4], [1]]
    assert body["peak_hours"] == [11, 12] and body["off_peak_hours"] == [8, 9]


def test_events_sum_equals_scheduled_and_shape(client, data_dir):
    ev = client.get("/v1/plans/2026-12/events", headers=HDR).json()["events"]
    assert sum(e["customer_count"] for e in ev) == len(_schedule(data_dir))
    e = ev[0]
    assert set(e) >= {"id", "start", "wave", "send_window", "channel", "customer_count", "actions", "expected_profit"}
    assert e["start"].endswith("+07:00")
    assert e["id"].startswith("2026-12|") and len({x["id"] for x in ev}) == len(ev)


def test_events_say_which_batch_of_the_wave_they_are(client):
    events = client.get("/v1/plans/2026-12/events", headers=HDR).json()["events"]
    for wave in (1, 2, 3):
        mine = [e for e in events if e["wave"] == wave]            # already in time order
        assert [e["batch_index"] for e in mine] == list(range(1, len(mine) + 1))
        assert {e["batch_total"] for e in mine} == {len(mine)}
        assert {e["wave_customer_total"] for e in mine} == {sum(e["customer_count"] for e in mine)}


def test_events_exclude_holdout_customers(client):
    text = client.get("/v1/plans/2026-12/events", headers=HDR).text
    assert "H001" not in text and "H002" not in text


def test_events_filters_from_to_wave(client):
    base = "/v1/plans/2026-12/events"
    only_w1 = client.get(base + "?wave=1", headers=HDR).json()["events"]
    assert only_w1 and all(e["wave"] == 1 for e in only_w1)
    day = client.get(base + "?from=2026-12-02&to=2026-12-02", headers=HDR).json()["events"]
    assert day and all(e["start"].startswith("2026-12-02") for e in day)
    assert client.get(base + "?from=2026-12-05&to=2026-12-01", headers=HDR).status_code == 422
    assert client.get(base + "?wave=9", headers=HDR).status_code == 422
    assert client.get(base + "?from=notadate", headers=HDR).status_code == 422


def test_etag_and_304(client):
    r1 = client.get("/v1/plans/2026-12", headers=HDR)
    etag = r1.headers["ETag"]
    r2 = client.get("/v1/plans/2026-12", headers={**HDR, "If-None-Match": etag})
    assert r2.status_code == 304 and r2.headers["ETag"] == etag
    r3 = client.get("/v1/plans/2026-12/events", headers={**HDR, "If-None-Match": etag})
    assert r3.status_code == 304


def test_missing_send_window_column_defaults(data_dir, client):
    rows = [{k: v for k, v in r.items() if k not in ("send_window", "send_time_hint")} for r in make_schedule_rows()]
    write_fixture_month(data_dir, schedule_rows=rows)
    ev = client.get("/v1/plans/2026-12/events", headers=HDR).json()["events"]
    assert ev and all(e["send_window"] == "Peak Window" for e in ev)


def test_empty_schedule_ok(data_dir, client):
    write_fixture_month(data_dir, schedule_rows=[])
    assert client.get("/v1/plans/2026-12/events", headers=HDR).json()["events"] == []
    t = client.get("/v1/plans/2026-12", headers=HDR).json()["totals"]
    assert t["scheduled"] == 0 and t["expected_profit"] == 0


def test_nan_profit_serialises_without_nan_token(data_dir, client):
    rows = make_schedule_rows()
    rows[0]["expected_profit"] = float("nan")
    write_fixture_month(data_dir, schedule_rows=rows)
    for path in ("/v1/plans/2026-12", "/v1/plans/2026-12/events"):
        r = client.get(path, headers=HDR)
        assert r.status_code == 200 and "NaN" not in r.text
        json.loads(r.text)


def test_missing_segments_file_gives_null_fields(data_dir, client):
    (data_dir / "segments_2026-12.json").unlink()
    body = client.get("/v1/plans/2026-12", headers=HDR).json()
    assert body["gate_status"] is None
    assert all(w["weeks"] is None for w in body["waves"])
