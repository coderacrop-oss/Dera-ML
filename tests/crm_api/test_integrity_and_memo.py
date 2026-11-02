import os
import threading

import pandas as pd

from conftest import HDR, make_schedule_rows, write_fixture_month
from crm_api import catalog, plans


def _bump(path, seconds=10):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 10**9))


# ── 1. integrity: files must agree with the summary, else 503 (never silently serve partial data) ──
def test_truncated_schedule_is_503_then_recovers(data_dir, client):
    sched = data_dir / "campaign_schedule_2026-12.csv"
    good = sched.read_text(encoding="utf-8")
    sched.write_text("\n".join(good.splitlines()[:7]) + "\n", encoding="utf-8")  # header + 6 of 12 rows, parses fine
    r = client.get("/v1/plans/2026-12", headers=HDR)
    assert r.status_code == 503 and r.json()["error"]["code"] == "data_unavailable"
    assert client.get("/v1/plans/2026-12/events", headers=HDR).status_code == 503
    sched.write_text(good, encoding="utf-8")
    _bump(sched)
    assert client.get("/v1/plans/2026-12", headers=HDR).status_code == 200


def test_summary_from_other_version_is_503(data_dir, client):
    write_fixture_month(data_dir, schedule_rows=make_schedule_rows(5))      # summary now says 5
    pd.DataFrame(make_schedule_rows(12)).to_csv(data_dir / "campaign_schedule_2026-12.csv", index=False)  # schedule 12
    assert client.get("/v1/plans/2026-12", headers=HDR).status_code == 503


def test_missing_holdout_file_is_503_when_summary_expects_holdout(data_dir, client):
    (data_dir / "ab_holdout_2026-12.csv").unlink()
    assert client.get("/v1/plans/2026-12", headers=HDR).status_code == 503


def test_holdout_count_mismatch_is_503(data_dir, client):
    pd.DataFrame([dict(customer_id="H001")]).to_csv(data_dir / "ab_holdout_2026-12.csv", index=False)
    assert client.get("/v1/plans/2026-12", headers=HDR).status_code == 503


def test_summary_without_totals_is_not_checked(data_dir, client):
    import json
    p = data_dir / "campaign_summary_2026-12.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    for k in ("total_scheduled", "total_holdout"):
        s.pop(k)
    p.write_text(json.dumps(s), encoding="utf-8")
    (data_dir / "ab_holdout_2026-12.csv").unlink()
    assert client.get("/v1/plans/2026-12", headers=HDR).status_code == 200


# ── 2. first customer lookup must not build the index once per concurrent request ──
def test_customer_index_built_once_under_concurrency(data_dir, client, monkeypatch):
    calls, real = [], catalog._build_index

    def counting(data):
        calls.append(1)
        import time
        time.sleep(0.05)
        return real(data)

    monkeypatch.setattr(catalog, "_build_index", counting)
    barrier = threading.Barrier(8)
    codes = []

    def hit():
        barrier.wait()
        codes.append(client.get("/v1/plans/2026-12/customers/C001", headers=HDR).status_code)

    ts = [threading.Thread(target=hit) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert codes == [200] * 8 and len(calls) == 1


# ── 3. derived views are computed once per loaded month, not on every request ──
def test_events_computed_once_across_requests_and_ics(data_dir, client, monkeypatch):
    calls, real = [], plans.build_events

    def counting(month, schedule):
        calls.append(1)
        return real(month, schedule)

    monkeypatch.setattr(plans, "build_events", counting)
    for _ in range(3):
        assert client.get("/v1/plans/2026-12/events", headers=HDR).status_code == 200
    assert client.get("/v1/plans/2026-12/events?wave=1", headers=HDR).status_code == 200
    assert client.get("/v1/plans/2026-12/calendar.ics", headers=HDR).status_code == 200
    assert len(calls) == 1


def test_promotions_and_ics_computed_once(data_dir, client, monkeypatch):
    import crm_api.ics as ics_mod
    calls, real = [], ics_mod.render_calendar

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(ics_mod, "render_calendar", counting)
    for _ in range(3):
        assert client.get("/v1/plans/2026-12/calendar.ics", headers=HDR).status_code == 200
    assert len(calls) == 1


def test_derived_views_refresh_when_files_change(data_dir, client):
    before = client.get("/v1/plans/2026-12/events", headers=HDR).json()["events"]
    write_fixture_month(data_dir, schedule_rows=make_schedule_rows(3))
    for f in data_dir.iterdir():
        _bump(f)
    after = client.get("/v1/plans/2026-12/events", headers=HDR).json()["events"]
    assert sum(e["customer_count"] for e in after) == 3 != sum(e["customer_count"] for e in before)
    assert client.get("/v1/plans/2026-12/calendar.ics", headers=HDR).text.count("BEGIN:VEVENT") == len(after)
