import os
from urllib.parse import quote

import pandas as pd

from conftest import HDR, make_schedule_rows, write_fixture_month


def _bump(path, seconds=10):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 10**9))


def _multi_rows(n=12):
    """12 customers in just two events (6 each), so pagination actually has pages."""
    rows = make_schedule_rows(n)
    for i, r in enumerate(rows):
        r["wave"] = 1 + i % 2
        r["send_window"] = "Off-Peak Window" if r["wave"] == 1 else "Peak Window"
        r["send_datetime"] = "2026-12-02 07:00:00" if r["wave"] == 1 else "2026-12-03 11:00:00"
        r["recommended_action"] = "Off-Peak Traffic Driver" if r["wave"] == 1 else "Standard Retention"
    return rows


def _use_multi(data_dir, **kw):
    write_fixture_month(data_dir, schedule_rows=kw.pop("rows", None) or _multi_rows())
    for f in data_dir.iterdir():
        _bump(f)


def _events(client):
    return client.get("/v1/plans/2026-12/events", headers=HDR).json()["events"]


def _pages(client, url, limit):
    """Follow next_cursor to the end; return (all customer dicts, list of page bodies)."""
    pages, cursor = [], None
    while True:
        q = f"?limit={limit}" + (f"&cursor={cursor}" if cursor else "")
        r = client.get(url + q, headers=HDR)
        assert r.status_code == 200, r.text
        body = r.json()
        pages.append(body)
        cursor = body["next_cursor"]
        if not cursor:
            return [c for p in pages for c in p["customers"]], pages


def test_events_expose_a_ready_made_customers_url(client):
    for e in _events(client):
        assert e["customers_url"] == f"/v1/plans/2026-12/events/{quote(e['id'], safe='')}/customers"
        assert "%7C" in e["customers_url"]


def test_pages_cover_the_event_exactly_once_and_sorted(client, data_dir):
    _use_multi(data_dir)
    sched = pd.read_csv(data_dir / "campaign_schedule_2026-12.csv")
    big = max(_events(client), key=lambda e: e["customer_count"])
    assert big["customer_count"] == 6
    customers, pages = _pages(client, big["customers_url"], limit=1)
    ids = [c["customer_id"] for c in customers]
    assert len(ids) == big["customer_count"] == len(set(ids)) and ids == sorted(ids)
    assert len(pages) == 6 and pages[-1]["next_cursor"] is None and all(p["next_cursor"] for p in pages[:-1])
    expect = sched[(sched.send_datetime.str.replace(" ", "T").str.startswith(big["start"][:16]))
                   & (sched.wave == big["wave"]) & (sched.send_window == big["send_window"])]
    assert set(ids) == set(expect.customer_id)


def test_body_shape_and_customer_fields(client):
    e = _events(client)[0]
    body = client.get(e["customers_url"], headers=HDR).json()
    assert {"month", "event_id", "start", "wave", "send_window", "customer_count", "customers", "next_cursor"} <= set(body)
    assert body["event_id"] == e["id"] and body["start"] == e["start"] and body["customer_count"] == e["customer_count"]
    c = body["customers"][0]
    assert set(c) == {"customer_id", "recommended_action", "offer", "message", "channel"}


def test_every_scheduled_customer_belongs_to_exactly_one_event_and_holdout_to_none(client, data_dir):
    _use_multi(data_dir)
    seen = []
    for e in _events(client):
        customers, _ = _pages(client, e["customers_url"], limit=1000)
        seen += [c["customer_id"] for c in customers]
    sched = pd.read_csv(data_dir / "campaign_schedule_2026-12.csv")
    assert sorted(seen) == sorted(sched.customer_id)
    assert not {"H001", "H002"} & set(seen)


def test_limit_is_validated_and_default_applies(client):
    url = _events(client)[0]["customers_url"]
    for bad in (0, 1001, -1):
        assert client.get(f"{url}?limit={bad}", headers=HDR).status_code == 422
    assert client.get(f"{url}?limit=1000", headers=HDR).status_code == 200
    assert client.get(url, headers=HDR).status_code == 200


def test_unknown_event_is_404(client):
    r = client.get("/v1/plans/2026-12/events/nope/customers", headers=HDR)
    assert r.status_code == 404 and r.json()["error"]["code"] == "event_not_found"


def test_invalid_or_foreign_cursor_is_422(client, data_dir):
    _use_multi(data_dir)
    ev = max(_events(client), key=lambda e: e["customer_count"])
    other = next(e for e in _events(client) if e["id"] != ev["id"])
    for cur in ("garbage", "e30", "!!!!"):
        r = client.get(f"{ev['customers_url']}?cursor={cur}", headers=HDR)
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_cursor", cur
    cursor = client.get(f"{ev['customers_url']}?limit=1", headers=HDR).json()["next_cursor"]
    r = client.get(f"{other['customers_url']}?cursor={cursor}", headers=HDR)   # cursor of another event
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_cursor"


def test_plan_changed_between_pages_is_409(client, data_dir):
    _use_multi(data_dir)
    ev = max(_events(client), key=lambda e: e["customer_count"])
    cursor = client.get(f"{ev['customers_url']}?limit=1", headers=HDR).json()["next_cursor"]
    rows = _multi_rows()
    rows[1]["message"] = "ข้อความใหม่"                 # same events, different content → different ETag
    _use_multi(data_dir, rows=rows)
    r = client.get(f"{ev['customers_url']}?cursor={cursor}", headers=HDR)
    assert r.status_code == 409 and r.json()["error"]["code"] == "plan_changed"
    fresh = client.get(f"{ev['customers_url']}?limit=1", headers=HDR)          # restarting works
    assert fresh.status_code == 200


def test_requires_key_and_supports_etag(client):
    url = _events(client)[0]["customers_url"]
    assert client.get(url).status_code == 401
    etag = client.get(url, headers=HDR).headers["ETag"]
    assert client.get(url, headers={**HDR, "If-None-Match": etag}).status_code == 304
