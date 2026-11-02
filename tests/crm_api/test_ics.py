from datetime import datetime, timezone

from fastapi.testclient import TestClient

from conftest import CAL_TOKEN, HDR, KEY
from crm_api.app import create_app
from crm_api.ics import render_calendar
from crm_api.settings import Settings

GEN = datetime(2026, 11, 3, 3, 0, tzinfo=timezone.utc)


def _event(action="Offer", count=3, eid="2026-12|2026-12-02T07:00|w1|off_peak_window"):
    return {"id": eid, "start": "2026-12-02T07:00:00+07:00", "wave": 1, "send_window": "Off-Peak Window",
            "channel": "Email/SMS", "customer_count": count, "actions": [{"name": action, "count": count}],
            "expected_profit": 12.5}


def _unfold(text):
    return text.replace("\r\n ", "")


def test_ics_structure_and_crlf(client):
    r = client.get("/v1/plans/2026-12/calendar.ics", headers=HDR)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/calendar")
    body = r.text
    assert body.startswith("BEGIN:VCALENDAR\r\n") and body.endswith("END:VCALENDAR\r\n")
    assert body.replace("\r\n", "").count("\n") == 0
    n_events = len(client.get("/v1/plans/2026-12/events", headers=HDR).json()["events"])
    assert body.count("BEGIN:VEVENT") == body.count("END:VEVENT") == n_events > 0
    assert "TZID:Asia/Bangkok" in body and "DTSTART;TZID=Asia/Bangkok:20261201T0" in body


def test_ics_uid_stable_across_regeneration(client, data_dir):
    uids = lambda t: sorted(l for l in _unfold(t).split("\r\n") if l.startswith("UID:"))
    a = uids(client.get("/v1/plans/2026-12/calendar.ics", headers=HDR).text)
    b = uids(client.get("/v1/plans/2026-12/calendar.ics", headers=HDR).text)
    assert a == b and len(set(a)) == len(a)


def test_ics_escapes_special_characters():
    out = _unfold(render_calendar("2026-12", [_event("ส่วนลด, พิเศษ; ที่ 1\nบรรทัดสอง \\ end")], GEN))
    assert r"ส่วนลด\, พิเศษ\; ที่ 1\nบรรทัดสอง \\ end" in out
    assert "\nบรรทัดสอง" not in out  # raw newline never reaches the file


def test_ics_long_thai_line_folded_on_character_boundary():
    name = "ส่วนลดพิเศษ" * 40
    out = render_calendar("2026-12", [_event(name)], GEN)
    lines = out.split("\r\n")
    assert all(len(l.encode("utf-8")) <= 75 for l in lines)
    assert any(l.startswith(" ") for l in lines)  # something was actually folded
    assert name in _unfold(out).replace("\\n", "\n")
    out.encode("utf-8").decode("utf-8")


def test_ics_dtend_and_timestamps():
    out = render_calendar("2026-12", [_event()], GEN)
    assert "DTSTART;TZID=Asia/Bangkok:20261202T070000" in out
    assert "DTEND;TZID=Asia/Bangkok:20261202T073000" in out
    assert "DTSTAMP:20261103T030000Z" in out and "LAST-MODIFIED:20261103T030000Z" in out


def test_ics_excludes_holdout(client):
    text = client.get("/v1/plans/2026-12/calendar.ics", headers=HDR).text
    assert "H001" not in text and "H002" not in text


def test_ics_requires_credentials(client):
    assert client.get("/v1/plans/2026-12/calendar.ics").status_code == 401
    assert client.get("/v1/plans/2026-12/calendar.ics?token=wrong").status_code == 401


def test_ics_token_only_works_on_ics(client):
    assert client.get(f"/v1/plans/2026-12/calendar.ics?token={CAL_TOKEN}").status_code == 200
    for path in ("/v1/plans/2026-12", "/v1/plans/2026-12/events", "/v1/months"):
        assert client.get(f"{path}?token={CAL_TOKEN}").status_code == 401, path


def test_calendar_token_disabled_when_not_configured(data_dir):
    c = TestClient(create_app(Settings(data_dir=data_dir, api_key=KEY, calendar_token=None)))
    assert c.get("/v1/plans/2026-12/calendar.ics?token=").status_code == 401
    assert c.get("/v1/plans/2026-12/calendar.ics", headers=HDR).status_code == 200
