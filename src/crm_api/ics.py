"""Hand-written iCalendar (RFC 5545 subset) so no extra dependency is needed."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from .auth import require_key_or_calendar_token
from .errors import valid_month
from .loaders import TZ, load_month
from .plans import build_events

router = APIRouter()

_VTIMEZONE = [
    "BEGIN:VTIMEZONE", "TZID:Asia/Bangkok",
    "BEGIN:STANDARD", "DTSTART:19700101T000000", "TZOFFSETFROM:+0700", "TZOFFSETTO:+0700", "TZNAME:+07",
    "END:STANDARD", "END:VTIMEZONE",
]


def _esc(text: str) -> str:
    return (text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n"))


def _fold(line: str) -> str:
    """Fold to <=75 octets per physical line, never splitting a UTF-8 character."""
    out, cur, size, limit = [], "", 0, 75
    for ch in line:
        n = len(ch.encode("utf-8"))
        if size + n > limit:
            out.append(cur)
            cur, size, limit = " " + ch, 1 + n, 75
        else:
            cur += ch
            size += n
    out.append(cur)
    return "\r\n".join(out)


def _local(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%Y%m%dT%H%M%S")


def _utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def render_calendar(month: str, events: list[dict], generated_at: datetime) -> str:
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//CRM Campaign API//EN", "CALSCALE:GREGORIAN",
             f"X-WR-CALNAME:{_esc('Campaign ' + month)}", "X-WR-TIMEZONE:Asia/Bangkok", *_VTIMEZONE]
    for e in events:
        start = datetime.fromisoformat(e["start"])
        top = e["actions"][0]["name"] if e["actions"] else "Campaign"
        desc = "\n".join([
            f"Window: {e['send_window']}", f"Channel: {e.get('channel') or '-'}",
            f"Customers: {e['customer_count']}",
            "Actions: " + ", ".join(f"{a['name']} ({a['count']})" for a in e["actions"]),
            f"Expected profit: {e['expected_profit']} THB"])
        lines += [
            "BEGIN:VEVENT", f"UID:{e['id']}@crm-api", f"DTSTAMP:{_utc(generated_at)}",
            f"LAST-MODIFIED:{_utc(generated_at)}", "SEQUENCE:0",
            f"DTSTART;TZID=Asia/Bangkok:{_local(start)}",
            f"DTEND;TZID=Asia/Bangkok:{_local(start + timedelta(minutes=30))}",
            "SUMMARY:" + _esc(f"Wave {e['wave']}: {top} ({e['customer_count']} customers)"),
            "DESCRIPTION:" + _esc(desc), "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold(l) for l in lines) + "\r\n"


@router.get("/plans/{month}/calendar.ics", dependencies=[Depends(require_key_or_calendar_token)])
def calendar_feed(request: Request, month: str = Depends(valid_month)):
    data = load_month(request.app.state.settings.data_dir, month)
    body = render_calendar(month, build_events(month, data.schedule), data.generated_at)
    return Response(content=body, media_type="text/calendar; charset=utf-8", headers={"ETag": data.etag})
