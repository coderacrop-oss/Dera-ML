from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .errors import ApiError

_SUMMARY_RE = re.compile(r"^campaign_summary_(\d{4}-(?:0[1-9]|1[0-2]))\.json$")


def list_months(data_dir: Path) -> list[str]:
    """Months with BOTH a summary json and a schedule csv, ascending."""
    months = []
    for p in Path(data_dir).glob("campaign_summary_*.json"):
        m = _SUMMARY_RE.match(p.name)
        if m and (Path(data_dir) / f"campaign_schedule_{m.group(1)}.csv").exists():
            months.append(m.group(1))
    return sorted(months)


# ── Month data ────────────────────────────────────────────────────────────────


TZ = ZoneInfo("Asia/Bangkok")
_REQUIRED_SCHEDULE_COLS = {"customer_id", "wave", "recommended_action", "send_datetime"}


@dataclass
class MonthData:
    month: str
    summary: dict
    schedule: pd.DataFrame      # send_datetime is tz-aware (Asia/Bangkok)
    holdout: pd.DataFrame
    segments: dict | None
    etag: str
    generated_at: datetime
    by_customer: dict | None = field(default=None, repr=False)  # lazily built by catalog


_cache: dict[tuple, tuple[tuple, MonthData]] = {}


def _signature(paths: list[Path]) -> tuple:
    sig = []
    for p in paths:
        try:
            st = p.stat()
            sig.append((st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            sig.append(None)
    return tuple(sig)


def _unavailable(name: str, exc: Exception) -> ApiError:
    return ApiError(503, "data_unavailable", f"{name} is missing or unreadable: {type(exc).__name__}")


def load_month(data_dir: Path, month: str) -> MonthData:
    """Load one month's files. 404 if the month has no plan, 503 if a file fails to parse.

    Cached by (mtime, size) of every file; failures are never cached.
    """
    d = Path(data_dir)
    p_summary = d / f"campaign_summary_{month}.json"
    p_sched = d / f"campaign_schedule_{month}.csv"
    p_hold = d / f"ab_holdout_{month}.csv"
    p_seg = d / f"segments_{month}.json"
    if not p_summary.exists() or not p_sched.exists():
        raise ApiError(404, "month_not_found", f"no plan for {month}")

    paths = [p_summary, p_sched, p_hold, p_seg]
    sig = _signature(paths)
    key = (str(d.resolve()), month)
    hit = _cache.get(key)
    if hit and hit[0] == sig:
        return hit[1]

    try:
        raw = {p: p.read_bytes() for p in paths if p.exists()}
        summary = json.loads(raw[p_summary].decode("utf-8"))
        schedule = pd.read_csv(io.BytesIO(raw[p_sched]))
        missing = _REQUIRED_SCHEDULE_COLS - set(schedule.columns)
        if missing:
            raise ValueError(f"schedule missing columns {sorted(missing)}")
        ts = pd.to_datetime(schedule["send_datetime"], errors="raise")
        schedule["send_datetime"] = ts.dt.tz_localize(TZ) if ts.dt.tz is None else ts.dt.tz_convert(TZ)
        holdout = pd.read_csv(io.BytesIO(raw[p_hold])) if p_hold in raw else pd.DataFrame(columns=["customer_id"])
        segments = json.loads(raw[p_seg].decode("utf-8")) if p_seg in raw else None
    except ApiError:
        raise
    except Exception as exc:  # parse errors of any file → retry-able 503
        raise _unavailable(f"data for {month}", exc) from exc

    h = hashlib.sha256()
    for p in paths:
        h.update(raw.get(p, b"\0"))
    data = MonthData(
        month=month, summary=summary, schedule=schedule, holdout=holdout, segments=segments,
        etag='"' + h.hexdigest()[:16] + '"',
        generated_at=datetime.fromtimestamp(p_summary.stat().st_mtime, TZ),
    )
    _cache[key] = (sig, data)
    return data
