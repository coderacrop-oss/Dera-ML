from __future__ import annotations

import hashlib
import io
import json
import re
import threading
from collections.abc import Callable
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
    _memo: dict = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)  # re-entrant: memo() may nest

    def memo(self, key: str, compute: Callable[[], object]):
        """Compute a derived view once per loaded month (thread-safe); it is dropped with this object
        when the underlying files change and the month is reloaded."""
        if key in self._memo:
            return self._memo[key]
        with self._lock:
            if key not in self._memo:
                self._memo[key] = compute()
            return self._memo[key]


_cache: dict[tuple, tuple[tuple, MonthData]] = {}
_load_locks: dict[tuple, threading.Lock] = {}
_load_locks_guard = threading.Lock()


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


def _check_consistent(month: str, summary: dict, schedule: pd.DataFrame, holdout: pd.DataFrame,
                      holdout_file_present: bool) -> None:
    """Files of one month are written one after another, so a reader can see a half-written CSV that
    still parses, or files from two different runs. The summary records the row counts; if they disagree
    with the files, answer a retry-able 503 instead of silently serving partial data."""
    want_s, want_h = summary.get("total_scheduled"), summary.get("total_holdout")
    if want_s is not None and int(want_s) != len(schedule):
        raise ApiError(503, "data_unavailable",
                       f"{month}: schedule has {len(schedule)} rows but summary says {want_s} (files being updated?)")
    if want_h is not None and int(want_h) != len(holdout):
        why = "holdout file missing" if not holdout_file_present else f"holdout has {len(holdout)} rows"
        raise ApiError(503, "data_unavailable", f"{month}: {why} but summary says {want_h} (files being updated?)")


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
    key = (str(d.resolve()), month)
    hit = _cache.get(key)
    if hit and hit[0] == _signature(paths):
        return hit[1]

    # Single-flight per month: concurrent first requests must not each parse the files.
    with _load_locks_guard:
        lock = _load_locks.setdefault(key, threading.Lock())
    with lock:
        sig = _signature(paths)
        hit = _cache.get(key)
        if hit and hit[0] == sig:
            return hit[1]
        return _load_uncached(d, month, paths, sig, key)


def _load_uncached(d: Path, month: str, paths: list[Path], sig: tuple, key: tuple) -> MonthData:
    p_summary, p_sched, p_hold, p_seg = paths
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
        _check_consistent(month, summary, schedule, holdout, p_hold in raw)
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
