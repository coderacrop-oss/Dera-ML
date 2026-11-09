"""store.py — Simple append-only JSONL storage for campaign history, outcomes, and segment snapshots.

Repo 1 storage is intentionally lightweight — Repo 2 owns the production DB.
These files exist for: (1) ML retraining signal, (2) trend comparison in /segment/health.

Thread-safety: each write path uses atomic_io to avoid partial writes.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _locks_guard:
        if path not in _locks:
            _locks[path] = threading.Lock()
        return _locks[path]


# ---------------------------------------------------------------------------
# Low-level JSONL helpers
# ---------------------------------------------------------------------------

def _append_line(path: str, record: dict) -> None:
    """Atomically append one JSON line to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock_for(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _read_lines(path: str) -> list[dict]:
    """Read all lines from a JSONL file, skipping malformed lines."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _campaign_history_path(data_dir: str) -> str:
    return os.path.join(data_dir, "campaign_history.jsonl")


def _outcomes_path(data_dir: str) -> str:
    return os.path.join(data_dir, "outcomes.jsonl")


def _snapshots_path(data_dir: str) -> str:
    return os.path.join(data_dir, "segment_snapshots.jsonl")


def _pos_path(data_dir: str, store_id: str) -> str:
    safe = "".join(c for c in store_id if c.isalnum() or c in ("-", "_"))
    return os.path.join(data_dir, "stores", safe, "latest_pos.csv")


# ---------------------------------------------------------------------------
# Campaign history
# ---------------------------------------------------------------------------

def append_plan(data_dir: str, record: dict) -> None:
    """Append a campaign plan result to campaign_history.jsonl."""
    record.setdefault("saved_at", _now_iso())
    _append_line(_campaign_history_path(data_dir), record)


def get_plans(data_dir: str, store_id: str, limit: int = 20, offset: int = 0) -> list[dict]:
    """Return plan records for a store, newest first."""
    all_records = _read_lines(_campaign_history_path(data_dir))
    store_records = [r for r in all_records if r.get("store_id") == store_id]
    store_records.reverse()   # newest first
    return store_records[offset: offset + limit]


def get_plan_by_id(data_dir: str, plan_id: str) -> dict | None:
    """Return a single plan by plan_id, or None."""
    for record in _read_lines(_campaign_history_path(data_dir)):
        if record.get("plan_id") == plan_id:
            return record
    return None


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

def append_outcome(data_dir: str, record: dict) -> None:
    """Append an outcome record to outcomes.jsonl."""
    record.setdefault("saved_at", _now_iso())
    _append_line(_outcomes_path(data_dir), record)


def get_outcome_by_plan(data_dir: str, plan_id: str) -> dict | None:
    """Return the most recent outcome for a plan_id, or None."""
    matches = [r for r in _read_lines(_outcomes_path(data_dir)) if r.get("plan_id") == plan_id]
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Segment snapshots
# ---------------------------------------------------------------------------

def append_snapshot(data_dir: str, record: dict) -> None:
    """Append a segment snapshot to segment_snapshots.jsonl."""
    record.setdefault("saved_at", _now_iso())
    _append_line(_snapshots_path(data_dir), record)


def get_snapshots(data_dir: str, store_id: str) -> list[dict]:
    """Return all snapshots for a store, oldest first."""
    return [r for r in _read_lines(_snapshots_path(data_dir)) if r.get("store_id") == store_id]


def get_prev_snapshot(
    data_dir: str,
    store_id: str,
    before_iso: str,
    within_days: int = 45,
) -> dict | None:
    """Return the most recent snapshot strictly before `before_iso` and within `within_days` days.

    Used for trend comparison: typically want the snapshot from ~30 days ago.
    """
    from datetime import timedelta

    before_dt = datetime.fromisoformat(before_iso)
    cutoff_dt = before_dt - timedelta(days=within_days)

    candidates = []
    for snap in get_snapshots(data_dir, store_id):
        ts_str = snap.get("snapshot_at") or snap.get("saved_at", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if cutoff_dt <= ts < before_dt:
                candidates.append((ts, snap))
        except ValueError:
            continue

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


# ---------------------------------------------------------------------------
# POS file management
# ---------------------------------------------------------------------------

def save_pos_file(data_dir: str, store_id: str, csv_bytes: bytes) -> str:
    """Save uploaded POS CSV bytes to disk and return the file path."""
    path = _pos_path(data_dir, store_id)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _lock_for(path):
        with open(path, "wb") as f:
            f.write(csv_bytes)
    return path


def get_pos_path(data_dir: str, store_id: str) -> str | None:
    """Return the path to the latest POS file for a store, or None if not found."""
    path = _pos_path(data_dir, store_id)
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Retrain requests
# ---------------------------------------------------------------------------

def _retrain_requests_path(data_dir: str) -> str:
    return os.path.join(data_dir, "retrain_requests.jsonl")


def append_retrain_request(data_dir: str, record: dict) -> None:
    """Persist a retrain request to retrain_requests.jsonl."""
    _append_line(_retrain_requests_path(data_dir), record)


def get_retrain_requests(
    data_dir: str,
    store_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Return retrain requests, optionally filtered by store_id and/or status."""
    records = _read_lines(_retrain_requests_path(data_dir))
    if store_id:
        records = [r for r in records if r.get("store_id") == store_id]
    if status:
        records = [r for r in records if r.get("status") == status]
    return records[-limit:]


def get_retrain_request_by_id(data_dir: str, request_id: str) -> dict | None:
    """Return a single retrain request by request_id, or None if not found."""
    for rec in _read_lines(_retrain_requests_path(data_dir)):
        if rec.get("request_id") == request_id:
            return rec
    return None


def update_retrain_request(data_dir: str, request_id: str, updates: dict) -> bool:
    """Rewrite retrain_requests.jsonl updating the last matching request_id.

    Returns True if the record was found and updated, False otherwise.
    Thread-safe via the same path-level lock used by _append_line.
    """
    path = _retrain_requests_path(data_dir)
    lock = _lock_for(path)
    with lock:
        if not os.path.exists(path):
            return False
        lines: list[str] = []
        found = False
        with open(path, encoding="utf-8") as f:
            raw = f.readlines()
        # Update the LAST matching record (most recent)
        last_idx = None
        for i, line in enumerate(raw):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
                if rec.get("request_id") == request_id:
                    last_idx = i
            except json.JSONDecodeError:
                pass
        for i, line in enumerate(raw):
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            try:
                rec = json.loads(stripped)
                if i == last_idx:
                    rec.update(updates)
                    lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
                    found = True
                else:
                    lines.append(line)
            except json.JSONDecodeError:
                lines.append(line)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return found
