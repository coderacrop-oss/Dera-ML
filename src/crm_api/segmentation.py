"""The shop's customer segmentation (customer -> segment) from customer_segments_latest.csv, independent of any monthly plan."""
from __future__ import annotations

import bisect
import hashlib
import io
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Query, Request
from fastapi import Path as PathParam

from . import docs
from .cursor import decode_cursor, encode_cursor
from .errors import ApiError
from .loaders import TZ
from .plans import respond

router = APIRouter()
FILENAME = "customer_segments_latest.csv"
_COLUMNS = {"customer_id", "segment_id", "segment_name"}


@dataclass
class SegmentationData:
    df: pd.DataFrame  # sorted by customer_id
    etag: str
    generated_at: datetime
    summary: list[dict]
    _memo: dict = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def memo(self, key: str, compute: Callable[[], object]):
        if key in self._memo:
            return self._memo[key]
        with self._lock:
            if key not in self._memo:
                self._memo[key] = compute()
            return self._memo[key]


_cache: dict[str, tuple[tuple, SegmentationData]] = {}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _signature(path: Path) -> tuple | None:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return None


def _parse(raw: bytes, mtime: float) -> SegmentationData:
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    if not _COLUMNS <= set(df.columns):
        raise ValueError("missing columns")
    df["segment_id"] = pd.to_numeric(df["segment_id"], errors="raise").astype(int)
    df["segment_name"] = df["segment_name"].where(df["segment_name"].notna(), None)
    df = df.dropna(subset=["customer_id"]).drop_duplicates("customer_id").sort_values("customer_id").reset_index(drop=True)
    summary = [
        {"id": int(sid), "name": next((n for n in g["segment_name"] if n), None), "customers": int(len(g))}
        for sid, g in df.groupby("segment_id", sort=True)
    ]
    return SegmentationData(df=df, etag='"' + hashlib.sha256(raw).hexdigest()[:16] + '"',
                            generated_at=datetime.fromtimestamp(mtime, TZ), summary=summary)


def load_segmentation(data_dir: Path) -> SegmentationData:
    """404 when the pipeline never wrote the file, 503 when it cannot be read; cached by (mtime, size), failures never cached."""
    path = Path(data_dir) / FILENAME
    key = str(path.resolve())
    sig = _signature(path)
    if sig is None:
        raise ApiError(404, "segmentation_not_found", "no customer segmentation yet (the pipeline has not run)")
    hit = _cache.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        sig = _signature(path)
        if sig is None:
            raise ApiError(404, "segmentation_not_found", "no customer segmentation yet (the pipeline has not run)")
        hit = _cache.get(key)
        if hit and hit[0] == sig:
            return hit[1]
        try:
            data = _parse(path.read_bytes(), path.stat().st_mtime)
        except Exception as exc:
            raise ApiError(503, "data_unavailable", f"customer segmentation is unreadable: {type(exc).__name__}") from exc
        _cache[key] = (sig, data)
        return data


def _rows(df: pd.DataFrame) -> list[dict]:
    return [{"customer_id": r.customer_id, "segment_id": int(r.segment_id),
             "segment_name": r.segment_name if isinstance(r.segment_name, str) else None} for r in df.itertuples(index=False)]


@router.get("/segmentation", responses=docs.ERR_SEGMENTATION, **docs.SEGMENTATION)
def segmentation(request: Request):
    data = load_segmentation(request.app.state.settings.data_dir)
    body = {"generated_at": data.generated_at.isoformat(), "total_customers": int(len(data.df)), "segments": data.summary}
    return respond(request, data, body)


@router.get("/segmentation/customers", responses=docs.ERR_SEGMENTATION, **docs.SEGMENTATION_CUSTOMERS)
def segmentation_customers(
    request: Request,
    cursor: str | None = Query(None, description=docs.P_CURSOR),
    limit: int = Query(100, ge=1, le=1000, description=docs.P_LIMIT),
    segment_id: int | None = Query(None, description=docs.P_SEGMENT_ID),
):
    data = load_segmentation(request.app.state.settings.data_dir)
    scope = "segmentation" if segment_id is None else f"segmentation:{segment_id}"

    def build():
        df = data.df if segment_id is None else data.df[data.df["segment_id"] == segment_id]
        rows = _rows(df)
        return [r["customer_id"] for r in rows], rows

    ids, rows = data.memo(scope, build)
    start = 0
    if cursor:
        c = decode_cursor(cursor)
        if c["e"] != scope:
            raise ApiError(422, "invalid_cursor", "cursor นี้เป็นของการค้นหาอื่น")
        if c["t"] != data.etag:
            raise ApiError(409, "data_changed", docs.ERR_SEGMENTATION[409]["description"])
        start = bisect.bisect_right(ids, c["a"])
    page = rows[start:start + limit]
    more = start + limit < len(rows)
    body = {"total": len(rows), "customers": page,
            "next_cursor": encode_cursor(scope, data.etag, page[-1]["customer_id"]) if more and page else None}
    return respond(request, data, body)


@router.get("/segmentation/customers/{customer_id}", responses=docs.ERR_SEGMENTATION, **docs.SEGMENTATION_CUSTOMER)
def segmentation_customer(request: Request, customer_id: str = PathParam(..., description=docs.P_CUSTOMER)):
    data = load_segmentation(request.app.state.settings.data_dir)
    by_id = data.memo("by_id", lambda: {r["customer_id"]: r for r in _rows(data.df)})
    row = by_id.get(customer_id)
    if row is None:
        raise ApiError(404, "customer_not_found", "no segment for this customer")
    return respond(request, data, row)
