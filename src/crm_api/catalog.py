from __future__ import annotations

import bisect

import pandas as pd
from fastapi import APIRouter, Depends, Path, Query, Request

from . import docs
from .cursor import decode_cursor, encode_cursor
from .errors import ApiError, valid_month
from .loaders import load_month
from .plans import _with_window, event_id, events_for, respond

router = APIRouter()


@router.get("/plans/{month}/segments", **docs.SEGMENTS)
def segments(request: Request, month: str = Depends(valid_month)):
    data = load_month(request.app.state.settings.data_dir, month)
    if data.segments is None:
        return respond(request, data, {"month": month, "segments": None, "reason": "segments file not generated"})
    return respond(request, data, {"month": month, "segments": data.segments.get("segments")})


def _build_promotions(data) -> list[dict]:
    df = _with_window(data.schedule)
    out = []
    for action, g in df.groupby("recommended_action", sort=True):
        msg = g["message"].dropna() if "message" in g.columns else pd.Series(dtype=str)
        out.append({
            "name": action,
            "wave": int(g["wave"].iloc[0]),
            "channel": g["channel"].dropna().iloc[0] if g["channel"].notna().any() else None,
            "customer_count": int(len(g)),
            "expected_profit": round(float(g["expected_profit"].sum()), 2),
            "example_message": msg.iloc[0] if len(msg) else None,
        })
    return out


@router.get("/plans/{month}/promotions", **docs.PROMOTIONS)
def promotions(request: Request, month: str = Depends(valid_month)):
    data = load_month(request.app.state.settings.data_dir, month)
    return respond(request, data, {"month": month, "promotions": data.memo("promotions", lambda: _build_promotions(data))})


_INDEX_COLS = ["customer_id", "wave", "recommended_action", "offer", "message", "channel", "send_window",
               "send_datetime"]


def _build_index(data) -> dict:
    """customer_id -> row dict (earliest send wins). Only the columns the endpoint serves are kept."""
    df = _with_window(data.schedule)
    df = df[[c for c in _INDEX_COLS if c in df.columns]].sort_values("send_datetime")
    df = df.drop_duplicates("customer_id", keep="first")
    return {str(r["customer_id"]): r for r in df.to_dict("records")}


def _index(data) -> dict:
    return data.memo("customers", lambda: _build_index(data))


@router.get("/plans/{month}/customers/{customer_id}", **docs.CUSTOMER)
def customer_offer(request: Request, customer_id: str = Path(..., description=docs.P_CUSTOMER), month: str = Depends(valid_month)):
    data = load_month(request.app.state.settings.data_dir, month)
    row = _index(data).get(customer_id)
    if row is None:  # holdout, suppressed and unknown customers all look the same on purpose
        raise ApiError(404, "customer_not_found", "no scheduled offer for this customer in this plan")
    body = {k: row.get(k) for k in ("customer_id", "wave", "recommended_action", "offer", "message",
                                    "channel", "send_window")}
    body["send_datetime"] = row["send_datetime"].isoformat()
    return respond(request, data, body)


# ── recipients of one event (batch sending) ──────────────────────────────────
_RECIPIENT_COLS = ["customer_id", "recommended_action", "offer", "message", "channel"]


def _build_event_index(data) -> dict:
    """event_id -> (event dict, sorted customer ids, recipient rows sorted by customer_id)."""
    events = {e["id"]: e for e in events_for(data)}
    df = _with_window(data.schedule).copy()
    df["customer_id"] = df["customer_id"].astype(str)
    out = {}
    for (start, wave, window), g in df.groupby(["send_datetime", "wave", "send_window"], sort=True):
        g = g.sort_values("customer_id").reindex(columns=_RECIPIENT_COLS)
        eid = event_id(data.month, start, wave, window)
        out[eid] = (events[eid], g["customer_id"].tolist(), g.to_dict("records"))
    return out


@router.get("/plans/{month}/events/{event_id}/customers", responses={409: {"description": docs.ERR_409}},
            **docs.EVENT_CUSTOMERS)
def event_customers(request: Request, event_id: str = Path(..., description=docs.P_EVENT_ID),
                    month: str = Depends(valid_month),
                    cursor: str | None = Query(None, description=docs.P_CURSOR),
                    limit: int = Query(100, ge=1, le=1000, description=docs.P_LIMIT)):
    data = load_month(request.app.state.settings.data_dir, month)
    entry = data.memo("event_index", lambda: _build_event_index(data)).get(event_id)
    if entry is None:
        raise ApiError(404, "event_not_found", "ไม่พบนัดส่งนี้ในแผนของเดือนนั้น")
    event, ids, rows = entry
    start = 0
    if cursor:
        c = decode_cursor(cursor)
        if c["e"] != event_id:
            raise ApiError(422, "invalid_cursor", "cursor นี้เป็นของนัดส่งอื่น")
        if c["t"] != data.etag:
            raise ApiError(409, "plan_changed", docs.ERR_409)
        start = bisect.bisect_right(ids, c["a"])
    page = rows[start:start + limit]
    more = start + limit < len(rows)
    body = {
        "month": month, "event_id": event_id, "start": event["start"], "wave": event["wave"],
        "send_window": event["send_window"], "customer_count": event["customer_count"],
        "customers": page,
        "next_cursor": encode_cursor(event_id, data.etag, page[-1]["customer_id"]) if more and page else None,
    }
    return respond(request, data, body)
