from __future__ import annotations

import math
import re
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response

from . import docs
from .errors import ApiError, valid_month
from .loaders import MonthData, load_month

router = APIRouter()

WAVE_NAMES = {1: "off_peak_traffic", 2: "peak_upsell", 3: "vip_early"}
DEFAULT_WINDOW = "Peak Window"


def clean(obj):
    """Make an object JSON-safe: NaN/inf → None, numpy scalars → python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is pd.NaT or obj is pd.NA:
        return None
    return obj


def respond(request: Request, data: MonthData, body: dict) -> Response:
    headers = {"ETag": data.etag}
    if request.headers.get("If-None-Match") == data.etag:
        return Response(status_code=304, headers=headers)
    return JSONResponse(content=clean(body), headers=headers)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _with_window(schedule: pd.DataFrame) -> pd.DataFrame:
    df = schedule.copy()
    if "send_window" not in df.columns:
        df["send_window"] = DEFAULT_WINDOW
    df["send_window"] = df["send_window"].fillna(DEFAULT_WINDOW)
    if "channel" not in df.columns:
        df["channel"] = None
    if "expected_profit" not in df.columns:
        df["expected_profit"] = 0.0
    return df


def build_events(month: str, schedule: pd.DataFrame) -> list[dict]:
    """One event per (send_datetime, wave, send_window); ids depend only on those fields."""
    if schedule.empty:
        return []
    df = _with_window(schedule)
    events = []
    for (start, wave, window), g in df.groupby(["send_datetime", "wave", "send_window"], sort=True):
        actions = g["recommended_action"].value_counts()
        events.append({
            "id": f"{month}|{start:%Y-%m-%dT%H:%M}|w{int(wave)}|{_slug(window)}",
            "start": start.isoformat(),
            "wave": int(wave),
            "send_window": window,
            "channel": g["channel"].dropna().iloc[0] if g["channel"].notna().any() else None,
            "customer_count": int(len(g)),
            "actions": [{"name": n, "count": int(c)} for n, c in actions.items()],
            "expected_profit": round(float(g["expected_profit"].sum()), 2),
        })
    return events


def events_for(data: MonthData) -> list[dict]:
    """All events of a loaded month, computed once (callers must not mutate the returned list)."""
    return data.memo("events", lambda: build_events(data.month, data.schedule))


@router.get("/plans/{month}", **docs.PLAN)
def plan_summary(request: Request, month: str = Depends(valid_month)):
    data = load_month(request.app.state.settings.data_dir, month)
    s = data.schedule
    profit = round(float(pd.to_numeric(s.get("expected_profit"), errors="coerce").sum()), 2) if len(s) else 0.0
    n = int(len(s))
    weeks = (data.segments or {}).get("wave_weeks") or {}
    waves = [{"wave": w, "name": WAVE_NAMES[w], "count": int((s["wave"] == w).sum()),
              "weeks": weeks.get(str(w))} for w in (1, 2, 3)]
    body = {
        "month": month, "generated_at": data.generated_at.isoformat(), "timezone": "Asia/Bangkok",
        "gate_status": (data.segments or {}).get("gate_status"), "currency": "THB",
        "totals": {"scheduled": n, "holdout": int(len(data.holdout)), "expected_profit": profit,
                   "avg_profit_per_customer": round(profit / n, 2) if n else 0.0},
        "peak_hours": data.summary.get("peak_hours_detected"),
        "off_peak_hours": data.summary.get("off_peak_hours_detected"),
        "waves": waves,
    }
    return respond(request, data, body)


@router.get("/plans/{month}/events", **docs.EVENTS)
def plan_events(request: Request, month: str = Depends(valid_month),
                from_: date | None = Query(None, alias="from", description=docs.P_FROM),
                to: date | None = Query(None, description=docs.P_TO),
                wave: int | None = Query(None, ge=1, le=3, description=docs.P_WAVE)):
    if from_ and to and from_ > to:
        raise ApiError(422, "invalid_range", "'from' must not be after 'to'")
    data = load_month(request.app.state.settings.data_dir, month)
    events = events_for(data)
    if from_:
        events = [e for e in events if e["start"][:10] >= from_.isoformat()]
    if to:
        events = [e for e in events if e["start"][:10] <= to.isoformat()]
    if wave:
        events = [e for e in events if e["wave"] == wave]
    return respond(request, data, {"month": month, "events": events})
