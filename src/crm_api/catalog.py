from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, Request

from .errors import ApiError, valid_month
from .loaders import load_month
from .plans import _with_window, respond

router = APIRouter()


@router.get("/plans/{month}/segments")
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


@router.get("/plans/{month}/promotions")
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


@router.get("/plans/{month}/customers/{customer_id}")
def customer_offer(request: Request, customer_id: str, month: str = Depends(valid_month)):
    data = load_month(request.app.state.settings.data_dir, month)
    row = _index(data).get(customer_id)
    if row is None:  # holdout, suppressed and unknown customers all look the same on purpose
        raise ApiError(404, "customer_not_found", "no scheduled offer for this customer in this plan")
    body = {k: row.get(k) for k in ("customer_id", "wave", "recommended_action", "offer", "message",
                                    "channel", "send_window")}
    body["send_datetime"] = row["send_datetime"].isoformat()
    return respond(request, data, body)
