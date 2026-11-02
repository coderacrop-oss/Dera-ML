from __future__ import annotations

import hmac

from fastapi import Request

from .errors import ApiError


def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def require_key(request: Request) -> None:
    if not _eq(request.headers.get("X-API-Key", ""), request.app.state.settings.api_key):
        raise ApiError(401, "unauthorized", "missing or invalid X-API-Key")


def require_key_or_calendar_token(request: Request) -> None:
    """Calendar apps cannot send headers: accept the API key OR the separate read-only ?token=."""
    settings = request.app.state.settings
    if _eq(request.headers.get("X-API-Key", ""), settings.api_key):
        return
    token = request.query_params.get("token", "")
    if settings.calendar_token and _eq(token, settings.calendar_token):
        return
    raise ApiError(401, "unauthorized", "missing or invalid X-API-Key / calendar token")
