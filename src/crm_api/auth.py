from __future__ import annotations

import hmac

from fastapi import Request, Security
from fastapi.security import APIKeyHeader, APIKeyQuery

from .errors import ApiError

# Declared as FastAPI security schemes so /docs shows an "Authorize" button and /openapi.json documents them.
# auto_error=False: we keep our own uniform 401 body.
_header_key = APIKeyHeader(name="X-API-Key", auto_error=False, description="API key (env CRM_API_KEY)")
_query_token = APIKeyQuery(name="token", auto_error=False,
                           description="Read-only calendar token (env CRM_API_CALENDAR_TOKEN); calendar.ics only")


def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def require_key(request: Request, api_key: str | None = Security(_header_key)) -> None:
    if not _eq(api_key or "", request.app.state.settings.api_key):
        raise ApiError(401, "unauthorized", "missing or invalid X-API-Key")


def require_key_or_calendar_token(request: Request, api_key: str | None = Security(_header_key),
                                  token: str | None = Security(_query_token)) -> None:
    """Calendar apps cannot send headers: accept the API key OR the separate read-only ?token=."""
    settings = request.app.state.settings
    if _eq(api_key or "", settings.api_key):
        return
    if settings.calendar_token and _eq(token or "", settings.calendar_token):
        return
    raise ApiError(401, "unauthorized", "missing or invalid X-API-Key / calendar token")
