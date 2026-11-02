from __future__ import annotations

import re

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def valid_month(month: str) -> str:
    """Path dependency: only YYYY-MM is accepted, so a month can never reach the filesystem as a path."""
    if not _MONTH_RE.fullmatch(month):
        raise ApiError(422, "invalid_month", "month must look like YYYY-MM (e.g. 2026-12)")
    return month


def _body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content=_body(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return JSONResponse(status_code=422, content=_body("validation_error", msg))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content=_body(f"http_{exc.status_code}", str(exc.detail)),
                            headers=getattr(exc, "headers", None))
