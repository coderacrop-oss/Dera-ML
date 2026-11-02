from __future__ import annotations

import hmac

from fastapi import Depends, FastAPI, Request

from .errors import ApiError, install_handlers
from .loaders import list_months
from . import catalog, plans
from .settings import Settings


def require_key(request: Request) -> None:
    given = request.headers.get("X-API-Key", "")
    expected = request.app.state.settings.api_key
    if not hmac.compare_digest(given.encode(), expected.encode()):
        raise ApiError(401, "unauthorized", "missing or invalid X-API-Key")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="CRM Campaign API", version="1.0.0")
    app.state.settings = settings
    install_handlers(app)

    @app.get("/health")
    def health():
        return {"status": "ok", "months_available": list_months(settings.data_dir)}

    @app.get("/v1/months", dependencies=[Depends(require_key)])
    def months():
        return {"months": list_months(settings.data_dir)}

    for r in (plans.router, catalog.router):
        app.include_router(r, prefix="/v1", dependencies=[Depends(require_key)])
    return app


def app_factory() -> FastAPI:
    """uvicorn entry: `uvicorn crm_api.app:app_factory --factory --app-dir src`."""
    return create_app()
