from __future__ import annotations

from fastapi import Depends, FastAPI

from .auth import require_key
from .errors import install_handlers
from .loaders import list_months
from . import catalog, ics, plans
from .settings import Settings


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
    app.include_router(ics.router, prefix="/v1")  # own auth: key OR calendar token
    return app


def app_factory() -> FastAPI:
    """uvicorn entry: `uvicorn crm_api.app:app_factory --factory --app-dir src`."""
    return create_app()
