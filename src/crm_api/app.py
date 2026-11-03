from __future__ import annotations

from fastapi import Depends, FastAPI

from . import docs
from .auth import require_key
from .errors import install_handlers
from .loaders import list_months
from . import catalog, ics, models, plans, segmentation
from .settings import Settings


ERRORS = {code: {"description": text} for code, text in docs.ERRORS_TH.items()}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title=docs.APP_TITLE, version="1.0.0", description=docs.APP_DESCRIPTION,
                  openapi_tags=docs.TAGS)
    app.state.settings = settings
    install_handlers(app)

    @app.get("/health", **docs.HEALTH)
    def health():
        return {"status": "ok", "months_available": list_months(settings.data_dir)}

    @app.get("/v1/months", dependencies=[Depends(require_key)], responses={401: ERRORS[401]}, **docs.MONTHS)
    def months():
        return {"months": list_months(settings.data_dir)}

    for r in (plans.router, catalog.router, segmentation.router):
        app.include_router(r, prefix="/v1", dependencies=[Depends(require_key)], responses=ERRORS)
    app.include_router(ics.router, prefix="/v1", responses=ERRORS)  # own auth: key OR calendar token
    app.include_router(models.router, prefix="/v1")  # operator tool: own auth (per-candidate token), hidden from the contract
    return app


def app_factory() -> FastAPI:
    """uvicorn entry: `uvicorn crm_api.app:app_factory --factory --app-dir src`."""
    return create_app()
