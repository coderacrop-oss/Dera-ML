"""Operator routes for approving a model candidate. Auth is the per-candidate token printed by the training
command (browsers following a link cannot send X-API-Key). Hidden from the published OpenAPI contract."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from candidates import candidate_model_bytes, check_token, load_candidate, mark_candidate
from model_registry import ModelRegistry, RegistryError, StaleCandidate
from pipeline_logger import log

from .errors import ApiError
from .model_page import render_candidate_page

router = APIRouter(prefix="/models/candidates", include_in_schema=False)


class Decision(BaseModel):
    token: str
    approver: str = Field(min_length=1, max_length=80)


def _root(request: Request) -> str:
    return str(request.app.state.settings.models_dir)


def _load(request: Request, candidate_id: str, token: str | None) -> dict:
    root = _root(request)
    try:
        cand = load_candidate(root, candidate_id)
    except KeyError:
        raise ApiError(404, "candidate_not_found", "candidate not found")
    if not token or not check_token(root, candidate_id, token):
        raise ApiError(401, "unauthorized", "missing or invalid candidate token")
    return cand


@router.get("/{candidate_id}", response_class=HTMLResponse)
def page(request: Request, candidate_id: str, token: str | None = Query(None)):
    cand = _load(request, candidate_id, token)
    current = ModelRegistry(_root(request)).current()
    return render_candidate_page(candidate_id, cand["report"], cand["status"],
                                 current.version if current else None, token)


def _pending(cand: dict) -> None:
    if cand["status"]["status"] != "pending":
        raise ApiError(409, "candidate_not_pending", f"candidate is already {cand['status']['status']}")


@router.post("/{candidate_id}/approve")
def approve(request: Request, candidate_id: str, body: Decision):
    cand = _load(request, candidate_id, body.token)
    _pending(cand)
    root, report = _root(request), cand["report"]
    meta = {"approved_by": body.approver, "approved_at": datetime.now(timezone.utc).isoformat(),
            "provenance": report["provenance"], "created_at": report["created_at"],
            "metrics": {"challenger": report["challenger"], "champion": report["champion"]}}
    try:
        version = ModelRegistry(root).promote(candidate_model_bytes(root, candidate_id),
                                              report["compared_against_version"], meta)
    except StaleCandidate as exc:
        raise ApiError(409, "candidate_stale", str(exc))
    try:
        mark_candidate(root, candidate_id, "approved", body.approver)
    except RegistryError:
        pass  # decided concurrently; the champion change above is already recorded in the manifest
    log("L4", "champion_approved", candidate_id=candidate_id, version=version, approver=body.approver,
        compared_against=report["compared_against_version"])
    return {"status": "approved", "version": version}


@router.post("/{candidate_id}/reject")
def reject(request: Request, candidate_id: str, body: Decision):
    cand = _load(request, candidate_id, body.token)
    _pending(cand)
    try:
        mark_candidate(_root(request), candidate_id, "rejected", body.approver)
    except RegistryError as exc:
        raise ApiError(409, "candidate_not_pending", str(exc))
    log("L4", "candidate_rejected", candidate_id=candidate_id, approver=body.approver)
    return {"status": "rejected"}
