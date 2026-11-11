"""campaign.py — Campaign, audience, content, and model endpoints.

Endpoints
---------
POST /campaign/plan           — run ML pipeline → campaign plan
POST /campaign/analyze        — sanity-check a manually drafted plan
GET  /campaign/history        — list campaign history for a store
POST /campaign/record-outcome — record actual results for calibration
POST /campaign/execute        — send campaign via LINE OA or Meta Ads
POST /audience/export         — export targeting audiences
POST /content/generate        — generate AI content
POST /model/retrain           — trigger model retraining (async)
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from .auth import require_key
from .errors import ApiError
from . import store as _store
from . import docs

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plan_id() -> str:
    return f"plan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _analysis_id() -> str:
    return f"analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _export_id() -> str:
    return f"export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _decode_b64(encoded: str, label: str) -> bytes:
    try:
        return base64.b64decode(encoded)
    except Exception:
        raise ApiError(400, "invalid_base64", f"{label} ต้องเป็น base64-encoded string")


def _add_src_to_path() -> None:
    src = str(Path(__file__).parent.parent)
    if src not in sys.path:
        sys.path.insert(0, src)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PlanRequest(BaseModel):
    store_id:      str
    pos_data:      str                       # base64 CSV
    mode:          str = "budget"            # budget | target_profit
    budget:        float | None = None
    target_profit: float | None = None
    goal:          str = "maximize_profit"
    # optional per-shop loyalty point cost (omit -> ML uses its 2%-of-basket estimate, cost_source "estimated")
    point_value_thb: float | None = Field(None, gt=0)       # baht one point costs the shop
    redemption_rate: float | None = Field(None, gt=0, le=1)  # share of awarded points redeemed

    @model_validator(mode="after")
    def _rate_needs_value(self):
        if self.redemption_rate is not None and self.point_value_thb is None:
            raise ValueError("redemption_rate ต้องระบุคู่กับ point_value_thb")
        return self


class AnalyzeRequest(BaseModel):
    store_id: str
    pos_data: str
    proposed_plan: dict[str, Any]
    campaign_history: list[dict] = Field(default_factory=list)


class RecordOutcomeRequest(BaseModel):
    plan_id:                     str
    actual_revenue:              float
    actual_customers_responded:  int
    measurement_period_days:     int = 30


class AudienceExportRequest(BaseModel):
    plan_id:  str
    store_id: str
    crm_data: str | None = None          # base64 CSV, optional


class ContentRequest(BaseModel):
    brief_type: str            # ml_driven | brand_driven
    brief:      dict[str, Any]
    brand_kit:  dict[str, Any]


class RetrainRequest(BaseModel):
    store_id: str
    pos_file: str | None = None   # path on server, optional
    psm_mode: str = "discount"    # discount | campaign_list | proxy


class RetrainRequestBody(BaseModel):
    store_id:     str
    reason:       str = "manual_request"  # manual_request | accuracy_concern | new_data
    psm_mode:     str = "auto"            # auto | discount | campaign_list | proxy
    requested_by: str = ""               # optional identifier (DotDash user, etc.)
    pos_data:     str | None = None      # base64-encoded POS CSV — stored for scheduler


class ExecuteRequest(BaseModel):
    plan_id:       str
    channel:       str                   # "line_oa" | "meta"
    content_id:    str | None = None     # optional — content to include
    ad_budget_thb: float | None = None   # Meta only
    message_text:  str | None = None     # LINE only — overrides default


# ---------------------------------------------------------------------------
# POST /campaign/plan
# ---------------------------------------------------------------------------

@router.post("/campaign/plan", **docs.CAMPAIGN_PLAN)
def campaign_plan(body: PlanRequest, request: Request, _: None = Depends(require_key)):
    """Run the ML pipeline and return a structured campaign plan."""
    _add_src_to_path()

    # Validate mode / budget
    if body.mode == "budget" and body.budget is None:
        raise ApiError(400, "invalid_input", "mode=budget ต้องระบุ budget")
    if body.mode == "target_profit" and body.target_profit is None:
        raise ApiError(400, "invalid_input", "mode=target_profit ต้องระบุ target_profit")

    # Decode and persist POS file
    pos_bytes = _decode_b64(body.pos_data, "pos_data")
    data_dir  = str(request.app.state.settings.data_dir)
    pos_path  = _store.save_pos_file(data_dir, body.store_id, pos_bytes)

    # Run campaign planner
    try:
        from campaign_planner import run as plan_run
        report = plan_run(
            budget=body.budget if body.mode == "budget" else None,
            target_profit=body.target_profit if body.mode == "target_profit" else None,
            goal=body.goal,
            pos_file=pos_path,
            point_value_thb=body.point_value_thb,
            redemption_rate=body.redemption_rate,
        )
    except ValueError as exc:
        raise ApiError(400, "invalid_input", str(exc))
    except Exception as exc:
        raise ApiError(500, "pipeline_error", f"Pipeline failed: {type(exc).__name__}: {exc}")

    plan_id = _plan_id()
    generated_at = _now_iso()

    response = {
        "plan_id":       plan_id,
        "store_id":      body.store_id,
        "generated_at":  generated_at,
        "mode":          report.get("mode"),
        "goal":          report.get("goal"),
        "goal_label":    report.get("goal_label"),
        "summary": {
            "total_customers":           report.get("total_customers"),
            "viable_after_guardrail":    report.get("viable_after_guardrail"),
            "suppressed_by_guardrail":   report.get("suppressed_by_guardrail"),
            "goal_filtered_to":          report.get("goal_filtered_to"),
            "customers_selected":        report.get("customers_selected"),
            "actual_spend":              report.get("actual_spend"),
            "expected_incremental_profit": report.get("expected_incremental_profit"),
            "roi":                       report.get("roi"),
            "budget_utilisation_pct":    report.get("budget_utilisation_pct"),
        },
        "action_breakdown":   report.get("action_breakdown", {}),
        "segment_breakdown":  report.get("segment_breakdown", {}),
        "dotdash_segments":   report.get("dotdash_segments", []),
        "feasibility": {
            "target_achievable":     report.get("target_achievable"),
            "profit_gap":            report.get("profit_gap"),
            "max_achievable_profit": report.get("max_achievable_profit"),
            "max_budget_needed":     report.get("max_budget_needed"),
        },
    }

    # Persist to history
    _store.append_plan(data_dir, {
        "plan_id":      plan_id,
        "store_id":     body.store_id,
        "generated_at": generated_at,
        "mode":         body.mode,
        "goal":         body.goal,
        "budget_input": body.budget,
        "target_profit_input": body.target_profit,
        "customers_selected": report.get("customers_selected"),
        "actual_spend":       report.get("actual_spend"),
        "expected_profit":    report.get("expected_incremental_profit"),
        "roi":                report.get("roi"),
        "status":             "plan_only",
    })

    return response


# ---------------------------------------------------------------------------
# POST /campaign/analyze
# ---------------------------------------------------------------------------

@router.post("/campaign/analyze", **docs.CAMPAIGN_ANALYZE)
def campaign_analyze(body: AnalyzeRequest, request: Request, _: None = Depends(require_key)):
    """Sanity-check a manually drafted plan — 4 advisory checks."""
    _add_src_to_path()

    plan = body.proposed_plan
    valid_goals = {"maximize_profit", "win_back", "retain_loyal", "maximize_reach"}
    if plan.get("goal") not in valid_goals:
        raise ApiError(400, "invalid_input",
                       f"proposed_plan.goal ต้องเป็น {' / '.join(valid_goals)}")

    # Decode and save POS file
    pos_bytes = _decode_b64(body.pos_data, "pos_data")
    data_dir  = str(request.app.state.settings.data_dir)
    pos_path  = _store.save_pos_file(data_dir, body.store_id, pos_bytes)

    # Run a quick campaign plan to get baseline profit estimates
    try:
        from campaign_planner import run as plan_run
        mode = plan.get("mode", "budget")
        report = plan_run(
            budget=plan.get("budget") if mode == "budget" else None,
            target_profit=plan.get("target_profit") if mode == "target_profit" else None,
            goal=plan.get("goal", "maximize_profit"),
            pos_file=pos_path,
        )
    except Exception as exc:
        raise ApiError(500, "pipeline_error", f"Pipeline failed: {exc}")

    est_profit = report.get("expected_incremental_profit", 0) or 0
    est_spend  = report.get("actual_spend", 0) or 0
    net        = round(est_profit - est_spend, 2)

    # Check 1: Profitability
    if net < 0:
        prof_status = "warning"
        prof_msg    = f"แผนนี้จะขาดทุน ฿{abs(net):,.0f} — ลดส่วนลดหรือเลือก segment ที่มี uplift สูงกว่า"
    elif net < est_spend * 0.5:
        prof_status = "warning"
        prof_msg    = f"ROI ต่ำ — ได้กำไรสุทธิ ฿{net:,.0f} จากงบ ฿{est_spend:,.0f}"
    else:
        prof_status = "ok"
        prof_msg    = f"คาดกำไรสุทธิ ฿{net:,.0f} — คุ้มค่าทำ"

    # Check 2: Overlap (customers recently promoted)
    overlap_count = 0
    min_days      = None
    if body.campaign_history:
        from datetime import timedelta
        scheduled = plan.get("scheduled_date")
        ref_date  = datetime.fromisoformat(scheduled) if scheduled else datetime.now(timezone.utc)
        for entry in body.campaign_history:
            last_str = entry.get("last_promo_date", "")
            if not last_str:
                continue
            try:
                last_dt = datetime.fromisoformat(last_str)
                days_ago = (ref_date - last_dt).days
                if days_ago < 21:
                    overlap_count += 1
                    min_days = min(min_days or days_ago, days_ago)
            except ValueError:
                continue

    if overlap_count > 0:
        overlap_status = "warning"
        overlap_msg    = (f"ลูกค้า {overlap_count} คนเพิ่งได้รับ promo เมื่อน้อยกว่า 21 วัน "
                          f"(ครั้งล่าสุด {min_days} วันที่แล้ว)")
    else:
        overlap_status = "ok"
        overlap_msg    = "ไม่มี overlap — ลูกค้าทุกคนพร้อมรับ promo"

    # Check 3: Segment readiness
    segs         = report.get("segment_breakdown", {})
    target_segs  = plan.get("segments", [])
    if target_segs:
        seg_counts   = {s: segs.get(s, 0) for s in target_segs}
        total_target = sum(seg_counts.values())
        seg_status   = "ok" if total_target >= 10 else "warning"
        seg_msg      = (f"รวม {total_target} คนใน segment ที่เลือก"
                        if total_target >= 10
                        else f"ลูกค้าใน segment เลือกมีน้อยมาก ({total_target} คน)")
    else:
        total_target = report.get("customers_selected", 0)
        seg_status   = "ok"
        seg_msg      = f"ML เลือก {total_target} คน — segment พร้อมทำ campaign"

    # Check 4: Budget efficiency
    max_profit  = report.get("max_achievable_profit", est_profit) or est_profit
    gain        = round(max_profit - est_profit, 2)
    if gain > est_profit * 0.3:
        eff_status = "suggestion"
        eff_msg    = (f"ถ้า allocate งบตาม uplift จริงแทนแบ่งเท่ากัน "
                      f"profit เพิ่มขึ้นได้ถึง ฿{gain:,.0f}")
    else:
        eff_status = "ok"
        eff_msg    = "การ allocate งบอยู่ในเกณฑ์ดี"

    # Overall verdict
    warnings = sum(1 for s in [prof_status, overlap_status, seg_status] if s == "warning")
    if warnings >= 2:
        verdict = "not_recommended"
    elif warnings == 1:
        verdict = "proceed_with_caution"
    else:
        verdict = "recommended"

    return {
        "analysis_id":   _analysis_id(),
        "store_id":      body.store_id,
        "analyzed_at":   _now_iso(),
        "proposed_plan": plan,
        "checks": {
            "profitability": {
                "status":           prof_status,
                "message":          prof_msg,
                "estimated_cost":   est_spend,
                "estimated_profit": est_profit,
                "net":              net,
            },
            "overlap": {
                "status":                 overlap_status,
                "message":                overlap_msg,
                "overlap_count":          overlap_count,
                "min_days_since_last_promo": min_days,
            },
            "segment_readiness": {
                "status":  seg_status,
                "message": seg_msg,
            },
            "budget_efficiency": {
                "status":                    eff_status,
                "message":                   eff_msg,
                "current_estimated_profit":  est_profit,
                "optimized_estimated_profit": max_profit,
                "gain":                      gain,
            },
        },
        "overall_verdict": verdict,
        "can_proceed":     True,
    }


# ---------------------------------------------------------------------------
# GET /campaign/history
# ---------------------------------------------------------------------------

@router.get("/campaign/history", **docs.CAMPAIGN_HISTORY)
def campaign_history(
    request: Request,
    store_id: str = Query(..., description=docs.P_STORE_ID),
    limit:    int = Query(20, ge=1, le=100, description=docs.P_LIMIT_H),
    offset:   int = Query(0, ge=0, description=docs.P_OFFSET),
    _: None = Depends(require_key),
):
    data_dir = str(request.app.state.settings.data_dir)
    plans    = _store.get_plans(data_dir, store_id, limit=limit, offset=offset)
    total    = len(_store.get_plans(data_dir, store_id, limit=10_000))

    # Build promo_log from plan history
    promo_log: list[dict] = []
    for p in plans:
        if p.get("status") in ("active", "awaiting_outcome", "completed"):
            # We don't have per-customer logs yet — placeholder
            pass

    return {
        "store_id":  store_id,
        "total":     total,
        "campaigns": plans,
        "promo_log": promo_log,
    }


# ---------------------------------------------------------------------------
# POST /campaign/record-outcome
# ---------------------------------------------------------------------------

@router.post("/campaign/record-outcome", **docs.CAMPAIGN_RECORD_OUTCOME)
def record_outcome(body: RecordOutcomeRequest, request: Request, _: None = Depends(require_key)):
    """Record actual campaign results for model calibration."""
    _add_src_to_path()
    data_dir = str(request.app.state.settings.data_dir)

    # Retrieve original plan for predicted profit
    plan = _store.get_plan_by_id(data_dir, body.plan_id)
    predicted_profit = plan.get("expected_profit", 0) if plan else 0

    calibration_ratio = (
        round(body.actual_revenue / predicted_profit, 4)
        if predicted_profit and predicted_profit > 0
        else None
    )

    outcome = {
        "plan_id":                    body.plan_id,
        "actual_revenue":             body.actual_revenue,
        "actual_customers_responded": body.actual_customers_responded,
        "measurement_period_days":    body.measurement_period_days,
        "predicted_profit":           predicted_profit,
        "calibration_ratio":          calibration_ratio,
        "recorded_at":                _now_iso(),
    }
    _store.append_outcome(data_dir, outcome)

    # Try to get calibration stats
    try:
        from promo_advisor import get_calibration_stats
        outcomes_path = os.path.join(data_dir, "outcomes.jsonl")
        if os.path.exists(outcomes_path):
            stats = get_calibration_stats(outcomes_path)
        else:
            stats = {"trend": "insufficient_data"}
    except Exception:
        stats = {"trend": "unavailable"}

    return {
        "plan_id":           body.plan_id,
        "outcome_recorded":  True,
        "calibration_ratio": calibration_ratio,
        "calibration_stats": stats,
    }


# ---------------------------------------------------------------------------
# POST /audience/export
# ---------------------------------------------------------------------------

@router.post("/audience/export", **docs.AUDIENCE_EXPORT)
def audience_export(body: AudienceExportRequest, request: Request, _: None = Depends(require_key)):
    """Export targeting audiences for LINE OA and Meta."""
    _add_src_to_path()
    data_dir = str(request.app.state.settings.data_dir)

    # Retrieve plan for segment breakdown
    plan = _store.get_plan_by_id(data_dir, body.plan_id)
    if plan is None:
        raise ApiError(404, "plan_not_found", f"ไม่พบ plan_id: {body.plan_id}")

    # Optional CRM data for LINE UID matching
    crm_path = None
    if body.crm_data:
        crm_bytes = _decode_b64(body.crm_data, "crm_data")
        pos_dir   = os.path.join(data_dir, "stores", body.store_id)
        os.makedirs(pos_dir, exist_ok=True)
        crm_path  = os.path.join(pos_dir, "crm_data.csv")
        with open(crm_path, "wb") as f:
            f.write(crm_bytes)

    # Build export using ad_targeting_export if POS file is available
    pos_path = _store.get_pos_path(data_dir, body.store_id)
    export_id = _export_id()
    generated_at = _now_iso()

    summary = {"total_customers": plan.get("customers_selected", 0)}
    audiences: dict = {"line_oa": {"send_list": [], "count": 0}, "meta": {}}

    if pos_path:
        try:
            from ad_targeting_export import audience_export as _do_export
            export_dir = os.path.join(data_dir, "exports", export_id)
            os.makedirs(export_dir, exist_ok=True)
            result = _do_export(pos_path, output_dir=export_dir, crm_path=crm_path)
            if result:
                summary.update(result.get("summary", {}))
                audiences = result.get("audiences", audiences)
        except Exception:
            pass   # fallback: return empty audiences

    return {
        "export_id":    export_id,
        "plan_id":      body.plan_id,
        "store_id":     body.store_id,
        "generated_at": generated_at,
        "summary":      summary,
        "audiences":    audiences,
    }


# ---------------------------------------------------------------------------
# POST /content/generate
# ---------------------------------------------------------------------------

@router.post("/content/generate", **docs.CONTENT_GENERATE)
def content_generate(body: ContentRequest, request: Request, _: None = Depends(require_key)):
    """Generate AI-driven campaign content via Claude API."""
    _add_src_to_path()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ApiError(503, "api_key_missing",
                       "ANTHROPIC_API_KEY ไม่ได้ตั้งค่า กรุณาตั้งค่า environment variable")

    try:
        from content_generator import generate_content, BrandKit, MLDrivenBrief, BrandDrivenBrief

        bk = BrandKit(**body.brand_kit)

        if body.brief_type == "ml_driven":
            brief = MLDrivenBrief(**body.brief)
        elif body.brief_type == "brand_driven":
            brief = BrandDrivenBrief(**body.brief)
        else:
            raise ApiError(400, "invalid_brief_type",
                           "brief_type ต้องเป็น 'ml_driven' หรือ 'brand_driven'")

        output = generate_content(brief, bk, api_key=api_key)

    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(500, "content_error", f"Content generation failed: {exc}")

    return {
        "headline":        output.headline,
        "body":            output.body,
        "cta":             output.cta,
        "image_prompt":    output.image_prompt,
        "meta_primary_text": output.meta_primary_text,
        "strategy": {
            "key_message": output.key_message,
            "emotion":     output.emotion,
            "tone_used":   output.tone_used,
        },
        "art_direction": {
            "visual_concept": output.visual_concept,
            "composition":    output.composition,
            "mood":           output.mood,
            "color_direction": output.color_direction,
            "hero_subject":   output.hero_subject,
        },
    }


# ---------------------------------------------------------------------------
# POST /model/retrain
# ---------------------------------------------------------------------------

@router.post("/model/retrain", **docs.MODEL_RETRAIN)
def model_retrain(
    body: RetrainRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    _: None = Depends(require_key),
):
    """Trigger model retraining as a background job."""
    job_id = f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def _run_retrain():
        _add_src_to_path()
        try:
            from retrain_trigger import assess_retrain
            assess_retrain()
        except Exception:
            pass

    background_tasks.add_task(_run_retrain)

    return {
        "status":   "queued",
        "job_id":   job_id,
        "store_id": body.store_id,
        "queued_at": _now_iso(),
        "message":  "Retraining job queued — ใช้ GET /model/status เพื่อตรวจสอบสถานะ",
    }


# ---------------------------------------------------------------------------
# POST /model/retrain-request
# ---------------------------------------------------------------------------

_VALID_REASONS  = {"manual_request", "accuracy_concern", "new_data"}
_VALID_PSM_MODES = {"auto", "discount", "campaign_list", "proxy"}


@router.post("/model/retrain-request", **docs.MODEL_RETRAIN_REQUEST)
def model_retrain_request(
    body: RetrainRequestBody,
    request: Request,
    _: None = Depends(require_key),
):
    """Queue a retrain request — scheduler picks it up in the next nightly run."""
    if body.reason not in _VALID_REASONS:
        raise ApiError(400, "invalid_input",
                       f"reason ต้องเป็นหนึ่งใน: {', '.join(sorted(_VALID_REASONS))}")

    if body.psm_mode not in _VALID_PSM_MODES:
        raise ApiError(400, "invalid_input",
                       f"psm_mode ต้องเป็นหนึ่งใน: {', '.join(sorted(_VALID_PSM_MODES))}")

    data_dir   = str(request.app.state.settings.data_dir)
    request_id = f"retrain_req_{uuid.uuid4().hex[:12]}"
    queued_at  = _now_iso()

    # Optionally save POS data for scheduler to pick up
    pos_saved_path: str | None = None
    if body.pos_data:
        try:
            pos_bytes = _decode_b64(body.pos_data, "pos_data")
            safe_store = "".join(c for c in body.store_id if c.isalnum() or c in ("-", "_"))
            pos_dir = os.path.join(data_dir, "stores", safe_store)
            os.makedirs(pos_dir, exist_ok=True)
            pos_saved_path = os.path.join(pos_dir, "latest_pos.csv")
            with open(pos_saved_path, "wb") as _f:
                _f.write(pos_bytes)
        except ApiError:
            raise
        except Exception:
            pos_saved_path = None  # non-fatal — still queue the request

    record = {
        "request_id":    request_id,
        "store_id":      body.store_id,
        "reason":        body.reason,
        "psm_mode":      body.psm_mode,
        "requested_by":  body.requested_by,
        "queued_at":     queued_at,
        "status":        "queued",
        "pos_available": pos_saved_path is not None,
    }
    _store.append_retrain_request(data_dir, record)

    return {
        "request_id":           request_id,
        "store_id":             body.store_id,
        "queued_at":            queued_at,
        "status":               "queued",
        "reason":               body.reason,
        "psm_mode":             body.psm_mode,
        "pos_received":         pos_saved_path is not None,
        "estimated_completion": "ภายใน 24 ชั่วโมง — scheduler รันกลางคืน",
    }


# ---------------------------------------------------------------------------
# GET /model/retrain-request/{request_id}
# ---------------------------------------------------------------------------

@router.get("/model/retrain-request/{request_id}", **docs.MODEL_RETRAIN_REQUEST_STATUS)
def model_retrain_request_status(
    request: Request,
    request_id: str = Path(..., description=docs.P_REQUEST_ID),
    _: None = Depends(require_key),
):
    """Return current status of a retrain request queued via POST /model/retrain-request."""
    data_dir = str(request.app.state.settings.data_dir)
    record = _store.get_retrain_request_by_id(data_dir, request_id)
    if record is None:
        raise ApiError(404, "request_not_found", f"ไม่พบ request_id: {request_id}")
    return {
        "request_id":    record.get("request_id"),
        "store_id":      record.get("store_id"),
        "status":        record.get("status", "queued"),
        "reason":        record.get("reason"),
        "psm_mode":      record.get("psm_mode"),
        "queued_at":     record.get("queued_at"),
        "processing_at": record.get("processing_at"),
        "completed_at":  record.get("completed_at"),
        "failed_at":     record.get("failed_at"),
        "error":         record.get("error"),
        "pos_available": record.get("pos_available", False),
    }


# ---------------------------------------------------------------------------
# POST /campaign/execute
# ---------------------------------------------------------------------------

@router.post("/campaign/execute", **docs.CAMPAIGN_EXECUTE)
def campaign_execute(body: ExecuteRequest, request: Request, _: None = Depends(require_key)):
    """Execute a campaign plan via LINE OA or Meta Ads."""
    _add_src_to_path()
    data_dir = str(request.app.state.settings.data_dir)

    # Validate channel
    if body.channel not in ("line_oa", "meta"):
        raise ApiError(400, "invalid_channel", "channel ต้องเป็น 'line_oa' หรือ 'meta'")

    # Retrieve plan
    plan = _store.get_plan_by_id(data_dir, body.plan_id)
    if plan is None:
        raise ApiError(404, "plan_not_found", f"ไม่พบ plan_id: {body.plan_id}")

    execution_id  = f"exec_{uuid.uuid4().hex[:12]}"
    executed_at   = _now_iso()
    delivery_stats: dict = {}
    status        = "failed"

    # --- LINE OA ---
    if body.channel == "line_oa":
        token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
        if not token:
            raise ApiError(
                503, "missing_credentials",
                "LINE_CHANNEL_ACCESS_TOKEN ไม่ได้ตั้งค่า — ไม่สามารถส่งผ่าน LINE OA ได้"
            )

        # Build user_ids from audience export (if pos file exists)
        pos_path   = _store.get_pos_path(data_dir, plan.get("store_id", ""))
        user_ids   = _get_line_user_ids(data_dir, plan, pos_path)

        if not user_ids:
            raise ApiError(
                422, "no_audience",
                "ไม่มี LINE user_ids ในแผนนี้ — ต้อง export audience ที่มี LINE userId ก่อน"
            )

        message_text = body.message_text or "สวัสดีครับ! มีโปรโมชันพิเศษสำหรับคุณ 🎉"
        message = {"type": "text", "text": message_text}

        try:
            from line_oa_client import multicast
            result = multicast(user_ids=user_ids, message_template=message,
                               channel_access_token=token)
            delivery_stats = result
            status = "sent" if result["failed"] == 0 else "partial"
        except Exception as exc:
            raise ApiError(500, "line_error", f"LINE Multicast failed: {exc}")

    # --- Meta Ads ---
    else:
        required_meta = ["META_ACCESS_TOKEN", "META_AD_ACCOUNT_ID", "META_PAGE_ID"]
        missing = [k for k in required_meta if not os.environ.get(k)]
        if missing:
            raise ApiError(
                503, "missing_credentials",
                f"ต้องตั้งค่า env vars: {', '.join(missing)}"
            )

        budget = body.ad_budget_thb or plan.get("budget_input") or 1000.0

        try:
            from meta_ads_client import MetaAdsClient
            client = MetaAdsClient()

            # Get phone list from pos data
            pos_path   = _store.get_pos_path(data_dir, plan.get("store_id", ""))
            phone_list = _get_phone_list(pos_path)

            if not phone_list:
                raise ApiError(422, "no_phones",
                               "ไม่มีหมายเลขโทรศัพท์ในข้อมูล — Meta Ads ต้องใช้ phone list")

            campaign_name = f"CRM Campaign — {plan.get('store_id', '')} {executed_at[:10]}"

            audience_id  = client.upload_custom_audience(phone_list, campaign_name)
            lookalike_id = client.create_lookalike(audience_id)
            campaign_id  = client.create_campaign(campaign_name, "CONVERSIONS", budget)
            ad_set_id    = client.create_ad_set(campaign_id, audience_id, budget)

            # Basic creative (no image — use plan content if available)
            headline = "โปรโมชันพิเศษสำหรับคุณ"
            body_txt = "รีบใช้สิทธิ์ก่อนหมดเขต!"
            creative_id = client.create_creative("", headline, body_txt, "SHOP_NOW")
            ad_id       = client.create_ad(ad_set_id, creative_id, campaign_name)

            delivery_stats = {
                "audience_id":  audience_id,
                "lookalike_id": lookalike_id,
                "campaign_id":  campaign_id,
                "ad_set_id":    ad_set_id,
                "creative_id":  creative_id,
                "ad_id":        ad_id,
                "phones_uploaded": len(phone_list),
            }
            status = "queued"   # Meta ads start paused and go through review

        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(500, "meta_error", f"Meta Ads pipeline failed: {exc}")

    # Update campaign history with execution info
    _store.append_plan(data_dir, {
        "plan_id":        body.plan_id,
        "store_id":       plan.get("store_id", ""),
        "execution_id":   execution_id,
        "executed_at":    executed_at,
        "channel":        body.channel,
        "status":         f"executed_{status}",
        "delivery_stats": delivery_stats,
    })

    return {
        "execution_id":    execution_id,
        "plan_id":         body.plan_id,
        "channel":         body.channel,
        "status":          status,
        "executed_at":     executed_at,
        "delivery_summary": delivery_stats,
    }


def _get_line_user_ids(data_dir: str, plan: dict, pos_path: str | None) -> list[str]:
    """Extract LINE user_ids from the audience export or CRM data, if available."""
    store_id = plan.get("store_id", "")
    crm_path = os.path.join(data_dir, "stores", store_id, "crm_data.csv")
    if os.path.exists(crm_path):
        try:
            import pandas as pd
            df = pd.read_csv(crm_path)
            id_col = next((c for c in df.columns
                           if "line" in c.lower() and "id" in c.lower()), None)
            if id_col:
                ids = df[id_col].dropna().astype(str).tolist()
                return [uid for uid in ids if uid.startswith("U") and len(uid) >= 10]
        except Exception:
            pass
    return []


def _get_phone_list(pos_path: str | None) -> list[str]:
    """Extract phone numbers from POS file for Meta audience upload."""
    if not pos_path or not os.path.exists(pos_path):
        return []
    try:
        import pandas as pd
        df = pd.read_csv(pos_path, nrows=10_000)
        phone_col = next((c for c in df.columns
                          if any(k in c.lower() for k in ("phone", "mobile", "เบอร์", "โทร"))), None)
        if phone_col:
            null_s = {"-", "", "nan", "None", "NaN", "null"}
            phones = (
                df[phone_col].dropna().astype(str)
                .str.replace(r"[-\s]", "", regex=True)
                .loc[lambda s: ~s.isin(null_s)]
                .unique().tolist()
            )
            return phones
    except Exception:
        pass
    return []
