"""Persist pipeline data the CRM API needs but the pipeline only kept in memory."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

from atomic_io import atomic_write_text

# Same week layout the scheduler uses (campaign_scheduler.build_schedule wave_week_map), 1-based weeks.
WAVE_WEEKS = {"1": [1, 3], "2": [2, 4], "3": [1]}


def _json_safe(obj):
    """NaN/inf -> None (strict JSON), numpy scalars -> python, anything else unknown -> str."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item) and getattr(obj, "shape", None) == ():
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def _get(d: dict, key: str):
    return d.get(key) if isinstance(d, dict) else None


def write_segments_latest(payload: dict, out_dir: str = "data/processed") -> str:
    """Write segments_latest.json (atomic) from the pipeline's merged payload; returns its path."""
    segments = []
    for seg in payload.get("target_segments", []):
        ue = seg.get("unit_economics") or {}
        segments.append({
            "id": seg.get("cluster_id"),
            "name": seg.get("segment_name"),
            "customers": seg.get("customer_count"),
            "broadcast": seg.get("broadcast_count"),
            "suppressed": seg.get("suppressed_count"),
            "recommended_promo": seg.get("recommended_promo"),
            "action_breakdown": seg.get("l4_action_breakdown"),
            "unit_economics": {
                "net_profit_per_head": _get(ue, "net_profit_per_head"),
                "total_projected_profit": _get(ue, "total_projected_profit"),
                "l4_total_projected_profit": _get(ue, "l4_total_projected_profit"),
                "financial_status": _get(ue, "financial_status"),
            },
        })
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_status": payload.get("gate_status"),
        "segments": segments,
        "wave_weeks": WAVE_WEEKS,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "segments_latest.json")
    atomic_write_text(path, json.dumps(_json_safe(doc), ensure_ascii=False, indent=2, allow_nan=False))
    return path


def export_segments_safe(payload: dict, out_dir: str = "data/processed") -> str | None:
    """Optional side-export for the CRM API: log and return None instead of ever aborting the pipeline."""
    try:
        return write_segments_latest(payload, out_dir)
    except Exception as exc:  # noqa: BLE001 - must never break the caller
        try:
            from pipeline_logger import log
            log("L4", "crm_export_failed", level="ERROR", exc=f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            pass
        return None


def write_customer_segments(df_rfm, segments: list, out_dir: str = "data/processed") -> str:
    """Write customer_segments_latest.csv: EVERY customer of the shop with the segment the pipeline gave them.

    Only ids and the segment (no recency/frequency/monetary numbers). It does not depend on any monthly plan: a shop
    whose Layer 4 recommends no sends still has a segmentation.
    """
    names = {int(s["cluster_id"]): s.get("segment_name") for s in segments}
    out = df_rfm[["customer_id", "cluster"]].copy()
    out["customer_id"] = out["customer_id"].astype(str)
    out["segment_id"] = out["cluster"].astype(int)
    out["segment_name"] = out["segment_id"].map(names)
    out = out.drop_duplicates("customer_id").sort_values("customer_id")[["customer_id", "segment_id", "segment_name"]]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "customer_segments_latest.csv")
    atomic_write_text(path, out.to_csv(index=False))
    return path


def export_customer_segments_safe(df_rfm, segments: list, out_dir: str = "data/processed") -> str | None:
    """Optional side-export: log and return None instead of ever aborting the pipeline."""
    try:
        return write_customer_segments(df_rfm, segments, out_dir)
    except Exception as exc:  # noqa: BLE001 - must never break the caller
        try:
            from pipeline_logger import log
            log("L1", "crm_export_failed", level="ERROR", exc=f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            pass
        return None
