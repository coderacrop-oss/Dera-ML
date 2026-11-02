"""Persist pipeline data the CRM API needs but the pipeline only kept in memory."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from atomic_io import atomic_write_text

# Same week layout the scheduler uses (campaign_scheduler.build_schedule wave_week_map), 1-based weeks.
WAVE_WEEKS = {"1": [1, 3], "2": [2, 4], "3": [1]}


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
    atomic_write_text(path, json.dumps(doc, ensure_ascii=False, indent=2, default=int))
    return path
