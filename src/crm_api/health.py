"""health.py — GET /segment/health endpoint.

Runs Layer 1 segmentation (no budget optimizer) → returns 10 IBM segments
with urgency scores, trend vs previous snapshot, and health score.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request

from .auth import require_key
from .errors import ApiError
from . import store as _store
from . import docs

router = APIRouter()

# IBM canonical segment names (always returned, count=0 if missing)
_IBM_SEGMENTS = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalist",
    "Promising",
    "Need Attention",
    "About to Sleep",
    "At-Risk",
    "Can't Lose Them",
    "Hibernating",
    "Lost",
]

# Static metadata per IBM segment — used by DotDash to render sort order,
# category badge, approach text, and recommended offer type.
# do_order = sort priority (1 = most urgent to act on)
# do_category = "ทำก่อน" | "ควรทำ" | "ทำได้" | "รักษา"
# approach = one-line Thai strategy
# recommended_offer = "ให้คูปอง" | "ให้แต้ม" | "ไม่มีโปร"
# recommended_offer_type = "coupon" | "points" | "none"
_SEGMENT_META: dict[str, dict] = {
    "Can't Lose Them": {
        "do_order":               1,
        "do_category":            "ทำก่อน",
        "approach":               "VIP กำลังหาย — ส่ง win-back ด่วนก่อนสูญถาวร",
        "recommended_offer":      "ให้คูปอง",
        "recommended_offer_type": "coupon",
    },
    "At-Risk": {
        "do_order":               2,
        "do_category":            "ทำก่อน",
        "approach":               "เริ่มห่างหาย — ดึงกลับด้วย bundle หรือส่วนลดพิเศษ",
        "recommended_offer":      "ให้คูปอง",
        "recommended_offer_type": "coupon",
    },
    "About to Sleep": {
        "do_order":               3,
        "do_category":            "ทำก่อน",
        "approach":               "เริ่มเงียบ — กระตุ้นก่อนหายเข้าช่วง Hibernating",
        "recommended_offer":      "ให้แต้ม",
        "recommended_offer_type": "points",
    },
    "Need Attention": {
        "do_order":               4,
        "do_category":            "ควรทำ",
        "approach":               "ค่าเฉลี่ยลดลง — ดูแลก่อนเลื่อนเข้า At-Risk",
        "recommended_offer":      "ให้แต้ม",
        "recommended_offer_type": "points",
    },
    "Hibernating": {
        "do_order":               5,
        "do_category":            "ควรทำ",
        "approach":               "ไม่ซื้อนาน — กระตุ้น reactivation ด้วยส่วนลดคุ้ม",
        "recommended_offer":      "ให้คูปอง",
        "recommended_offer_type": "coupon",
    },
    "Potential Loyalist": {
        "do_order":               6,
        "do_category":            "ทำได้",
        "approach":               "ซื้อบ่อยขึ้น — สร้างนิสัยด้วยแต้มสะสม",
        "recommended_offer":      "ให้แต้ม",
        "recommended_offer_type": "points",
    },
    "Promising": {
        "do_order":               7,
        "do_category":            "ทำได้",
        "approach":               "เพิ่งเริ่มมา — ชักจูงให้กลับมาซื้อครั้งที่ 2",
        "recommended_offer":      "ให้แต้ม",
        "recommended_offer_type": "points",
    },
    "Lost": {
        "do_order":               8,
        "do_category":            "ทำได้",
        "approach":               "หายไปนาน — ลองส่ง reactivation ถ้างบเหลือ",
        "recommended_offer":      "ให้คูปอง",
        "recommended_offer_type": "coupon",
    },
    "Champions": {
        "do_order":               9,
        "do_category":            "รักษา",
        "approach":               "ลูกค้าดีที่สุด — รักษาด้วยแต้มพิเศษ ไม่จำเป็นต้องลดราคา",
        "recommended_offer":      "ให้แต้ม",
        "recommended_offer_type": "points",
    },
    "Loyal Customers": {
        "do_order":               10,
        "do_category":            "รักษา",
        "approach":               "กลุ่มประจำ — reward และ bundle เพื่อเพิ่ม basket",
        "recommended_offer":      "ให้แต้ม",
        "recommended_offer_type": "points",
    },
}

_HEALTHY_SEGS   = {"Champions", "Loyal Customers", "Potential Loyalist", "Promising"}
_AT_RISK_SEGS   = {"At-Risk", "Can't Lose Them", "About to Sleep"}
_CHURNED_SEGS   = {"Hibernating", "Lost"}

_ALERT_WARNING_SEGS   = {"At-Risk", "Can't Lose Them", "Hibernating"}
_ALERT_INFO_SEGS      = {"Need Attention", "About to Sleep"}
_ALERT_WARNING_THRESH = 20.0
_ALERT_INFO_THRESH    = 10.0
_MIN_COUNT_FOR_ALERT  = 10


_VALID_GOALS = {"maximize_profit", "win_back", "retain_loyal", "maximize_reach"}


@router.get("/send-date/recommend", **docs.SEND_DATE_RECOMMEND)
def send_date_recommend(
    request: Request,
    store_id: str    = Query(...,  description=docs.P_STORE_ID),
    segment:  str    = Query(...,  description="IBM segment name เช่น 'At-Risk', 'Champions'"),
    month:    str    = Query(...,  description="เดือนเป้าหมาย รูปแบบ YYYY-MM เช่น 2026-11"),
    goal:     str    = Query("maximize_profit", description="เป้าหมาย campaign: maximize_profit | win_back | retain_loyal | maximize_reach"),
    cooldown_days: int = Query(14, description="ระยะ cooldown ขั้นต่ำ (วัน) ระหว่าง campaign"),
    top_n:    int    = Query(3,   description="จำนวน alternative dates ที่แสดง"),
    _: None = Depends(require_key),
):
    """Return the best send date for a campaign in the given month."""
    # -- Validate inputs -----------------------------------------------------
    try:
        yr, mo = int(month[:4]), int(month[5:7])
        if not (2020 <= yr <= 2050 and 1 <= mo <= 12):
            raise ValueError
    except (ValueError, IndexError):
        raise ApiError(400, "invalid_input", "month ต้องเป็นรูปแบบ YYYY-MM เช่น 2026-11")

    if goal not in _VALID_GOALS:
        raise ApiError(400, "invalid_input",
                       f"goal ต้องเป็นหนึ่งใน: {', '.join(sorted(_VALID_GOALS))}")

    if cooldown_days < 0 or cooldown_days > 365:
        raise ApiError(400, "invalid_input", "cooldown_days ต้องอยู่ระหว่าง 0–365")

    if top_n < 1 or top_n > 10:
        raise ApiError(400, "invalid_input", "top_n ต้องอยู่ระหว่าง 1–10")

    data_dir = str(request.app.state.settings.data_dir)

    # -- Load POS file -------------------------------------------------------
    pos_path = _store.get_pos_path(data_dir, store_id)
    if pos_path is None:
        raise ApiError(
            404, "store_not_found",
            "ไม่พบข้อมูลของ store_id นี้ กรุณาอัปโหลดข้อมูลผ่าน /campaign/plan ก่อน"
        )

    pos_df = _load_pos_df(pos_path)

    # -- Load campaign history -----------------------------------------------
    campaign_history = _store.get_plans(data_dir, store_id, limit=200)

    # -- Run recommender -----------------------------------------------------
    _add_src_to_path()
    try:
        from send_date_recommender import recommend_send_date, get_payday_dates
    except ImportError as exc:
        raise ApiError(500, "pipeline_error", f"send_date_recommender not available: {exc}")

    try:
        result = recommend_send_date(
            segment_name=segment,
            month=month,
            pos_df=pos_df,
            campaign_history=campaign_history,
            goal=goal,
            cooldown_days=cooldown_days,
            top_n=top_n,
        )
    except Exception as exc:
        raise ApiError(500, "pipeline_error", f"ไม่สามารถคำนวณวันที่ได้: {type(exc).__name__}: {exc}")

    return {
        "store_id":          store_id,
        "segment":           segment,
        "month":             month,
        "goal":              goal,
        "recommended_date":  result["recommended_date"],
        "score":             result["score"],
        "reason":            result["reason"],
        "payday_window":     result["payday_window"],
        "overlap_warning":   result["overlap_warning"],
        "alternatives":      result["alternatives"],
        "payday_dates":      get_payday_dates(month),
    }


@router.get("/segment/health", **docs.SEGMENT_HEALTH)
def segment_health(
    request: Request,
    store_id: str = Query(..., description=docs.P_STORE_ID),
    _: None = Depends(require_key),
):
    """Return current segment counts, urgency scores, trends, and health score."""
    data_dir = str(request.app.state.settings.data_dir)

    # -- Load POS file for this store -----------------------------------
    pos_path = _store.get_pos_path(data_dir, store_id)
    if pos_path is None:
        raise ApiError(
            404, "store_not_found",
            "ไม่พบข้อมูลของ store_id นี้ กรุณาอัปโหลดข้อมูลผ่าน /campaign/plan ก่อน"
        )

    # -- Run Layer 1 segmentation (lightweight — no uplift model) -------
    segments = _run_segmentation(pos_path)

    # -- Snapshot timestamp ---------------------------------------------
    snapshot_at = datetime.now(timezone.utc).isoformat()

    # -- Load previous snapshot for trend -------------------------------
    prev_snap = _store.get_prev_snapshot(data_dir, store_id, snapshot_at)
    prev_counts: dict[str, int] = {}
    if prev_snap:
        for seg_name, seg_data in prev_snap.get("segments_raw", {}).items():
            prev_counts[seg_name] = seg_data.get("count", 0)

    # -- Build padded segment dict (10 IBM segments always) -------------
    seg_dict: dict[str, dict] = {}
    total_customers = sum(s["customer_count"] for s in segments)

    for name in _IBM_SEGMENTS:
        ml_seg = next((s for s in segments if _name_match(s.get("segment_name", ""), name)), None)
        count = int(ml_seg["customer_count"]) if ml_seg else 0
        prev  = prev_counts.get(name, prev_counts.get(_fuzzy_key(name), 0))

        urgency_score   = int(ml_seg.get("urgency_score", 0)) if ml_seg else 0
        priority        = ml_seg.get("priority", "monitor") if ml_seg else "monitor"
        priority_color  = ml_seg.get("priority_color", "gray") if ml_seg else "gray"
        action_label    = ml_seg.get("action_label", "—") if ml_seg else "—"

        # Trend
        if prev_snap is None or prev == 0:
            trend_pct = None
            trend     = "none"
        else:
            trend_pct = round((count - prev) / prev * 100, 1) if prev > 0 else None
            if trend_pct is None:
                trend = "none"
            elif trend_pct > 5:
                trend = "growing"
            elif trend_pct < -5:
                trend = "shrinking"
            else:
                trend = "stable"

        if count == 0:
            urgency_score  = 0
            priority       = "monitor"
            priority_color = "gray"
            action_label   = "—"

        meta = _SEGMENT_META.get(name, {})
        seg_dict[name] = {
            "count":          count,
            "prev_count":     prev if prev_snap else None,
            "trend":          trend,
            "trend_pct":      trend_pct,
            "urgency_score":  urgency_score,
            "priority":       priority,
            "priority_color": priority_color,
            "action_label":   action_label,
            # DotDash display fields
            "do_order":               meta.get("do_order", 99),
            "do_category":            meta.get("do_category", "ทำได้"),
            "approach":               meta.get("approach", ""),
            "recommended_offer":      meta.get("recommended_offer", "ให้แต้ม"),
            "recommended_offer_type": meta.get("recommended_offer_type", "points"),
        }

    # -- Health score ---------------------------------------------------
    healthy  = sum(seg_dict[s]["count"] for s in _HEALTHY_SEGS if s in seg_dict)
    at_risk  = sum(seg_dict[s]["count"] for s in _AT_RISK_SEGS if s in seg_dict)
    churned  = sum(seg_dict[s]["count"] for s in _CHURNED_SEGS if s in seg_dict)
    total    = max(total_customers, 1)

    healthy_pct = round(healthy / total * 100, 1)
    at_risk_pct = round(at_risk  / total * 100, 1)
    churned_pct = round(churned  / total * 100, 1)

    health_label = (
        "ดี"       if healthy_pct >= 40 else
        "ปานกลาง"  if healthy_pct >= 25 else
        "ต้องดูแล"
    )

    # -- Alerts ---------------------------------------------------------
    alerts: list[dict] = []
    for name, data in seg_dict.items():
        if data["count"] < _MIN_COUNT_FOR_ALERT or data["trend_pct"] is None:
            continue
        if name in _ALERT_WARNING_SEGS and data["trend_pct"] >= _ALERT_WARNING_THRESH:
            alerts.append({
                "level":   "warning",
                "segment": name,
                "message": f"{name} โตขึ้น {data['trend_pct']:.0f}% จากช่วงก่อน — แนะนำทำ campaign โดยเร็ว",
            })
        elif name in _ALERT_INFO_SEGS and data["trend_pct"] >= _ALERT_INFO_THRESH:
            alerts.append({
                "level":   "info",
                "segment": name,
                "message": f"{name} เพิ่มขึ้น {data['trend_pct']:.0f}% — ควรวางแผน campaign",
            })

    # -- Recommendation -------------------------------------------------
    recommendation = _pick_recommendation(seg_dict)

    # -- Save snapshot --------------------------------------------------
    _store.append_snapshot(data_dir, {
        "store_id":     store_id,
        "snapshot_at":  snapshot_at,
        "segments_raw": {name: {"count": d["count"]} for name, d in seg_dict.items()},
        "total_customers": total_customers,
    })

    all_snapshots = _store.get_snapshots(data_dir, store_id)
    has_history   = len(all_snapshots) > 1

    return {
        "store_id":    store_id,
        "snapshot_at": snapshot_at,
        "total_customers": total_customers,
        "segments":    seg_dict,
        "health_score": {
            "healthy_pct": healthy_pct,
            "at_risk_pct": at_risk_pct,
            "churned_pct": churned_pct,
            "label":       health_label,
        },
        "alerts":      alerts,
        "recommendation": recommendation,
        "snapshot_history_available": has_history,
        "first_snapshot_at": all_snapshots[0].get("snapshot_at") if all_snapshots else None,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_segmentation(pos_path: str) -> list[dict]:
    """Run Layer 1 segmentation on a POS file. Returns segment list."""
    # Add src to path so we can import pipeline modules
    _add_src_to_path()
    from campaign_pipeline import load_and_preprocess_csv, layer1_csv_segmentation

    df = load_and_preprocess_csv(pos_path)
    if df is None or df.empty:
        return []

    try:
        segments, _ = layer1_csv_segmentation(df)
    except Exception:
        return []

    return segments


def _name_match(ml_name: str, ibm_name: str) -> bool:
    """Fuzzy match between ML segment name and IBM canonical name."""
    ml_lower  = ml_name.lower().strip()
    ibm_lower = ibm_name.lower().strip()
    if ibm_lower in ml_lower or ml_lower in ibm_lower:
        return True
    # Handle common variations
    aliases = {
        "champions":          ["champion"],
        "loyal customers":    ["loyal customer", "loyal"],
        "potential loyalist":  ["potential loyalists", "potential loyal"],
        "need attention":     ["needs attention"],
        "about to sleep":     ["about to sleep"],
        "at-risk":            ["at risk", "atrisk"],
        "can't lose them":    ["cant lose", "can't lose", "vip drifting"],
        "hibernating":        ["hibernate"],
    }
    for canonical, alts in aliases.items():
        if ibm_lower == canonical:
            if any(a in ml_lower for a in alts):
                return True
    return False


def _fuzzy_key(name: str) -> str:
    return name.lower().replace("'", "").replace("-", " ").strip()


def _add_src_to_path() -> None:
    src_dir = str(Path(__file__).parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _load_pos_df(pos_path: str):
    """Load a POS CSV file and return a normalised DataFrame (or None)."""
    try:
        _add_src_to_path()
        from campaign_pipeline import load_and_preprocess_csv
        return load_and_preprocess_csv(pos_path)
    except Exception:
        return None


def _pick_recommendation(seg_dict: dict) -> dict:
    """Pick the best campaign goal based on current segment state."""
    cant_lose    = seg_dict.get("Can't Lose Them", {})
    at_risk      = seg_dict.get("At-Risk", {})
    hibernating  = seg_dict.get("Hibernating", {})
    champions    = seg_dict.get("Champions", {})
    need_att     = seg_dict.get("Need Attention", {})

    # Prioritise win_back if critical segments are growing
    if (
        (cant_lose.get("trend") == "growing" and cant_lose.get("count", 0) >= 10)
        or (at_risk.get("trend") == "growing" and at_risk.get("count", 0) >= 20)
    ):
        return {
            "goal":    "win_back",
            "reason":  "Can't Lose Them และ At-Risk กำลังโต — Win-Back จะให้ ROI สูงสุด",
            "urgency": "high",
        }

    if hibernating.get("count", 0) >= 50:
        return {
            "goal":    "win_back",
            "reason":  "มีลูกค้า Hibernating จำนวนมาก — ควรพยายาม reactivate",
            "urgency": "medium",
        }

    if champions.get("trend") == "shrinking" or need_att.get("trend") == "growing":
        return {
            "goal":    "retain_loyal",
            "reason":  "Champions ลดลงหรือ Need Attention เพิ่ม — ควรรักษาฐานลูกค้าดี",
            "urgency": "medium",
        }

    return {
        "goal":    "maximize_profit",
        "reason":  "สถานการณ์ปกติ — optimize profit สูงสุดจากทุก segment",
        "urgency": "low",
    }
