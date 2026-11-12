"""plan_insights.py — what a CRM screen needs to say about a monthly plan and about a plan once it has been measured.

Pure functions (pandas in, plain dicts out); nothing here reads or writes a file.

  plan_rows(...)            one row per (segment, reward): who, when (hours and days), what the plan expects
  top1_counterfactual(rows) what the plan expects from its best row alone ("if only the first one")
  model_accuracy(...)       expected vs actual profit (error %, optional bootstrap interval) and whether a retrain is recommended
  retrain_criteria()        the rules behind "retrain recommended" (the same constants retrain_trigger uses, plus the calibration band)
  outcome_summary(...)      a short Thai summary and next_actions[] after a plan has been measured

The words come from fixed templates: the numbers decide, nothing is generated.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from retrain_trigger import PSI_RETRAIN, QINI_MIN_ACCEPTABLE

# actual / predicted outside this band => the model's expectations were off enough to recommend a retrain
CALIBRATION_BAND = (0.8, 1.25)
MEASURE_DAYS = 30


# ── when to send ──────────────────────────────────────────────────────────────────────────────

def hour_blocks(hours: Iterable[int]) -> list[tuple[int, int]]:
    """Contiguous runs of hours as (start, end_exclusive): [11,12,13,17,18] -> [(11, 14), (17, 19)]."""
    hs = sorted({int(h) for h in hours if h is not None and 0 <= int(h) <= 23})
    blocks: list[tuple[int, int]] = []
    for h in hs:
        if blocks and blocks[-1][1] == h:
            blocks[-1] = (blocks[-1][0], h + 1)
        else:
            blocks.append((h, h + 1))
    return blocks


def window_kind(send_window: Any) -> str:
    """'off_peak' for the engine's "Off-Peak Window", 'peak' otherwise (the engine's own rule in _send_hour)."""
    return "off_peak" if "Off-Peak" in str(send_window) else "peak"


def _hh(h: int) -> str:
    return f"{min(max(h, 0), 23):02d}:00" if h < 24 else "23:59"


def window_for(kind: str, peak_hours: Iterable[int], off_peak_hours: Iterable[int]) -> dict | None:
    """The hours of the busy (or quiet) period a message is sent ahead of: the block that starts at the earliest hour,
    e.g. peak hours [11,12,13,17,18] -> 11:00-14:00. The message itself goes out the hour before it opens (what the scheduler does)."""
    blocks = hour_blocks(off_peak_hours if kind == "off_peak" else peak_hours)
    if not blocks:
        return None
    start, end = blocks[0]
    notify = max(0, start - 1)
    return {"start": _hh(start), "end": _hh(end), "kind": kind, "notify_hour": notify, "notify_at": _hh(notify),
            "all_blocks": [{"start": _hh(s), "end": _hh(e)} for s, e in blocks]}


def plan_rows(schedule_df: pd.DataFrame, holdout_df: pd.DataFrame | None, peak_hours: Iterable[int], off_peak_hours: Iterable[int]) -> list[dict]:
    """One row per (segment_name, recommended_action) of the schedule, best expected profit first."""
    if schedule_df is None or len(schedule_df) == 0:
        return []
    peak_hours, off_peak_hours = list(peak_hours), list(off_peak_hours)
    held: dict[tuple[str, str], int] = {}
    if holdout_df is not None and len(holdout_df):
        for (seg, act), g in holdout_df.groupby([holdout_df["segment_name"].astype(str), holdout_df["recommended_action"].astype(str)]):
            held[(seg, act)] = int(len(g))
    df = schedule_df.copy()
    df["_seg"] = df["segment_name"].astype(str)
    df["_act"] = df["recommended_action"].astype(str)
    df["_kind"] = df["send_window"].map(window_kind) if "send_window" in df.columns else "peak"
    dates = pd.to_datetime(df["send_datetime"]).dt.strftime("%Y-%m-%d") if "send_datetime" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["_date"] = dates
    rows: list[dict] = []
    for (seg, act), g in df.groupby(["_seg", "_act"], sort=False):
        profit = round(float(pd.to_numeric(g["expected_profit"], errors="coerce").fillna(0).sum()), 2)
        kinds = g["_kind"].value_counts()
        windows = [w for k in kinds.index if (w := window_for(k, peak_hours, off_peak_hours))]
        for w in windows:
            w["customers"] = int(kinds[w["kind"]])
        days = sorted({d for d in g["_date"].dropna()})
        def first(col: str):
            return next((str(v) for v in g[col].dropna() if str(v) not in ("", "nan", "None")), None) if col in g.columns else None
        rows.append({
            "segment": seg, "action": act, "action_key": first("action_key"), "action_group": first("action_group"),
            "customers": int(len(g)), "holdout_customers": held.get((seg, act), 0),
            "expected_profit": profit, "profit_per_customer": round(profit / len(g), 2),
            "waves": sorted({int(w) for w in g["wave"].dropna()}) if "wave" in g.columns else [],
            "send_window": windows[0] if windows else None,           # the one most of the row's customers get
            "send_windows": windows,                                  # every kind the row has (peak and off-peak)
            "recommended_dates": days, "first_send_date": days[0] if days else None, "last_send_date": days[-1] if days else None,
        })
    rows.sort(key=lambda r: (-r["expected_profit"], r["segment"], r["action"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def top1_counterfactual(rows: list[dict]) -> dict | None:
    """What the plan expects from its first row alone. An estimate from the engine's own per-customer expectation: nothing was tried."""
    if not rows:
        return None
    best = max(rows, key=lambda r: (r["expected_profit"], -r["rank"]))
    total = round(sum(r["expected_profit"] for r in rows), 2)
    return {
        "rank": 1, "rank_by": "expected_profit", "segment": best["segment"], "action": best["action"], "action_key": best.get("action_key"),
        "customers": best["customers"], "expected_profit": best["expected_profit"], "plan_expected_profit": total,
        "share_of_plan_pct": round(best["expected_profit"] / total * 100, 1) if total else None,
        "note_th": "ค่าประมาณจากกำไรที่โมเดลคาดไว้ของแถวอันดับ 1 เท่านั้น ไม่ใช่ผลจริงของการส่งเฉพาะแถวนี้ (ไม่ได้ทดลองจริง)",
    }


# ── how close the model was ───────────────────────────────────────────────────────────────────

def retrain_criteria() -> list[dict]:
    """The rules behind 'retrain recommended'. The first two are retrain_trigger's; the third is the band for actual / predicted profit."""
    return [
        {"id": "feature_drift", "source": "ml", "threshold": PSI_RETRAIN, "text_th": f"ฟีเจอร์ของลูกค้าเปลี่ยนไปจากตอนเทรน (PSI สูงสุดตั้งแต่ {PSI_RETRAIN} ขึ้นไป) = ควร retrain"},
        {"id": "champion_qini", "source": "ml", "threshold": QINI_MIN_ACCEPTABLE, "text_th": f"คะแนน Qini ของโมเดลต่ำกว่า {QINI_MIN_ACCEPTABLE} (แยกกลุ่มที่โปรได้ผลไม่ได้) = ควร retrain"},
        {"id": "calibration", "source": "ml", "threshold": list(CALIBRATION_BAND), "text_th": f"กำไรจริง/ที่คาด อยู่นอกช่วง {CALIBRATION_BAND[0]}–{CALIBRATION_BAND[1]} = ควร retrain"},
    ]


def _bootstrap_error_pct(samples: list[tuple[float, float]], n_boot: int = 1000, seed: int = 42) -> list[float] | None:
    """95% interval of (sum actual - sum predicted) / sum predicted over per-customer (predicted, actual) pairs; None when there are too few."""
    if len(samples) < 30:
        return None
    arr = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        p, a = arr[idx, 0].sum(), arr[idx, 1].sum()
        if p > 0:
            errs.append((a - p) / p * 100)
    if len(errs) < n_boot // 2:
        return None
    lo, hi = np.percentile(errs, [2.5, 97.5])
    return [round(float(lo), 1), round(float(hi), 1)]


def model_accuracy(predicted: float | None, actual: float | None, samples: list[tuple[float, float]] | None = None) -> dict:
    """Expected vs actual profit. `samples` (per-customer predicted, actual) adds a bootstrap 95% interval for the error."""
    pred = float(predicted) if predicted is not None else None
    act = float(actual) if actual is not None else None
    ok = pred is not None and act is not None and pred > 0
    ratio = round(act / pred, 3) if ok else None
    outside = ratio is not None and (ratio < CALIBRATION_BAND[0] or ratio > CALIBRATION_BAND[1])
    return {
        "predicted_profit": round(pred, 2) if pred is not None else None, "actual_profit": round(act, 2) if act is not None else None,
        "error_pct": round((act - pred) / pred * 100, 1) if ok else None, "calibration_ratio": ratio,
        "error_interval_pct": _bootstrap_error_pct(samples) if (ok and samples) else None,
        "within_band": None if ratio is None else (not outside), "band": list(CALIBRATION_BAND),
        "retrain_recommended": bool(outside), "retrain_criteria": retrain_criteria(),
    }


# ── what to say after a plan was measured ─────────────────────────────────────────────────────

def _baht(n: float) -> str:
    return f"{n:,.2f}".rstrip("0").rstrip(".")


def outcome_summary(predicted: float | None, actual: float | None, *, lift_pct: float | None = None, window_complete: bool = True,
                    days: int = MEASURE_DAYS, groups: list[dict] | None = None, samples: list[tuple[float, float]] | None = None) -> dict:
    """summary_th, next_actions[] and model_accuracy for a plan that has been measured.
    `groups`: per campaign/row results, each with a 'measurable' flag (compared with its own control group)."""
    acc = model_accuracy(predicted, actual, samples)
    ratio = acc["calibration_ratio"]
    measurable = [g for g in (groups or []) if g.get("measurable")]
    unmeasurable = [g for g in (groups or []) if not g.get("measurable")]
    actions: list[dict] = []
    if not window_complete:
        actions.append({"code": "WAIT_FOR_WINDOW", "text_th": f"ยังไม่ครบ {days} วันหลังส่ง ผลนี้ยังไม่เสถียร ส่งผลอีกครั้งเมื่อครบ"})
    if acc["retrain_recommended"]:
        actions.append({"code": "REQUEST_RETRAIN", "text_th": f"ผลจริงต่างจากที่คาดมาก (จริง/คาด = {ratio}) ขอ retrain (reason: accuracy_concern)"})
    if lift_pct is not None and lift_pct <= 0:
        actions.append({"code": "PROMO_NO_LIFT", "text_th": "คนที่ได้โปรซื้อไม่มากกว่ากลุ่มเทียบ พิจารณาลดงบหรือเปลี่ยนข้อเสนอของกลุ่มนี้ในรอบถัดไป"})
    if groups and not measurable:
        actions.append({"code": "RAISE_HOLDOUT", "text_th": "ไม่มีแคมเปญไหนวัดผลได้ (กลุ่มควบคุมน้อยเกินไป) รอบถัดไปเพิ่มสัดส่วนกลุ่มควบคุม หรือรวมกลุ่มเล็กเข้าด้วยกัน"})
    elif unmeasurable:
        actions.append({"code": "SOME_NOT_MEASURABLE", "text_th": f"{len(unmeasurable)} จาก {len(groups or [])} แคมเปญวัดผลเทียบไม่ได้ (ดูเหตุผลในแต่ละแคมเปญ)"})
    if ratio is not None and not acc["retrain_recommended"] and (lift_pct is None or lift_pct > 0) and window_complete and (not groups or measurable):
        actions.append({"code": "KEEP_GOING", "text_th": "ผลใกล้เคียงที่คาดและโปรได้ผล ทำแผนรอบถัดไปตามแนวทางเดิมได้"})
    parts = [f"แผนนี้คาดกำไร {_baht(acc['predicted_profit'] or 0)} บาท"
             + (f" ผลจริงประมาณ {_baht(acc['actual_profit'])} บาท" if acc["actual_profit"] is not None else " (ยังประมาณผลจริงไม่ได้)")
             + (f" (จริง/คาด {ratio})" if ratio is not None else "")]
    if lift_pct is not None:
        parts.append(f"คนที่ได้โปรซื้อ{'มากกว่า' if lift_pct >= 0 else 'น้อยกว่า'}กลุ่มเทียบ {abs(lift_pct):g}%")
    if groups:
        parts.append(f"วัดผลเทียบกลุ่มควบคุมได้ {len(measurable)} จาก {len(groups)} แคมเปญ")
    if not window_complete:
        parts.append("ผลยังไม่ครบกำหนด")
    return {"summary_th": " · ".join(parts), "next_actions": actions, "model_accuracy": acc}
