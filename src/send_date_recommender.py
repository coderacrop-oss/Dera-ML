"""send_date_recommender.py — Recommend the best send date for a campaign.

Used by the DotDash calendar view (✦ วันที่ AI แนะนำ) so stores know not just
*what* to send but *when* to send it.

Three signals are combined into a score for each candidate date:
  1. Buy-day affinity      — which days of the week / month this segment buys most
  2. Payday window boost   — Thai payday falls 25th–end of month & 1st–5th
  3. Campaign overlap penalty — deduct score when another campaign already touches
                                the same segment within a cooldown window

Usage
-----
    from send_date_recommender import recommend_send_date, check_date_overlap

    # Recommend the best date in November 2026 for At-Risk win-back
    result = recommend_send_date(
        segment_name    = "At-Risk",
        month           = "2026-11",
        pos_df          = pos_df,          # historical transactions
        campaign_history= history,         # list[dict] from campaign_history.jsonl
        goal            = "win_back",
    )
    # → {date, score, reason, payday_window, alternatives: [...]}

    # Check if a proposed date clashes with existing campaigns
    clash = check_date_overlap("2026-11-15", history, segment="At-Risk")
    # → {has_overlap, nearest_campaign, days_apart}
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Thai paycheck pattern: monthly salary typically hits 25th–28th (private),
# 1st (government). Use a window to capture both.
_PAYDAY_START_DOM = 25   # day-of-month start
_PAYDAY_END_DOM   = 5    # day-of-month end (wraps into next month)
_PAYDAY_BOOST     = 0.20 # score multiplier applied during payday window

# Minimum days between campaigns targeting overlapping segments
_DEFAULT_COOLDOWN_DAYS = 14

# How many candidate dates to score (starting from the 1st of the month)
_CANDIDATE_WINDOW_DAYS = 31

# Day-of-week weights: 0=Mon … 6=Sun
# Thai retail: Sat/Sun strongest, Fri next, Mon weakest (adjust from data)
_DOW_DEFAULT_WEIGHTS = {0: 0.6, 1: 0.65, 2: 0.70, 3: 0.75, 4: 0.85, 5: 1.0, 6: 0.95}

# How far back to look at POS history to infer buy-day patterns
_HISTORY_LOOKBACK_DAYS = 90

# Goal → preferred week of month (1-indexed, 1 = first week)
# win_back / rescue → start of month (payday energy)
# retain / maximize_profit → middle of month (steady traffic)
# maximize_reach → early to maximise window
_GOAL_WEEK_PREFERENCE: dict[str, int] = {
    "win_back":        1,
    "retain_loyal":    2,
    "maximize_profit": 2,
    "maximize_reach":  1,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend_send_date(
    segment_name: str,
    month: str,
    pos_df: Any = None,
    campaign_history: list[dict] | None = None,
    goal: str = "maximize_profit",
    cooldown_days: int = _DEFAULT_COOLDOWN_DAYS,
    top_n: int = 3,
) -> dict:
    """Return the best send date for a campaign in the given month.

    Parameters
    ----------
    segment_name : str
        IBM segment name e.g. "At-Risk", "Champions".
    month : str
        Target month "YYYY-MM".
    pos_df : pd.DataFrame, optional
        Historical POS transactions with a date/timestamp column.
        Used to infer buy-day patterns.  Skipped if None.
    campaign_history : list[dict], optional
        Previously executed campaigns.  Each dict should have:
        ``{execution_id, executed_at, segment_name, plan_id}``
    goal : str
        One of: maximize_profit | win_back | retain_loyal | maximize_reach.
    cooldown_days : int
        Minimum days since last campaign touching the same segment.
    top_n : int
        Number of alternative dates to include.

    Returns
    -------
    dict
        ``{recommended_date, score, reason, payday_window,
           alternatives: [{date, score, reason}],
           overlap_warning: str | None}``
    """
    campaign_history = campaign_history or []
    yr, mo = _parse_month(month)

    # Build candidate dates (all days in the month)
    candidates = _month_dates(yr, mo)

    # Compute buy-day affinity weights from POS history
    dow_weights = _infer_dow_weights(pos_df, segment_name)
    dom_weights = _infer_dom_weights(pos_df, segment_name)

    # Score each candidate
    scored: list[dict] = []
    for d in candidates:
        score, signals = _score_date(
            d, dow_weights, dom_weights, campaign_history,
            segment_name, cooldown_days, goal,
        )
        scored.append({"date": d.isoformat(), "score": round(score, 4),
                       "signals": signals})

    scored.sort(key=lambda x: x["score"], reverse=True)

    best   = scored[0]
    alts   = scored[1:top_n]

    best_date = date.fromisoformat(best["date"])
    reason    = _build_reason(best["signals"], best_date)
    overlap   = check_date_overlap(best["date"], campaign_history, segment_name, cooldown_days)

    return {
        "recommended_date": best["date"],
        "score":            best["score"],
        "reason":           reason,
        "payday_window":    _is_payday(best_date),
        "alternatives":     [
            {"date": a["date"], "score": a["score"],
             "reason": _build_reason(a["signals"], date.fromisoformat(a["date"]))}
            for a in alts
        ],
        "overlap_warning": (
            f"มีแคมเปญใกล้เคียงใน {overlap['days_apart']} วัน "
            f"({overlap['nearest_campaign_date']}) — แนะนำเลื่อนออกไป"
            if overlap["has_overlap"] else None
        ),
    }


def check_date_overlap(
    proposed_date: str,
    campaign_history: list[dict],
    segment: str = "",
    cooldown_days: int = _DEFAULT_COOLDOWN_DAYS,
) -> dict:
    """Check if ``proposed_date`` is too close to an existing campaign.

    Considers campaigns that overlap with ``segment`` (or all campaigns if
    segment is empty).

    Returns
    -------
    dict
        ``{has_overlap, days_apart, nearest_campaign_date, nearest_campaign_id}``
    """
    target = date.fromisoformat(proposed_date) if isinstance(proposed_date, str) else proposed_date

    nearest_days: int | None = None
    nearest_date_str         = ""
    nearest_id               = ""

    for c in campaign_history:
        exec_at_str = c.get("executed_at", "")
        if not exec_at_str:
            continue
        try:
            exec_dt = datetime.fromisoformat(exec_at_str.replace("Z", "+00:00"))
            exec_d  = exec_dt.date()
        except ValueError:
            continue

        # Segment overlap check: skip if segment specified and no match
        if segment:
            c_seg = str(c.get("segment_name", "") or c.get("segments", "") or "").lower()
            if c_seg and segment.lower() not in c_seg and c_seg not in segment.lower():
                continue

        days_apart = abs((target - exec_d).days)
        if nearest_days is None or days_apart < nearest_days:
            nearest_days     = days_apart
            nearest_date_str = exec_d.isoformat()
            nearest_id       = str(c.get("execution_id", c.get("plan_id", "")))

    has_overlap = nearest_days is not None and nearest_days < cooldown_days

    return {
        "has_overlap":             has_overlap,
        "days_apart":              nearest_days if nearest_days is not None else None,
        "nearest_campaign_date":   nearest_date_str,
        "nearest_campaign_id":     nearest_id,
    }


def get_payday_dates(month: str) -> list[str]:
    """Return all dates in ``month`` that fall in the Thai payday window.

    Payday window: 25th of previous month → 5th of this month.
    """
    yr, mo = _parse_month(month)
    result: list[str] = []
    for d in _month_dates(yr, mo):
        if _is_payday(d):
            result.append(d.isoformat())
    return result


# ---------------------------------------------------------------------------
# Scoring internals
# ---------------------------------------------------------------------------

def _score_date(
    d: date,
    dow_weights: dict[int, float],
    dom_weights: dict[int, float],
    campaign_history: list[dict],
    segment_name: str,
    cooldown_days: int,
    goal: str,
) -> tuple[float, dict]:
    """Score a single candidate date. Returns (score, signals_dict)."""
    signals: dict[str, float] = {}

    # 1. Day-of-week affinity (0–1 scale, normalised)
    dow_w = dow_weights.get(d.weekday(), 0.5)
    signals["dow_affinity"] = round(dow_w, 3)

    # 2. Day-of-month affinity from POS history
    dom_w = dom_weights.get(d.day, 0.5)
    signals["dom_affinity"] = round(dom_w, 3)

    # 3. Payday boost
    payday_boost = _PAYDAY_BOOST if _is_payday(d) else 0.0
    signals["payday_boost"] = payday_boost

    # 4. Overlap penalty: subtract for proximity to existing campaigns
    overlap = check_date_overlap(d.isoformat(), campaign_history, segment_name, cooldown_days)
    if overlap["has_overlap"] and overlap["days_apart"] is not None:
        # Linear penalty: 0 days apart → -0.5, cooldown_days → 0
        penalty = 0.5 * (1 - overlap["days_apart"] / cooldown_days)
        signals["overlap_penalty"] = -round(penalty, 3)
    else:
        signals["overlap_penalty"] = 0.0

    # 5. Goal week preference
    week_of_month  = (d.day - 1) // 7 + 1   # 1–5
    preferred_week = _GOAL_WEEK_PREFERENCE.get(goal, 2)
    week_dist      = abs(week_of_month - preferred_week)
    goal_weight    = max(0.0, 1.0 - week_dist * 0.2)
    signals["goal_week_fit"] = round(goal_weight, 3)

    # Combine — equal weights except overlap penalty
    score = (
        0.30 * signals["dow_affinity"]
        + 0.20 * signals["dom_affinity"]
        + 0.20 * signals["payday_boost"]
        + 0.20 * signals["goal_week_fit"]
        + signals["overlap_penalty"]   # already negative
    )
    return max(0.0, score), signals


# ---------------------------------------------------------------------------
# Buy-day pattern inference from POS
# ---------------------------------------------------------------------------

def _infer_dow_weights(pos_df: Any, segment_name: str) -> dict[int, float]:
    """Return {day_of_week: normalised_weight} from recent POS history.

    Falls back to `_DOW_DEFAULT_WEIGHTS` if pos_df is None or has no data.
    """
    try:
        import pandas as pd
        if pos_df is None or not isinstance(pos_df, pd.DataFrame) or pos_df.empty:
            return dict(_DOW_DEFAULT_WEIGHTS)

        df = pos_df.copy()
        date_col = _detect_date_col(df)
        if date_col is None:
            return dict(_DOW_DEFAULT_WEIGHTS)

        cutoff = datetime.now(timezone.utc) - timedelta(days=_HISTORY_LOOKBACK_DAYS)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df = df[df[date_col] >= cutoff].dropna(subset=[date_col])

        if df.empty:
            return dict(_DOW_DEFAULT_WEIGHTS)

        counts = df[date_col].dt.day_of_week.value_counts()
        total  = counts.sum()
        if total == 0:
            return dict(_DOW_DEFAULT_WEIGHTS)

        weights = {int(dow): round(float(cnt) / total * 7, 3) for dow, cnt in counts.items()}
        # Fill any missing days with low weight
        for d in range(7):
            if d not in weights:
                weights[d] = 0.1
        return weights

    except Exception:
        return dict(_DOW_DEFAULT_WEIGHTS)


def _infer_dom_weights(pos_df: Any, segment_name: str) -> dict[int, float]:
    """Return {day_of_month: normalised_weight} from recent POS history."""
    try:
        import pandas as pd
        if pos_df is None or not isinstance(pos_df, pd.DataFrame) or pos_df.empty:
            return {}

        df = pos_df.copy()
        date_col = _detect_date_col(df)
        if date_col is None:
            return {}

        cutoff = datetime.now(timezone.utc) - timedelta(days=_HISTORY_LOOKBACK_DAYS)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df = df[df[date_col] >= cutoff].dropna(subset=[date_col])

        if df.empty:
            return {}

        counts = df[date_col].dt.day.value_counts()
        total  = counts.sum()
        if total == 0:
            return {}

        weights = {int(dom): round(float(cnt) / total * 28, 3) for dom, cnt in counts.items()}
        return weights

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_month(month: str) -> tuple[int, int]:
    yr, mo = int(month[:4]), int(month[5:7])
    return yr, mo


def _month_dates(yr: int, mo: int) -> list[date]:
    """Return all dates in the given month."""
    d = date(yr, mo, 1)
    result: list[date] = []
    while d.month == mo:
        result.append(d)
        d += timedelta(days=1)
    return result


def _is_payday(d: date) -> bool:
    """Return True if ``d`` falls in the Thai payday window (25th–5th)."""
    return d.day >= _PAYDAY_START_DOM or d.day <= _PAYDAY_END_DOM


def _detect_date_col(df: Any) -> str | None:
    """Return the most likely date column name."""
    for col in ("date", "transaction_date", "timestamp", "created_at", "order_date"):
        if col in df.columns:
            return col
    return None


def _build_reason(signals: dict, d: date) -> str:
    """Build a one-line Thai reason string from scoring signals."""
    parts: list[str] = []

    dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัส", "ศุกร์", "เสาร์", "อาทิตย์"]
    parts.append(f"วัน{dow_names[d.weekday()]}")

    if signals.get("payday_boost", 0) > 0:
        parts.append("ช่วงเงินเดือนออก")

    if signals.get("dom_affinity", 0.5) > 0.8:
        parts.append(f"ลูกค้าซื้อบ่อยช่วงวันที่ {d.day}")

    if signals.get("overlap_penalty", 0) < -0.1:
        parts.append("⚠️ ใกล้กับ campaign ก่อนหน้า")

    if signals.get("goal_week_fit", 0) >= 0.8:
        parts.append("ตรงช่วงเวลาที่เหมาะกับเป้าหมาย")

    return " · ".join(parts) if parts else f"วันที่ {d.isoformat()}"
