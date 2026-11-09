"""lead_scorer.py — Rule-based lead scoring for premium products.

Scores LINE OA members or CRM customers by their engagement signals, assigning
a numeric score and a tier so the campaign engine knows who deserves a
high-touch premium follow-up vs a standard nurture sequence.

Signal hierarchy (cumulative, strongest wins)
----------------------------------------------
Signals are read from LINE OA tags and/or a CRM activity log.

TAG SIGNALS (set in LINE OA Manager)
  asked_price          → +40  (highest intent: enquired about price)
  replied / replied_*  → +25  (sent a reply message in OA chat)
  clicked / clicked_*  → +15  (clicked a link/CTA in a broadcast)
  viewed / viewed_*    → +5   (opened a broadcast message)
  interested_premium   → +20  (manually tagged by staff as interested)
  vip                  → +10  (existing VIP customer)

PURCHASE SIGNALS (from POS history)
  has_premium_purchase → +30  (bought a premium item before)
  total_purchases ≥ 3  → +10  (repeat buyer → higher lifetime value)

RECENCY BONUS (last activity within N days)
  within 7 days        → +10
  within 30 days       → +5
  >30 days             → +0

SCORE TIERS
  hot    ≥ 60   → direct sales call recommended
  warm   ≥ 35   → premium campaign, personalised message
  cold   ≥ 10   → standard nurture
  unfit  <  10  → do nothing (mass or non-buyer)

Usage
-----
    from lead_scorer import score_lead, score_leads_df, get_lead_report

    # Score a single lead
    lead = {
        "customer_id": "C1",
        "tags": ["asked_price", "clicked_promo"],
        "has_premium_purchase": True,
        "total_purchases": 5,
        "last_activity_days": 3,
    }
    result = score_lead(lead)
    # → {"customer_id": "C1", "score": 90, "tier": "hot", "signals": [...]}

    # Score a whole DataFrame
    scored_df = score_leads_df(df)        # adds "lead_score" and "lead_tier" columns
    report    = get_lead_report(scored_df)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score weights
# ---------------------------------------------------------------------------

# Tag-based signals: each tuple is (match_fn_or_prefix, points, label)
_TAG_RULES: list[tuple] = [
    ("asked_price",         40, "asked_price"),
    ("interested_premium",  20, "interested_premium"),
    ("replied",             25, "replied"),
    ("clicked",             15, "clicked"),
    ("viewed",              5,  "viewed"),
    ("vip",                 10, "vip"),
]

_PURCHASE_BONUS_PREMIUM   = 30
_PURCHASE_BONUS_REPEAT    = 10   # total_purchases >= 3
_PURCHASE_REPEAT_THRESHOLD = 3

_RECENCY_BONUS_7D   = 10
_RECENCY_BONUS_30D  = 5
_RECENCY_7D_DAYS    = 7
_RECENCY_30D_DAYS   = 30

# Tier thresholds
_TIER_HOT   = 60
_TIER_WARM  = 35
_TIER_COLD  = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_lead(lead: dict) -> dict:
    """Score a single lead dict and return enriched result.

    Parameters
    ----------
    lead : dict
        Keys used (all optional):
          ``customer_id``       str
          ``tags``              list[str]  — LINE OA tags for this member
          ``has_premium_purchase`` bool    — bought a premium item
          ``total_purchases``   int        — total POS transactions
          ``last_activity_days`` int|None  — days since last tag/chat activity
          ``scored_at``         str        — override ISO timestamp (for testing)

    Returns
    -------
    dict
        ``{customer_id, score, tier, signals, scored_at}``
    """
    tags            = [str(t).lower().strip() for t in lead.get("tags", []) or []]
    has_premium     = bool(lead.get("has_premium_purchase", False))
    total_purchases = int(lead.get("total_purchases", 0) or 0)
    last_days       = lead.get("last_activity_days")

    score   = 0
    signals: list[str] = []

    # --- Tag signals ---
    for tag_prefix, points, label in _TAG_RULES:
        if _tag_matches(tags, tag_prefix):
            score += points
            signals.append(label)

    # --- Purchase signals ---
    if has_premium:
        score += _PURCHASE_BONUS_PREMIUM
        signals.append("premium_purchase")

    if total_purchases >= _PURCHASE_REPEAT_THRESHOLD:
        score += _PURCHASE_BONUS_REPEAT
        signals.append("repeat_buyer")

    # --- Recency bonus ---
    if last_days is not None:
        try:
            days = int(last_days)
            if days <= _RECENCY_7D_DAYS:
                score += _RECENCY_BONUS_7D
                signals.append("active_7d")
            elif days <= _RECENCY_30D_DAYS:
                score += _RECENCY_BONUS_30D
                signals.append("active_30d")
        except (TypeError, ValueError):
            pass

    tier = _score_to_tier(score)

    return {
        "customer_id": str(lead.get("customer_id", "") or ""),
        "score":       score,
        "tier":        tier,
        "signals":     signals,
        "scored_at":   lead.get("scored_at") or datetime.now(timezone.utc).isoformat(),
    }


def score_leads_df(df, tags_col: str = "tags", **kwargs) -> Any:
    """Add ``lead_score`` and ``lead_tier`` columns to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``customer_id``.  Optional columns: ``tags``, ``has_premium_purchase``,
        ``total_purchases``, ``last_activity_days``.
    tags_col : str
        Column name that holds the list of tags.  Default "tags".
    **kwargs
        Passed through to each ``score_lead`` call (e.g. ``scored_at`` for testing).

    Returns
    -------
    pd.DataFrame
        Copy of df with ``lead_score``, ``lead_tier``, ``lead_signals`` added.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required")

    df = df.copy()

    scores: list[int]      = []
    tiers:  list[str]      = []
    signals: list[list]    = []

    for _, row in df.iterrows():
        lead = {
            "customer_id":           row.get("customer_id", ""),
            "tags":                  row.get(tags_col, []) or [],
            "has_premium_purchase":  bool(row.get("has_premium_purchase", False)),
            "total_purchases":       int(row.get("total_purchases", 0) or 0),
            "last_activity_days":    row.get("last_activity_days"),
            **{k: v for k, v in kwargs.items()},
        }
        result = score_lead(lead)
        scores.append(result["score"])
        tiers.append(result["tier"])
        signals.append(result["signals"])

    df["lead_score"]   = scores
    df["lead_tier"]    = tiers
    df["lead_signals"] = signals

    _logger.info(
        "score_leads_df: scored %d leads — hot=%d warm=%d cold=%d unfit=%d",
        len(df),
        (df["lead_tier"] == "hot").sum(),
        (df["lead_tier"] == "warm").sum(),
        (df["lead_tier"] == "cold").sum(),
        (df["lead_tier"] == "unfit").sum(),
    )
    return df


def get_lead_report(df_or_results, top_n: int = 10) -> str:
    """Return a Thai summary of scored leads suitable for a sales briefing.

    Parameters
    ----------
    df_or_results : pd.DataFrame | list[dict]
        Output of ``score_leads_df`` (DataFrame) or a list of ``score_lead`` results.
    top_n : int
        Number of hot/warm leads to list by name.

    Returns
    -------
    str
    """
    try:
        import pandas as pd
        if isinstance(df_or_results, pd.DataFrame):
            records = df_or_results.to_dict("records")
        else:
            records = list(df_or_results)
    except ImportError:
        records = list(df_or_results)

    if not records:
        return "ไม่มีข้อมูล leads"

    # Normalise: score_lead results use "score"/"tier"; df uses "lead_score"/"lead_tier"
    def _get(r, key, fallback_key, default):
        return r.get(key, r.get(fallback_key, default))

    hot    = [r for r in records if _get(r, "lead_tier", "tier", "") == "hot"]
    warm   = [r for r in records if _get(r, "lead_tier", "tier", "") == "warm"]
    cold   = [r for r in records if _get(r, "lead_tier", "tier", "") == "cold"]
    unfit  = [r for r in records if _get(r, "lead_tier", "tier", "") == "unfit"]

    lines = [
        "รายงาน Lead Scoring (Premium)",
        f"  Leads ทั้งหมด           : {len(records):,}",
        f"  🔥 Hot  (≥{_TIER_HOT})          : {len(hot):,}  — โทรหาเลย",
        f"  🌡  Warm (≥{_TIER_WARM})          : {len(warm):,}  — ส่ง campaign premium",
        f"  🧊 Cold (≥{_TIER_COLD})          : {len(cold):,}  — nurture sequence",
        f"  ⬜ Unfit (<{_TIER_COLD})         : {len(unfit):,}  — Do Nothing",
    ]

    priority = sorted(
        hot + warm,
        key=lambda r: _get(r, "lead_score", "score", 0),
        reverse=True,
    )[:top_n]

    if priority:
        lines.append("")
        lines.append(f"  Top {len(priority)} leads ที่ควรติดต่อก่อน:")
        for r in priority:
            cid    = r.get("customer_id", "")
            score  = _get(r, "lead_score", "score", 0)
            tier   = _get(r, "lead_tier",  "tier",  "")
            sigs   = r.get("lead_signals", r.get("signals", []))
            sig_str = ", ".join(sigs) if sigs else "—"
            lines.append(f"    {str(cid):20}  score={score:3d}  [{tier}]  signals: {sig_str}")

    return "\n".join(lines)


def get_priority_leads(
    df_or_results,
    tier: str = "hot",
    min_score: int = 0,
) -> list[dict]:
    """Return leads filtered by tier and minimum score, sorted descending.

    Parameters
    ----------
    df_or_results : pd.DataFrame | list[dict]
    tier : str
        "hot" | "warm" | "cold" | "" (empty = all tiers).
    min_score : int
        Additional minimum score filter on top of tier.

    Returns
    -------
    list[dict]
        Sorted descending by score.
    """
    try:
        import pandas as pd
        if isinstance(df_or_results, pd.DataFrame):
            records = df_or_results.to_dict("records")
        else:
            records = list(df_or_results)
    except ImportError:
        records = list(df_or_results)

    def _get(r, key, fallback_key, default):
        return r.get(key, r.get(fallback_key, default))

    filtered = [
        r for r in records
        if (not tier or _get(r, "lead_tier", "tier", "") == tier)
        and _get(r, "lead_score", "score", 0) >= min_score
    ]
    return sorted(filtered, key=lambda r: _get(r, "lead_score", "score", 0), reverse=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tag_matches(tags: list[str], prefix: str) -> bool:
    """Return True if any tag equals or starts with the given prefix."""
    for t in tags:
        if t == prefix or t.startswith(prefix + "_") or t.startswith(prefix + ":"):
            return True
    return False


def _score_to_tier(score: int) -> str:
    if score >= _TIER_HOT:
        return "hot"
    if score >= _TIER_WARM:
        return "warm"
    if score >= _TIER_COLD:
        return "cold"
    return "unfit"
