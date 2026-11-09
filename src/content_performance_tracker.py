"""content_performance_tracker.py — Meta ad performance feedback loop.

Closes the loop between content generation and real-world ad performance:

  Step 1: record_content()         — store ContentOutput metadata to content_registry.jsonl
  Step 2: link_content_to_ad()     — pair content_id with Meta ad_id after execution
  Step 3: track_meta_performance() — pull Meta ad insights → content_outcomes.jsonl
  Step 4: get_top_performers()     — surface best-performing examples by segment/action
  Step 5: enrich_brand_kit_examples() — auto-populate BrandKit.reference_examples

The end result: next time generate_content() is called with the enriched BrandKit,
Claude sees real performance data as few-shot examples and writes better copy.

Storage files (under data_dir/)
  content_registry.jsonl  — one record per generated content piece
  content_outcomes.jsonl  — performance metrics pulled from Meta (append-only)

Usage
-----
    from content_performance_tracker import (
        record_content, link_content_to_ad,
        track_meta_performance, get_top_performers,
        enrich_brand_kit_examples,
    )
    from content_generator import BrandKit

    # After generate_content():
    content_id = record_content(data_dir, content_output, plan_id="plan_abc", store_id="store_001")

    # After execute() returns ad_id:
    link_content_to_ad(data_dir, content_id, ad_id="1234567890")

    # Nightly job — pull Meta insights for all linked ads:
    result = track_meta_performance(data_dir)

    # Next content generation — inject top performers:
    kit = enrich_brand_kit_examples(data_dir, kit, segment="At-Risk", action="Win-Back")
    output = generate_content(brief, kit)
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

_logger = logging.getLogger(__name__)

_DEFAULT_TOP_N      = 3
_MIN_IMPRESSIONS    = 100   # ignore ads with too few impressions
_RANK_BY            = "ctr"  # ctr | conversions | clicks


# ---------------------------------------------------------------------------
# Step 1: Record generated content
# ---------------------------------------------------------------------------

def record_content(
    data_dir: str,
    content_output: Any,              # ContentOutput dataclass
    plan_id: str = "",
    store_id: str = "",
    execution_id: str = "",
) -> str:
    """Store a ContentOutput to content_registry.jsonl.

    Returns the generated content_id.
    """
    content_id = f"content_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    record = {
        "content_id":   content_id,
        "plan_id":      plan_id,
        "store_id":     store_id,
        "execution_id": execution_id,
        "mode":         getattr(content_output, "mode",    ""),
        "segment":      getattr(content_output, "segment", ""),
        "action":       getattr(content_output, "action",  ""),
        "channel":      getattr(content_output, "channel", "meta"),
        "headline":     getattr(content_output, "headline", ""),
        "body":         getattr(content_output, "body",     ""),
        "cta":          getattr(content_output, "cta",      ""),
        "emotion":      getattr(content_output, "emotion",  ""),
        "tone_used":    getattr(content_output, "tone_used", ""),
        "key_message":  getattr(content_output, "key_message", ""),
        "meta_primary_text": getattr(content_output, "meta_primary_text", ""),
        "ad_id":        None,    # filled by link_content_to_ad()
        "created_at":   datetime.now(timezone.utc).isoformat(),
    }

    _append_jsonl(data_dir, "content_registry.jsonl", record)
    _logger.info("content_performance_tracker: recorded content %s (segment=%s, action=%s)",
                 content_id, record["segment"], record["action"])
    return content_id


# ---------------------------------------------------------------------------
# Step 2: Link content to Meta ad_id after execution
# ---------------------------------------------------------------------------

def link_content_to_ad(data_dir: str, content_id: str, ad_id: str) -> bool:
    """Update content_registry.jsonl: set ad_id on the matching content_id.

    Returns True if the record was found and updated, False otherwise.
    """
    path = os.path.join(data_dir, "content_registry.jsonl")
    if not os.path.exists(path):
        _logger.warning("link_content_to_ad: content_registry.jsonl not found")
        return False

    lines: list[str] = []
    found = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            try:
                rec = json.loads(stripped)
                if rec.get("content_id") == content_id and rec.get("ad_id") is None:
                    rec["ad_id"]     = ad_id
                    rec["linked_at"] = datetime.now(timezone.utc).isoformat()
                    lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
                    found = True
                else:
                    lines.append(line)
            except json.JSONDecodeError:
                lines.append(line)

    if found:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        _logger.info("link_content_to_ad: linked %s → ad_id %s", content_id, ad_id)

    return found


# ---------------------------------------------------------------------------
# Step 3: Pull Meta insights → content_outcomes.jsonl
# ---------------------------------------------------------------------------

def track_meta_performance(
    data_dir: str,
    date_preset: str = "last_30d",
    dry_run: bool = False,
) -> dict:
    """Pull Meta ad insights for all content records that have an ad_id.

    Skips records already tracked (content_id present in content_outcomes.jsonl).
    Skips ads with fewer than _MIN_IMPRESSIONS (too early / low signal).

    Returns
    -------
    dict
        ``{tracked, skipped_already_done, skipped_low_signal, failed}``
    """
    registry = _load_registry(data_dir)
    already_tracked = _load_tracked_ids(data_dir)

    tracked            = 0
    skipped_done       = 0
    skipped_low_signal = 0
    failed             = 0

    # Lazy import to avoid hard dependency
    meta_client = None
    if not dry_run:
        try:
            import sys
            src = os.path.dirname(__file__)
            if src not in sys.path:
                sys.path.insert(0, src)
            from meta_ads_client import MetaAdsClient
            meta_client = MetaAdsClient()
        except Exception as exc:
            _logger.error("track_meta_performance: cannot init MetaAdsClient — %s", exc)

    for rec in registry:
        ad_id      = rec.get("ad_id")
        content_id = rec.get("content_id", "")

        if not ad_id:
            continue

        if content_id in already_tracked:
            skipped_done += 1
            continue

        if dry_run:
            outcome = _make_outcome(rec, {
                "impressions": 1000, "clicks": 32, "ctr": 3.2,
                "conversions": 5, "spend_thb": 120.0,
            })
            _append_jsonl(data_dir, "content_outcomes.jsonl", outcome)
            tracked += 1
            continue

        if meta_client is None:
            failed += 1
            continue

        try:
            insights = meta_client.get_ad_insights(ad_id, date_preset=date_preset)
        except Exception as exc:
            _logger.warning("track_meta_performance: get_ad_insights(%s) failed — %s", ad_id, exc)
            failed += 1
            continue

        if insights.get("impressions", 0) < _MIN_IMPRESSIONS:
            _logger.debug("track_meta_performance: ad %s has only %d impressions — skip",
                          ad_id, insights.get("impressions", 0))
            skipped_low_signal += 1
            continue

        outcome = _make_outcome(rec, insights)
        _append_jsonl(data_dir, "content_outcomes.jsonl", outcome)
        tracked += 1
        _logger.info(
            "track_meta_performance: tracked %s — ctr=%.2f%% conversions=%d",
            content_id, outcome.get("ctr", 0), outcome.get("conversions", 0),
        )

    return {
        "tracked":            tracked,
        "skipped_already_done": skipped_done,
        "skipped_low_signal": skipped_low_signal,
        "failed":             failed,
    }


# ---------------------------------------------------------------------------
# Step 4: Surface top performers
# ---------------------------------------------------------------------------

def get_top_performers(
    data_dir: str,
    segment: str = "",
    action: str = "",
    top_n: int = _DEFAULT_TOP_N,
    rank_by: str = _RANK_BY,
) -> list[dict]:
    """Return top-performing content examples filtered by segment/action.

    Parameters
    ----------
    data_dir : str
        Root data directory.
    segment : str, optional
        Filter by segment name (e.g. "At-Risk"). Empty string = all segments.
    action : str, optional
        Filter by action type (e.g. "Win-Back"). Empty string = all actions.
    top_n : int
        Maximum number of examples to return.
    rank_by : str
        Metric to rank by: ``"ctr"`` | ``"conversions"`` | ``"clicks"``.

    Returns
    -------
    list[dict]
        Each dict: ``{headline, cta, emotion, ctr, conversions, segment, action}``
    """
    outcomes = _load_outcomes(data_dir)

    if segment:
        segment_lower = segment.lower()
        outcomes = [o for o in outcomes if segment_lower in o.get("segment", "").lower()]

    if action:
        action_lower = action.lower()
        outcomes = [o for o in outcomes if action_lower in o.get("action", "").lower()]

    # Filter out low-signal results
    outcomes = [o for o in outcomes if o.get("impressions", 0) >= _MIN_IMPRESSIONS]

    # Sort descending by rank_by metric
    outcomes.sort(key=lambda x: x.get(rank_by, 0), reverse=True)

    return [
        {
            "headline":    o.get("headline",    ""),
            "cta":         o.get("cta",          ""),
            "emotion":     o.get("emotion",      ""),
            "key_message": o.get("key_message",  ""),
            "ctr":         o.get("ctr",          0.0),
            "conversions": o.get("conversions",  0),
            "segment":     o.get("segment",      ""),
            "action":      o.get("action",       ""),
        }
        for o in outcomes[:top_n]
    ]


# ---------------------------------------------------------------------------
# Step 5: Inject into BrandKit
# ---------------------------------------------------------------------------

def enrich_brand_kit_examples(
    data_dir: str,
    brand_kit: Any,
    segment: str = "",
    action: str = "",
    top_n: int = _DEFAULT_TOP_N,
) -> Any:
    """Return a copy of brand_kit with reference_examples populated from top performers.

    If no performance data is available, returns brand_kit unchanged.

    The examples are formatted as:
      "headline: ... | cta: ... | emotion: ... (CTR X.X%)"

    so Claude can learn what phrasing actually worked for this segment/action.
    """
    import dataclasses
    top = get_top_performers(data_dir, segment=segment, action=action, top_n=top_n)
    if not top:
        return brand_kit

    examples = [
        f"headline: {ex['headline']} | cta: {ex['cta']} | emotion: {ex['emotion']} (CTR {ex['ctr']:.1f}%)"
        for ex in top
        if ex.get("headline")
    ]
    if not examples:
        return brand_kit

    # Merge with existing reference_examples (deduplicate)
    existing  = list(getattr(brand_kit, "reference_examples", []))
    combined  = existing + [e for e in examples if e not in existing]

    return dataclasses.replace(brand_kit, reference_examples=combined)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _append_jsonl(data_dir: str, filename: str, record: dict) -> None:
    path = os.path.join(data_dir, filename)
    os.makedirs(data_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_registry(data_dir: str) -> list[dict]:
    return _read_jsonl(os.path.join(data_dir, "content_registry.jsonl"))


def _load_outcomes(data_dir: str) -> list[dict]:
    return _read_jsonl(os.path.join(data_dir, "content_outcomes.jsonl"))


def _load_tracked_ids(data_dir: str) -> set[str]:
    return {o["content_id"] for o in _load_outcomes(data_dir) if o.get("content_id")}


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _make_outcome(registry_rec: dict, insights: dict) -> dict:
    return {
        "content_id":   registry_rec.get("content_id", ""),
        "plan_id":      registry_rec.get("plan_id",    ""),
        "store_id":     registry_rec.get("store_id",   ""),
        "execution_id": registry_rec.get("execution_id", ""),
        "ad_id":        registry_rec.get("ad_id",      ""),
        "segment":      registry_rec.get("segment",    ""),
        "action":       registry_rec.get("action",     ""),
        "mode":         registry_rec.get("mode",       ""),
        "channel":      registry_rec.get("channel",    "meta"),
        "headline":     registry_rec.get("headline",   ""),
        "cta":          registry_rec.get("cta",        ""),
        "emotion":      registry_rec.get("emotion",    ""),
        "key_message":  registry_rec.get("key_message", ""),
        "impressions":  insights.get("impressions", 0),
        "clicks":       insights.get("clicks",       0),
        "ctr":          insights.get("ctr",           0.0),
        "conversions":  insights.get("conversions",   0),
        "spend_thb":    insights.get("spend_thb",     0.0),
        "tracked_at":   datetime.now(timezone.utc).isoformat(),
    }
