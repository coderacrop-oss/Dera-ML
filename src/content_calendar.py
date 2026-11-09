"""content_calendar.py — Monthly content calendar planner.

Applies a store-configured content mix (e.g. 30% knowledge / 60% story / 10% sales)
to produce a monthly content plan with assigned slots and content type suggestions.

The three categories:
  knowledge  — educational, informational (new_menu tips, service_highlight)
  story      — brand story, social proof, seasonal engagement
  sales      — ML-driven promo content (from campaign_planner output)

ML-driven promo actions are assigned to "sales" slots first.
Any actions exceeding the sales quota are returned as overflow (defer to next month).

Usage
-----
    from content_calendar import plan_content_calendar, ContentMixConfig
    from content_generator import BrandKit

    mix = ContentMixConfig(knowledge=0.30, story=0.60, sales=0.10, pieces_per_month=12)
    kit = BrandKit(brand_name="คาเฟ่ริมน้ำ", category="food_and_beverage")

    plan = plan_content_calendar(mix, kit, campaign_actions=[...], month="2026-11")

    for slot in plan.slots:
        print(f"Week {slot.week} | {slot.category:10} | {slot.content_type:20} | {slot.topic_suggestion}")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ContentMixConfig:
    """Store-configured content mix ratios for one month.

    Parameters
    ----------
    knowledge : float
        Fraction of content that is educational / informational.
    story : float
        Fraction of content that is brand story / social proof / engagement.
    sales : float
        Fraction of content that is promotional / ML-driven.
    pieces_per_month : int
        Total content pieces planned for the month.

    All three fractions must sum to 1.0 (±0.01 tolerance).
    """
    knowledge:        float = 0.30
    story:            float = 0.60
    sales:            float = 0.10
    pieces_per_month: int   = 12

    def __post_init__(self) -> None:
        total = self.knowledge + self.story + self.sales
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Content mix ratios must sum to 1.0, got {total:.2f} "
                f"(knowledge={self.knowledge}, story={self.story}, sales={self.sales})"
            )
        if self.pieces_per_month < 1:
            raise ValueError("pieces_per_month must be >= 1")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContentSlot:
    """One planned content piece in the monthly calendar."""
    slot_number:      int         # 1-based position
    week:             int         # 1–4
    day_of_week:      str         # suggested publishing day
    category:         str         # knowledge | story | sales
    content_type:     str         # new_menu | service_highlight | brand_story |
                                  # bestseller | seasonal | ml_promo
    topic_suggestion: str         # suggested topic / focus for this slot
    brief:            dict | None = None   # pre-filled brief for ML-driven slots
    status:           str = "planned"      # planned | overflow


@dataclass
class ContentCalendarPlan:
    """Monthly content calendar with all slots and overflow."""
    month:            str
    total_pieces:     int
    knowledge_count:  int
    story_count:      int
    sales_count:      int
    overflow_actions: list[dict]   # ML actions deferred to next month
    slots:            list[ContentSlot]
    mix_config:       ContentMixConfig


# ---------------------------------------------------------------------------
# Content type pools per category
# ---------------------------------------------------------------------------

# (content_type, topic_suggestion_template)
_KNOWLEDGE_POOL = [
    ("service_highlight", "แนะนำบริการ / สิ่งที่ลูกค้าอาจยังไม่รู้เกี่ยวกับร้าน"),
    ("new_menu",          "แนะนำสินค้า / เมนู หรือ how-to เพื่อให้ลูกค้าได้ประโยชน์สูงสุด"),
    ("service_highlight", "tips & tricks จากร้าน — ความรู้ที่มีประโยชน์สำหรับลูกค้า"),
    ("new_menu",          "highlight จุดเด่นของสินค้า / วัตถุดิบ / กระบวนการที่น่าสนใจ"),
]

_STORY_POOL = [
    ("brand_story", "เล่าเรื่องร้าน / behind the scenes / ค่านิยม"),
    ("bestseller",  "สินค้าขายดี — social proof + ทำไมลูกค้าถึงชอบ"),
    ("seasonal",    "content ตาม season / เทศกาล / moment ที่กำลังมาถึง"),
    ("brand_story", "ทีมงาน / ความตั้งใจ / เหตุผลที่เปิดร้าน"),
    ("bestseller",  "customer story / รีวิวจริง + ภาพประกอบ"),
    ("seasonal",    "เชื่อม brand กับ moment ของลูกค้า — ไม่ต้องมีโปร"),
]

_DAY_ROTATION = ["Mon", "Wed", "Fri", "Tue", "Thu", "Mon", "Wed", "Fri",
                 "Tue", "Thu", "Sat", "Mon"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_content_calendar(
    mix: ContentMixConfig,
    brand_kit: Any,
    campaign_actions: list[dict] | None = None,
    month: str | None = None,
) -> ContentCalendarPlan:
    """Build a monthly content calendar from mix config and ML campaign actions.

    Parameters
    ----------
    mix : ContentMixConfig
        Store-configured content mix ratios and total piece count.
    brand_kit : BrandKit
        Store's brand identity (used for topic suggestions).
    campaign_actions : list[dict], optional
        ML-driven promo actions from campaign_planner output.
        Each dict: ``{segment, action, promo_type, top_product, ...}``
        Assigned to sales slots first; excess returned as overflow.
    month : str, optional
        Calendar month label e.g. ``"2026-11"``.
        Defaults to current UTC month.

    Returns
    -------
    ContentCalendarPlan
        Fully populated calendar with ContentSlot list and overflow actions.
    """
    from datetime import datetime, timezone

    if month is None:
        month = datetime.now(timezone.utc).strftime("%Y-%m")

    n             = mix.pieces_per_month
    knowledge_cnt = _round_slot(n, mix.knowledge)
    sales_cnt     = _round_slot(n, mix.sales)
    story_cnt     = n - knowledge_cnt - sales_cnt   # absorb rounding remainder

    # Assign ML actions to sales slots; collect overflow
    actions  = list(campaign_actions or [])
    overflow = actions[sales_cnt:]
    sales_actions = actions[:sales_cnt]              # may be shorter than sales_cnt

    # Build category queues
    knowledge_q = [_KNOWLEDGE_POOL[i % len(_KNOWLEDGE_POOL)] for i in range(knowledge_cnt)]
    story_q     = [_STORY_POOL[i % len(_STORY_POOL)]         for i in range(story_cnt)]
    sales_q     = [
        ("ml_promo", _sales_topic(sales_actions[i] if i < len(sales_actions) else None))
        for i in range(sales_cnt)
    ]

    # Interleave so each week gets a category mix
    ordered = _interleave(knowledge_q, story_q, sales_q)

    # Build ContentSlot list
    slots: list[ContentSlot] = []
    sales_idx = 0
    for idx, (content_type, topic) in enumerate(ordered, start=1):
        category = _content_type_to_category(content_type)
        week     = _week_for_slot(idx, n)
        day      = _DAY_ROTATION[(idx - 1) % len(_DAY_ROTATION)]

        brief: dict | None = None
        if category == "sales":
            action = sales_actions[sales_idx] if sales_idx < len(sales_actions) else None
            brief  = _make_brief_dict(action) if action else None
            sales_idx += 1

        slots.append(ContentSlot(
            slot_number      = idx,
            week             = week,
            day_of_week      = day,
            category         = category,
            content_type     = content_type,
            topic_suggestion = topic,
            brief            = brief,
            status           = "planned",
        ))

    return ContentCalendarPlan(
        month            = month,
        total_pieces     = n,
        knowledge_count  = knowledge_cnt,
        story_count      = story_cnt,
        sales_count      = sales_cnt,
        overflow_actions = overflow,
        slots            = slots,
        mix_config       = mix,
    )


def summarise_calendar(plan: ContentCalendarPlan) -> str:
    """Return a human-readable Thai summary of the calendar plan."""
    lines = [
        f"แผนคอนเทนท์ {plan.month}  ({plan.total_pieces} ชิ้น)",
        f"  ความรู้  : {plan.knowledge_count} ชิ้น  ({plan.mix_config.knowledge*100:.0f}%)",
        f"  เคส/Story: {plan.story_count} ชิ้น  ({plan.mix_config.story*100:.0f}%)",
        f"  ขาย      : {plan.sales_count} ชิ้น  ({plan.mix_config.sales*100:.0f}%)",
    ]
    if plan.overflow_actions:
        lines.append(
            f"  ⚠️  โปรโมชั่นเกินโควต้า {len(plan.overflow_actions)} รายการ → เลื่อนไปเดือนหน้า"
        )
    lines.append("")
    for slot in plan.slots:
        ml_tag = " [ML]" if slot.brief else ""
        lines.append(
            f"  Week {slot.week} {slot.day_of_week:3}  "
            f"[{slot.category:9}] {slot.content_type:20} — {slot.topic_suggestion[:55]}{ml_tag}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round_slot(total: int, ratio: float) -> int:
    """Round ratio * total (minimum 0)."""
    return max(0, round(total * ratio))


def _week_for_slot(slot_number: int, total: int) -> int:
    """Assign slot to week 1–4 as evenly as possible."""
    per_week = total / 4.0
    return min(4, max(1, math.ceil(slot_number / per_week)))


def _content_type_to_category(content_type: str) -> str:
    if content_type == "ml_promo":
        return "sales"
    if content_type in ("new_menu", "service_highlight"):
        return "knowledge"
    return "story"


def _interleave(
    knowledge_q: list[tuple],
    story_q:     list[tuple],
    sales_q:     list[tuple],
) -> list[tuple]:
    """Interleave queues so each week gets a category mix.

    Pattern: story → knowledge → story → story → knowledge → story → sales → …
    ensuring story (largest pool) dominates but knowledge and sales are spread.
    """
    result: list[tuple] = []
    # Round-robin with story appearing twice per cycle to reflect higher weight
    cycle = ["story", "knowledge", "story", "sales"]
    pools = {"story": list(story_q), "knowledge": list(knowledge_q), "sales": list(sales_q)}
    total = sum(len(v) for v in pools.values())

    ci = 0
    while len(result) < total:
        cat = cycle[ci % len(cycle)]
        ci += 1
        if pools[cat]:
            result.append(pools[cat].pop(0))
        else:
            # Try next non-empty pool
            for fallback in ("story", "knowledge", "sales"):
                if pools[fallback]:
                    result.append(pools[fallback].pop(0))
                    break

    return result


def _sales_topic(action: dict | None) -> str:
    if action is None:
        return "โปรโมชั่น — ยังไม่มีแผน ML (จะอัปเดตเมื่อรันแผนแคมเปญ)"
    parts = [p for p in [
        action.get("segment", ""),
        action.get("action", ""),
        action.get("promo_type", ""),
        action.get("top_product", ""),
    ] if p]
    return " — ".join(parts) if parts else "โปรโมชั่น ML-driven"


def _make_brief_dict(action: dict) -> dict:
    return {
        "segment":      action.get("segment", ""),
        "action":       action.get("action", ""),
        "promo_type":   action.get("promo_type", ""),
        "top_product":  action.get("top_product", ""),
        "top_category": action.get("top_category", ""),
        "send_window":  action.get("send_window", ""),
        "promo_cost":   action.get("promo_cost", 0.0),
        "channel":      action.get("channel", "meta"),
    }


# ---------------------------------------------------------------------------
# CLI quick preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.replace("content_calendar.py", ""))

    from content_generator import BrandKit

    kit = BrandKit(brand_name="คาเฟ่ริมน้ำ", category="food_and_beverage")

    # Example: 30/60/10 mix, 12 pieces, 2 ML promo actions
    mix = ContentMixConfig(knowledge=0.30, story=0.60, sales=0.10, pieces_per_month=12)

    ml_actions = [
        {"segment": "At-Risk", "action": "Win-Back", "promo_type": "discount_15",
         "top_product": "Latte", "channel": "meta"},
        {"segment": "Champions", "action": "VIP Reward", "promo_type": "points_x2",
         "top_product": "Croissant", "channel": "line_oa"},
        {"segment": "Rescue", "action": "Rescue", "promo_type": "discount_25",
         "top_product": "Coffee", "channel": "meta"},  # will overflow
    ]

    plan = plan_content_calendar(mix, kit, campaign_actions=ml_actions, month="2026-11")
    print(summarise_calendar(plan))
