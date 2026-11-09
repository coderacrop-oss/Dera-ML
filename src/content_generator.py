"""Content Generator — AI-driven ad content pipeline.

Full pipeline per piece of content:
  Step 1: Strategy layer  — ML insight + Brand Kit → key message, tone, emotion
  Step 2: Art direction   — message → visual brief, composition, mood, color grading
  Step 3: Image prompt    — art direction → image generation prompt (pluggable API)

Supports 2 modes:
  Mode 1: ML-driven  — triggered by campaign decision engine, per-action
  Mode 2: Brand-driven — triggered by store, awareness content (no promo required)

Usage:
  from content_generator import generate_content, MLDrivenBrief, BrandDrivenBrief, BrandKit

  brand_kit = BrandKit(brand_name="คาเฟ่ริมน้ำ", category="food_and_beverage", tone="friendly")
  brief = MLDrivenBrief(segment="At-Risk", action="Win-Back", promo_type="discount_15",
                        top_product="Latte", top_category="Coffee", send_window="17:00-19:00")
  result = generate_content(brief, brand_kit)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

# Add src to path for sibling import
_SRC = os.path.dirname(__file__)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from visual_style_library import (  # noqa: E402
    VisualStyle,
    auto_select_style,
    build_image_prompt,
    get_style,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BrandKit:
    """Brand identity config — ร้านกรอกครั้งเดียวใน Platform settings."""
    brand_name: str
    category: Literal[
        "food_and_beverage", "retail", "beauty_and_health", "services", "other"
    ] = "food_and_beverage"
    tone: Literal["friendly", "formal", "playful", "premium", "warm"] = "friendly"
    primary_color: str = "#000000"
    secondary_color: str = "#FFFFFF"
    photo_style: str = "warm lifestyle, natural light"       # free text — described by store
    forbidden_words: list[str] = field(default_factory=list)
    preferred_words: list[str] = field(default_factory=list)
    reference_examples: list[str] = field(default_factory=list)  # approved copy ที่ผ่านมา


@dataclass
class MLDrivenBrief:
    """Input จาก ML pipeline — Mode 1: Promo content."""
    segment: str               # "At-Risk"
    action: str                # "Win-Back"
    promo_type: str            # "discount_15" / "points_x2" / "free_item" etc.
    top_product: str = ""      # "Latte"
    top_category: str = ""     # "Coffee"
    send_window: str = ""      # "17:00-19:00"
    promo_cost: float = 0.0    # บาท
    channel: Literal["meta", "line_oa", "email", "sms"] = "meta"


@dataclass
class BrandDrivenBrief:
    """Input จากร้าน — Mode 2: Awareness / Brand content."""
    content_type: Literal[
        "new_menu", "service_highlight", "brand_story", "seasonal", "bestseller"
    ]
    topic: str                  # "เมนูใหม่ Matcha Latte" / "ต้อนรับปีใหม่" etc.
    top_product: str = ""
    top_category: str = ""
    season_or_event: str = ""   # "ปีใหม่ 2027" / "วาเลนไทน์" etc.
    channel: Literal["meta", "line_oa", "email", "sms"] = "meta"


@dataclass
class ContentOutput:
    """Output ครบ 1 ชิ้น — พร้อม approve และส่ง image API."""
    mode: str                   # "ml_driven" / "brand_driven"
    channel: str
    segment: str                # ว่างสำหรับ brand_driven
    action: str                 # ว่างสำหรับ brand_driven

    # Step 1: Strategy
    key_message: str = ""
    emotion: str = ""           # "urgency" / "warmth" / "exclusivity" / "curiosity"
    tone_used: str = ""

    # Step 2: Art direction
    visual_concept: str = ""    # อธิบาย concept ภาพ 1-2 ประโยค
    composition: str = ""       # "hero center" / "split left-right" / "lifestyle scene"
    mood: str = ""              # "warm golden hour" / "clean minimal" / "vibrant energetic"
    color_direction: str = ""   # สีที่ควรโดดเด่น + mood ของ palette
    hero_subject: str = ""      # สิ่งที่เป็น focal point ของภาพ

    # Step 3: Copy + image prompt
    headline: str = ""
    body: str = ""
    cta: str = ""
    meta_primary_text: str = "" # Meta-specific copy (longer, storytelling)
    image_prompt: str = ""      # prompt สำหรับ image generation API (from Style Library)
    image_negative_prompt: str = ""  # negative prompt สำหรับ image API

    # Visual style metadata
    style_key: str = ""         # e.g. "warm_hero", "bold_promo"
    style_name_th: str = ""     # e.g. "Hero อบอุ่น"
    composition_zone: str = ""  # e.g. "bottom_third"
    overlay_bg: str = ""        # e.g. "dark_gradient"
    badge_position: str = ""    # e.g. "top_right"
    font_weight: str = ""       # e.g. "bold"
    aspect_ratios: list[str] = field(default_factory=list)

    # Composed image paths (populated if generate_image=True)
    composed_paths: dict[str, str] = field(default_factory=dict)

    # Metadata
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Action pattern → tone & strategy mapping
# ---------------------------------------------------------------------------

_ACTION_PATTERNS: dict[str, dict] = {
    "win_back": {
        "matches": ["win-back", "win back", "winback"],
        "emotion": "nostalgia + urgency",
        "strategy": "เน้น empathy ก่อน — ร้านคิดถึง ตามด้วย offer ที่ชัดเจนและหมดอายุ",
        "copy_guide": "ใช้ 'เราคิดถึงคุณ' / 'กลับมาหาเราสิ' + ระบุ offer + deadline",
        "visual_mood": "warm, inviting, familiar — ไม่ใช่ aggressive sale",
    },
    "vip_reward": {
        "matches": ["vip reward", "vip", "reward", "แต้ม", "points"],
        "emotion": "exclusivity + appreciation",
        "strategy": "ห้ามใช้คำ 'ลด/โปร' — เป็น VIP experience เฉพาะคุณเท่านั้น",
        "copy_guide": "ใช้ 'สิทธิพิเศษสำหรับคุณ' / 'ก่อนใคร' / 'ขอบคุณที่ไว้วางใจ'",
        "visual_mood": "premium, clean, elevated — gold tones หรือ brand primary ที่สง่า",
    },
    "reactivation": {
        "matches": ["reactivation", "reactivate", "กลับมา", "nudge"],
        "emotion": "curiosity + strong incentive",
        "strategy": "emotional hook ก่อน — บอกว่าอะไรใหม่ที่ร้าน จากนั้น incentive แรง",
        "copy_guide": "เปิดด้วยสิ่งใหม่ที่น่าสนใจ แล้วค่อยบอก offer ที่แรงกว่าปกติ",
        "visual_mood": "fresh, exciting, 'มีอะไรใหม่' — สีสดใส มีพลัง",
    },
    "early_winback": {
        "matches": ["early win-back", "early winback", "churn risk"],
        "emotion": "gentle urgency",
        "strategy": "อ่อนโยนกว่า win-back ปกติ — เพิ่งเริ่มห่าง ยังไม่ต้อง aggressive",
        "copy_guide": "ใช้ 'นานไม่เจอกัน' + เสนอ offer เบาๆ + อย่าลืมบอก deadline",
        "visual_mood": "casual, friendly, approachable — ไม่เครียด",
    },
    "milestone_reward": {
        "matches": ["milestone", "แต้มสะสม", "สั่งครบ"],
        "emotion": "achievement + delight",
        "strategy": "celebrate ความสำเร็จของลูกค้า — progress framing",
        "copy_guide": "บอก milestone ที่ทำสำเร็จ + reward ที่ได้ ใช้ 'ยินดีด้วย!' / 'คุณทำได้แล้ว'",
        "visual_mood": "celebratory, bright, joyful — confetti / achievement feel",
    },
    "welcome": {
        "matches": ["welcome", "ยินดีต้อนรับ", "new customer"],
        "emotion": "warmth + invitation",
        "strategy": "ต้อนรับอบอุ่น — แนะนำตัวร้าน + offer พิเศษสำหรับลูกค้าใหม่",
        "copy_guide": "ใช้ 'ยินดีต้อนรับ' + บอก brand personality + offer ที่ทำให้อยากกลับมา",
        "visual_mood": "welcoming, bright, warm smile — first impression ที่ดี",
    },
    "upsell_nudge": {
        "matches": ["upsell", "bundle", "upgrade", "potential loyalist"],
        "emotion": "aspiration + progress",
        "strategy": "progress framing — 'คุณอยู่ใกล้แล้ว' + bundle deal เล็กน้อย",
        "copy_guide": "ใช้ progress bar concept / 'อีกนิดเดียว' / bundle value proposition",
        "visual_mood": "aspirational, clean, forward-looking",
    },
    "zero_cost": {
        "matches": ["zero cost", "zero-cost", "notification", "แจ้งเตือน"],
        "emotion": "value reminder",
        "strategy": "ไม่มี promo — remind ลูกค้า high-value ว่าร้านยังอยู่ + value ที่ได้รับ",
        "copy_guide": "เน้น brand value / seasonal relevance / ไม่ต้อง push offer",
        "visual_mood": "subtle, premium, brand-centric",
    },
    "rescue": {
        "matches": ["rescue", "can't lose", "cant lose", "aggressive"],
        "emotion": "urgency + high value",
        "strategy": "offer แรงที่สุด — นี่คือ last chance ก่อนเสียลูกค้าถาวร",
        "copy_guide": "ใช้ urgency + offer ที่ดีที่สุดที่เราให้ได้ + deadline ชัดเจน",
        "visual_mood": "bold, high contrast, urgent — but not spammy",
    },
    "default": {
        "matches": [],
        "emotion": "friendly engagement",
        "strategy": "general engagement — remind ลูกค้าเกี่ยวกับ brand + offer",
        "copy_guide": "friendly tone, clear offer, simple CTA",
        "visual_mood": "clean, on-brand, inviting",
    },
}

_BRAND_CONTENT_TYPES: dict[str, dict] = {
    "new_menu": {
        "emotion": "excitement + curiosity",
        "strategy": "reveal สิ่งใหม่ — สร้าง FOMO อยากลอง",
        "copy_guide": "ใช้ 'ใหม่ล่าสุด!' / 'มาแล้ว!' + บอก key feature ของเมนู + invite to try",
        "visual_mood": "fresh, vibrant, appetite-stimulating — hero shot ของ product ใหม่",
    },
    "service_highlight": {
        "emotion": "trust + value",
        "strategy": "educate ลูกค้าเกี่ยวกับบริการ — สร้าง awareness และ consideration",
        "copy_guide": "บอก benefit ชัดเจน + ทำไมต้องเลือกเรา + simple CTA",
        "visual_mood": "clean, professional, trustworthy",
    },
    "brand_story": {
        "emotion": "connection + authenticity",
        "strategy": "build emotional connection — เล่าเรื่องร้านที่ทำให้คนรู้สึกว่า brand มีตัวตน",
        "copy_guide": "เล่าเรื่องจริงสั้นๆ ที่ relatable + human touch + invite engagement",
        "visual_mood": "authentic, behind-the-scenes, warm and human",
    },
    "seasonal": {
        "emotion": "festive + relevance",
        "strategy": "tie brand กับ moment ที่ทุกคนรู้สึกร่วมกัน — timely และ celebratory",
        "copy_guide": "tie เทศกาล + brand naturally + offer หรือ greeting + seasonal visual",
        "visual_mood": "festive, seasonal colors, celebratory atmosphere",
    },
    "bestseller": {
        "emotion": "social proof + confidence",
        "strategy": "social proof — 'คนอื่นเลือกแล้ว' ทำให้ลูกค้าใหม่กล้าลอง",
        "copy_guide": "ใช้ 'ขายดีที่สุด' / 'ลูกค้าชื่นชอบ' + ทำไมถึง popular + simple CTA",
        "visual_mood": "confident, popular, desirable — hero product shot",
    },
}


def _match_action_pattern(action: str) -> dict:
    """Match action string to pattern. Returns pattern dict."""
    action_lower = action.lower()
    for pattern_key, pattern in _ACTION_PATTERNS.items():
        if pattern_key == "default":
            continue
        for kw in pattern["matches"]:
            if kw in action_lower:
                return pattern
    return _ACTION_PATTERNS["default"]


def _promo_label(promo_type: str) -> str:
    """Convert promo_type code to Thai label."""
    mapping = {
        "discount_10": "ส่วนลด 10%",
        "discount_15": "ส่วนลด 15%",
        "discount_20": "ส่วนลด 20%",
        "discount_25": "ส่วนลด 25%",
        "points_x2":   "แต้มสะสม 2 เท่า",
        "points_50":   "แต้ม 50 คะแนน",
        "free_item":   "ของแถมฟรี",
        "free_drink":  "เครื่องดื่มฟรี 1 แก้ว",
        "bundle":      "ซื้อ 2 แถม 1",
        "cashback":    "คืนเงิน 10%",
    }
    return mapping.get(promo_type, promo_type.replace("_", " "))


def _build_forbidden_instruction(brand_kit: BrandKit) -> str:
    if not brand_kit.forbidden_words:
        return ""
    words = ", ".join(f'"{w}"' for w in brand_kit.forbidden_words)
    return f"ห้ามใช้คำเหล่านี้เด็ดขาด: {words}"


def _build_preferred_instruction(brand_kit: BrandKit) -> str:
    if not brand_kit.preferred_words:
        return ""
    words = ", ".join(f'"{w}"' for w in brand_kit.preferred_words)
    return f"คำที่ร้านชอบใช้: {words}"


def _build_reference_instruction(brand_kit: BrandKit) -> str:
    if not brand_kit.reference_examples:
        return ""
    examples = "\n".join(f"  - {ex}" for ex in brand_kit.reference_examples[:3])
    return f"ตัวอย่าง copy ที่ร้านเคย approve:\n{examples}"


# ---------------------------------------------------------------------------
# Step 1 + 2: Strategy + Art Direction (single Claude call)
# ---------------------------------------------------------------------------

def _brand_color_desc(brand_kit: BrandKit) -> str:
    """Convert hex primary color to descriptive phrase for image prompts."""
    hex_to_desc = {
        "#": "brand color",
        "#8B4513": "warm saddle brown",
        "#FF6B35": "vibrant coral orange",
        "#2E8B57": "rich forest green",
        "#1E3A5F": "deep navy blue",
        "#C0392B": "bold red",
        "#F39C12": "warm golden yellow",
        "#6C3483": "deep purple",
        "#000000": "clean black",
        "#FFFFFF": "pure white",
    }
    for k, v in hex_to_desc.items():
        if brand_kit.primary_color.upper().startswith(k.upper()) and k != "#":
            return v
    return f"{brand_kit.primary_color} brand color tones"


def _build_style_instruction(
    style: VisualStyle | None,
    product: str,
    brand_kit: BrandKit,
    season: str = "",
) -> str:
    """Build the style injection block for Claude prompt."""
    if not style:
        return (
            "## Image Prompt\n"
            "สำหรับ image_prompt: เขียน detailed English prompt สำหรับ AI image generation "
            "— include style, subject, lighting, composition, mood, colors. Do NOT include text/words in the image."
        )
    base_prompt, _ = build_image_prompt(
        style,
        product=product or "food",
        photo_style=brand_kit.photo_style,
        brand_color_desc=_brand_color_desc(brand_kit),
        season=season or "",
    )
    return (
        f"## Visual Style: /{style.key} — {style.name_th}\n"
        f"Mood: {style.mood}\n"
        f"Lighting: {style.lighting}\n"
        f"Color palette: {style.color_palette}\n"
        f"Composition zone: {style.composition}\n\n"
        f"Base image prompt (customize สำหรับ product และ brand นี้):\n{base_prompt}\n\n"
        f"สำหรับ image_prompt ใน JSON: ใช้ prompt นี้เป็น base แล้วปรับ product/brand detail ให้ specific ขึ้น"
    )


def _build_ml_driven_prompt(brief: MLDrivenBrief, brand_kit: BrandKit, style: VisualStyle | None = None) -> tuple[str, str]:
    """Build system + user prompt for ML-driven mode."""
    pattern = _match_action_pattern(brief.action)
    promo_label = _promo_label(brief.promo_type)

    image_prompt_instruction = _build_style_instruction(
        style,
        product=brief.top_product or brief.top_category or "food",
        brand_kit=brand_kit,
    )

    system = (
        "You are a creative director specializing in Thai SME marketing. "
        "Given a customer segment insight and brand kit, you will:\n"
        "1. Define the communication strategy (key message, emotion, tone)\n"
        "2. Create a complete art direction brief (visual concept, composition, mood, colors, hero subject)\n"
        "3. Write all ad copy (headline, body, CTA) in Thai\n"
        "4. Refine the image generation prompt based on the visual style provided\n\n"
        "CRITICAL RULES:\n"
        "- Always follow the brand kit's tone and forbidden words exactly\n"
        "- Copy must be concise: headline ≤ 10 words, body ≤ 30 words, CTA ≤ 5 words\n"
        "- Output MUST be valid JSON only — no markdown, no explanation outside JSON"
    )

    user = f"""สร้าง ad content สำหรับ:

## Customer Insight
- Segment: {brief.segment}
- Action: {brief.action}
- Promo: {promo_label}
- สินค้าที่ชอบ: {brief.top_product or 'ไม่ระบุ'}
- Category: {brief.top_category or 'ไม่ระบุ'}
- เวลาที่เหมาะ: {brief.send_window or 'ไม่ระบุ'}
- ช่องทาง: {brief.channel}

## Action Strategy Guide
- Emotion target: {pattern['emotion']}
- Strategy: {pattern['strategy']}
- Copy guide: {pattern['copy_guide']}
- Visual mood: {pattern['visual_mood']}

## Brand Kit
- ชื่อร้าน: {brand_kit.brand_name}
- Category: {brand_kit.category}
- Tone: {brand_kit.tone}
- สีหลัก: {brand_kit.primary_color}
- Photo style ของร้าน: {brand_kit.photo_style}
- {_build_forbidden_instruction(brand_kit)}
- {_build_preferred_instruction(brand_kit)}
- {_build_reference_instruction(brand_kit)}

{image_prompt_instruction}

## Output Format (JSON เท่านั้น)
{{
  "key_message": "สิ่งที่ต้องการให้ลูกค้ารู้สึก/รู้/ทำ — 1 ประโยค",
  "emotion": "emotion หลักที่ต้องการ",
  "tone_used": "tone ที่ใช้จริง",
  "visual_concept": "concept ภาพ 1-2 ประโยค อธิบายว่าภาพนี้สื่อสารอะไร",
  "composition": "hero center / split / lifestyle scene / flat lay / etc.",
  "mood": "คำอธิบาย mood ของภาพ เช่น warm golden hour, clean minimal",
  "color_direction": "สีที่ควรโดดเด่น + mood ของ palette",
  "hero_subject": "สิ่งที่เป็น focal point หลักของภาพ",
  "headline": "หัวข้อโฆษณา (ภาษาไทย ≤ 10 คำ)",
  "body": "เนื้อหา (ภาษาไทย ≤ 30 คำ)",
  "cta": "call to action (ภาษาไทย ≤ 5 คำ)",
  "meta_primary_text": "copy สำหรับ Meta โดยเฉพาะ — storytelling style ยาวกว่า body ได้",
  "image_prompt": "refined English image generation prompt based on the style above — specific to this product and brand"
}}"""

    return system, user


def _build_brand_driven_prompt(brief: BrandDrivenBrief, brand_kit: BrandKit, style: VisualStyle | None = None) -> tuple[str, str]:
    """Build system + user prompt for Brand-driven mode."""
    content_config = _BRAND_CONTENT_TYPES.get(brief.content_type, _BRAND_CONTENT_TYPES["brand_story"])

    system = (
        "You are a creative director specializing in Thai SME marketing. "
        "Given a brand content brief and brand kit, you will:\n"
        "1. Define the communication strategy (key message, emotion, tone)\n"
        "2. Create a complete art direction brief (visual concept, composition, mood, colors, hero subject)\n"
        "3. Write all ad copy (headline, body, CTA) in Thai\n"
        "4. Write a detailed image generation prompt in English for an AI image API\n\n"
        "CRITICAL RULES:\n"
        "- This is awareness content, NOT a promo — do NOT invent discounts or offers unless specified\n"
        "- Always follow the brand kit's tone and forbidden words exactly\n"
        "- Output MUST be valid JSON only — no markdown, no explanation outside JSON"
    )

    user = f"""สร้าง brand content สำหรับ:

## Content Brief
- Type: {brief.content_type}
- Topic: {brief.topic}
- สินค้า/บริการ: {brief.top_product or 'ไม่ระบุ'}
- Category: {brief.top_category or 'ไม่ระบุ'}
- เทศกาล/Event: {brief.season_or_event or 'ไม่ระบุ'}
- ช่องทาง: {brief.channel}

## Content Strategy Guide
- Emotion target: {content_config['emotion']}
- Strategy: {content_config['strategy']}
- Copy guide: {content_config['copy_guide']}
- Visual mood: {content_config['visual_mood']}

## Brand Kit
- ชื่อร้าน: {brand_kit.brand_name}
- Category: {brand_kit.category}
- Tone: {brand_kit.tone}
- สีหลัก: {brand_kit.primary_color}
- Photo style ของร้าน: {brand_kit.photo_style}
- {_build_forbidden_instruction(brand_kit)}
- {_build_preferred_instruction(brand_kit)}
- {_build_reference_instruction(brand_kit)}

{_build_style_instruction(style, brief.top_product or brief.topic, brand_kit, brief.season_or_event)}

## Output Format (JSON เท่านั้น)
{{
  "key_message": "สิ่งที่ต้องการให้ลูกค้ารู้สึก/รู้/ทำ — 1 ประโยค",
  "emotion": "emotion หลักที่ต้องการ",
  "tone_used": "tone ที่ใช้จริง",
  "visual_concept": "concept ภาพ 1-2 ประโยค อธิบายว่าภาพนี้สื่อสารอะไร",
  "composition": "hero center / split / lifestyle scene / flat lay / etc.",
  "mood": "คำอธิบาย mood ของภาพ",
  "color_direction": "สีที่ควรโดดเด่น + mood ของ palette",
  "hero_subject": "สิ่งที่เป็น focal point หลักของภาพ",
  "headline": "หัวข้อ (ภาษาไทย ≤ 10 คำ)",
  "body": "เนื้อหา (ภาษาไทย ≤ 30 คำ)",
  "cta": "call to action (ภาษาไทย ≤ 5 คำ)",
  "meta_primary_text": "copy สำหรับ Meta — storytelling style",
  "image_prompt": "detailed English prompt for AI image generation — include style, subject, lighting, composition, mood, colors. Do NOT include text/words in the image."
}}"""

    return system, user


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def _call_claude(system: str, user: str, api_key: Optional[str] = None) -> dict:
    """Call Claude API and return parsed JSON output."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set — pass api_key or set env var")

    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude returned invalid JSON: {e}\nRaw output:\n{raw}")


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def generate_content(
    brief: MLDrivenBrief | BrandDrivenBrief,
    brand_kit: BrandKit,
    api_key: Optional[str] = None,
    style_key: Optional[str] = None,
    tier: str = "casual",
    generate_image: bool = False,
    gemini_api_key: Optional[str] = None,
    badge_text: str = "",
    output_dir: Optional[str] = None,
) -> ContentOutput:
    """Generate full ad content — strategy + art direction + copy + image prompt.

    Args:
        brief:           MLDrivenBrief (promo) or BrandDrivenBrief (awareness)
        brand_kit:       Store's brand identity config
        api_key:         Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
        style_key:       Visual style preset key e.g. "warm_hero", "bold_promo"
                         If None → auto-selected from action + tier + channel
        tier:            Brand tier "casual" / "cafe" / "premium" — affects style selection
        generate_image:  If True → call Gemini + compose ad images after copy generation
        gemini_api_key:  Gemini API key for image generation (falls back to GEMINI_API_KEY)
        badge_text:      Promo badge text e.g. "ลด 15%" — added to composed images
        output_dir:      If set → save composed images here (requires generate_image=True)

    Returns:
        ContentOutput with all fields populated.
        If generate_image=True, also saves composed images and sets output.composed_paths.
    """
    is_ml = isinstance(brief, MLDrivenBrief)
    channel = brief.channel

    # Resolve visual style
    if style_key:
        style = get_style(style_key)
    else:
        action_for_style = brief.action if is_ml else brief.content_type
        style = auto_select_style(action_for_style, tier=tier, channel=channel)

    # Build negative prompt from style
    _, neg_prompt = build_image_prompt(
        style,
        product=(brief.top_product if is_ml else brief.top_product) or "food",
        photo_style=brand_kit.photo_style,
        brand_color_desc=_brand_color_desc(brand_kit),
    )

    if is_ml:
        system, user = _build_ml_driven_prompt(brief, brand_kit, style=style)
        segment = brief.segment
        action = brief.action
        mode = "ml_driven"
    else:
        system, user = _build_brand_driven_prompt(brief, brand_kit, style=style)
        segment = ""
        action = brief.content_type
        mode = "brand_driven"

    result = _call_claude(system, user, api_key=api_key)

    output = ContentOutput(
        mode=mode,
        channel=channel,
        segment=segment,
        action=action,
        key_message=result.get("key_message", ""),
        emotion=result.get("emotion", ""),
        tone_used=result.get("tone_used", ""),
        visual_concept=result.get("visual_concept", ""),
        composition=result.get("composition", ""),
        mood=result.get("mood", ""),
        color_direction=result.get("color_direction", ""),
        hero_subject=result.get("hero_subject", ""),
        headline=result.get("headline", ""),
        body=result.get("body", ""),
        cta=result.get("cta", ""),
        meta_primary_text=result.get("meta_primary_text", ""),
        image_prompt=result.get("image_prompt", ""),
        image_negative_prompt=neg_prompt,
        # Style metadata
        style_key=style.key,
        style_name_th=style.name_th,
        composition_zone=style.composition,
        overlay_bg=style.overlay_bg,
        badge_position=style.badge_position,
        font_weight=style.font_weight,
        aspect_ratios=style.aspect_ratios,
    )

    # Optional image generation + composition
    if generate_image:
        from image_generator import generate_for_content
        from image_composer import compose_ad, save_composed_ads

        pil_image = generate_for_content(output, api_key=gemini_api_key)
        composed = compose_ad(pil_image, output, badge_text=badge_text)

        if output_dir:
            prefix = f"{output.action}__{output.style_key}".replace(" ", "_").replace("-", "_")
            output.composed_paths = save_composed_ads(composed, output_dir, prefix=prefix)

    return output


def generate_campaign_content(
    campaign_actions: list[dict],
    brand_kit: BrandKit,
    api_key: Optional[str] = None,
    channel: str = "meta",
) -> list[ContentOutput]:
    """Generate content for all actions in a campaign plan.

    Args:
        campaign_actions: list of dicts from campaign_planner output, each with:
                          segment, action, promo_type, top_product, top_category, send_window
        brand_kit:        Store's brand identity config
        api_key:          Anthropic API key
        channel:          Target channel for all content

    Returns:
        list of ContentOutput, one per action
    """
    results = []
    for act in campaign_actions:
        brief = MLDrivenBrief(
            segment=act.get("segment", ""),
            action=act.get("action", ""),
            promo_type=act.get("promo_type", ""),
            top_product=act.get("top_product", ""),
            top_category=act.get("top_category", ""),
            send_window=act.get("send_window", ""),
            promo_cost=act.get("promo_cost", 0.0),
            channel=channel,
        )
        output = generate_content(brief, brand_kit, api_key=api_key)
        results.append(output)
    return results


# ---------------------------------------------------------------------------
# CLI quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== Content Generator — Quick Test ===\n")

    kit = BrandKit(
        brand_name="คาเฟ่ริมน้ำ",
        category="food_and_beverage",
        tone="friendly",
        primary_color="#8B4513",
        photo_style="warm lifestyle, natural light, cozy café atmosphere",
        forbidden_words=["ถูก", "ลดราคา"],
        preferred_words=["สมาชิก", "พิเศษสำหรับคุณ"],
    )

    # --- Mode 1: ML-driven ---
    print("Mode 1: ML-driven (Win-Back)")
    ml_brief = MLDrivenBrief(
        segment="At-Risk",
        action="Win-Back",
        promo_type="discount_15",
        top_product="Latte",
        top_category="Coffee",
        send_window="17:00-19:00",
        channel="meta",
    )

    try:
        out = generate_content(ml_brief, kit)
        print(f"Key message:    {out.key_message}")
        print(f"Emotion:        {out.emotion}")
        print(f"Visual concept: {out.visual_concept}")
        print(f"Composition:    {out.composition}")
        print(f"Mood:           {out.mood}")
        print(f"Hero subject:   {out.hero_subject}")
        print(f"Headline:       {out.headline}")
        print(f"Body:           {out.body}")
        print(f"CTA:            {out.cta}")
        print(f"Image prompt:   {out.image_prompt[:120]}...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n---\n")

    # --- Mode 2: Brand-driven ---
    print("Mode 2: Brand-driven (New Menu)")
    brand_brief = BrandDrivenBrief(
        content_type="new_menu",
        topic="Matcha Latte สูตรใหม่",
        top_product="Matcha Latte",
        top_category="Coffee",
        channel="meta",
    )

    try:
        out2 = generate_content(brand_brief, kit)
        print(f"Key message:    {out2.key_message}")
        print(f"Visual concept: {out2.visual_concept}")
        print(f"Headline:       {out2.headline}")
        print(f"Body:           {out2.body}")
        print(f"CTA:            {out2.cta}")
        print(f"Image prompt:   {out2.image_prompt[:120]}...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n✅ Done")
