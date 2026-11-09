"""Visual Style Library — Ad Creative Presets for Thai SME F&B

Based on Pinterest reference research (144 screenshots, 24 topics).
Each style defines:
  - base_prompt: English prompt for image generation API (FLUX / GPT-image)
  - composition: layout zone rules for text overlay
  - mood / color / lighting guidance
  - campaign action mapping
  - tier suitability (casual / cafe / premium)

Usage:
    from visual_style_library import get_style, build_image_prompt

    style = get_style("warm_hero")
    prompt = build_image_prompt(style, product="Latte", brand_color="#8B4513", photo_style="natural light")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

Tier = Literal["casual", "cafe", "premium", "all"]
Channel = Literal["meta", "line_oa", "both"]
CompositionZone = Literal[
    "bottom_third",   # text band at bottom 30%
    "top_third",      # text band at top 30%
    "right_half",     # right 50% clear for text
    "left_half",      # left 50% clear for text
    "center_overlay", # text centered over image (needs dark/blur overlay)
    "full_bleed",     # image fills entirely — text floats with shadow
    "split_bottom",   # bottom 40% solid color band
]


@dataclass
class VisualStyle:
    key: str                        # unique identifier — used as /command
    name_th: str                    # Thai display name in UI
    description_th: str             # one-line description for UI tooltip

    # Image generation
    base_prompt: str                # English base prompt — slots: {product}, {brand_color_desc}, {photo_style}
    negative_prompt: str            # things to avoid

    # Composition / layout
    composition: CompositionZone    # where text overlay goes
    aspect_ratios: list[str]        # supported: "1:1", "4:5", "9:16"
    image_safe_zone: str            # description of where image can "breathe"

    # Mood & visual language
    lighting: str
    color_palette: str              # description — not hex
    mood: str
    visual_language: str            # what makes this style recognizable

    # Campaign mapping
    action_types: list[str]         # which campaign actions this fits
    tiers: list[Tier]               # which brand tiers this suits
    channels: list[Channel]

    # Overlay guidance (for Pillow/Canvas layer)
    overlay_bg: str                 # "none" / "dark_gradient" / "solid_band" / "blur_panel"
    badge_position: str             # "top_right" / "bottom_left" / "none"
    font_weight: Literal["light", "regular", "bold", "black"]


# ---------------------------------------------------------------------------
# Style definitions — 10 presets
# ---------------------------------------------------------------------------

STYLES: dict[str, VisualStyle] = {}


def _reg(s: VisualStyle) -> VisualStyle:
    STYLES[s.key] = s
    return s


# 1. warm_hero — ภาพอาหาร hero shot อบอุ่น (ใช้มากที่สุดในตลาดไทย)
_reg(VisualStyle(
    key="warm_hero",
    name_th="Hero อบอุ่น",
    description_th="ภาพอาหาร/เครื่องดื่ม hero shot แสงธรรมชาติ อบอุ่น เหมาะกับทุกร้าน",

    base_prompt=(
        "Professional food photography of {product}, hero shot at 45-degree angle, "
        "warm golden natural light from the left, shallow depth of field, "
        "bokeh background, {photo_style}, steam rising, appetizing close-up, "
        "wooden or stone surface, Thai restaurant atmosphere, "
        "commercial ad quality, 4K, no text, no watermark"
    ),
    negative_prompt="text, watermark, logo, dark, cold light, overhead shot, messy background, plastic",

    composition="bottom_third",
    aspect_ratios=["1:1", "4:5", "9:16"],
    image_safe_zone="Bottom 30% kept clear or naturally dark for text band",

    lighting="Warm golden hour, 45° side light, natural window light",
    color_palette="Warm amber, cream, brown, golden tones — appetite-stimulating",
    mood="Inviting, warm, authentic, appetite-stimulating",
    visual_language="Single hero food shot, shallow DOF bokeh, warm cast, real restaurant feel",

    action_types=["win_back", "early_winback", "welcome", "zero_cost", "bestseller"],
    tiers=["casual", "cafe"],
    channels=["meta", "line_oa", "both"],

    overlay_bg="dark_gradient",
    badge_position="top_right",
    font_weight="bold",
))


# 2. bold_promo — โปสเตอร์โปร bold typography + ภาพอาหาร (แบบ GRUB, Limited Time)
_reg(VisualStyle(
    key="bold_promo",
    name_th="โปรโมชั่น Bold",
    description_th="Typography ใหญ่ โดดเด่น + ภาพอาหาร — เหมาะกับโปรลดราคา ด่วน",

    base_prompt=(
        "Food advertisement poster, {product} as hero on the right half, "
        "left half reserved for large typography, "
        "bold graphic design style, vibrant {brand_color_desc} background, "
        "clean flat color zones, commercial poster quality, "
        "high contrast, energetic, no text in image, no watermark"
    ),
    negative_prompt="text, watermark, busy background, low contrast, dark moody, minimalist",

    composition="right_half",
    aspect_ratios=["1:1", "4:5", "9:16"],
    image_safe_zone="Right 55% for product, left 45% clear for large headline text",

    lighting="Bright studio light, even exposure, clean shadows",
    color_palette="Brand primary + high contrast accent — red/yellow/orange or brand color bold",
    mood="Energetic, urgent, promotional, eye-catching",
    visual_language="Split layout: graphic color left + product right, starburst/badge elements",

    action_types=["rescue", "reactivation", "win_back", "early_winback"],
    tiers=["casual"],
    channels=["meta", "both"],

    overlay_bg="solid_band",
    badge_position="top_right",
    font_weight="black",
))


# 3. flat_lay — มองจากบน จัดวาง props (แบบ flat lay photography)
_reg(VisualStyle(
    key="flat_lay",
    name_th="Flat Lay จากบน",
    description_th="มองจากด้านบน จัดวาง props สวยงาม เหมาะกับ new menu, seasonal",

    base_prompt=(
        "Overhead flat lay food photography, {product} as centerpiece, "
        "surrounded by complementary ingredients and props, "
        "marble or wooden surface texture, {photo_style}, "
        "artful arrangement with negative space on one side, "
        "natural daylight, soft shadows, commercial editorial quality, "
        "no text, no watermark, 4K"
    ),
    negative_prompt="text, watermark, vertical angle, dark, messy, crowded frame, plastic utensils",

    composition="bottom_third",
    aspect_ratios=["1:1", "4:5"],
    image_safe_zone="One corner or bottom edge kept minimal for text overlay",

    lighting="Soft diffused natural daylight, minimal shadows, even exposure",
    color_palette="Earthy neutrals, marble white, wooden warm — product as color accent",
    mood="Editorial, artful, fresh, curated",
    visual_language="Perfect top-down symmetry or intentional asymmetry, props tell a story",

    action_types=["new_menu", "seasonal", "milestone_reward", "bestseller"],
    tiers=["cafe", "premium"],
    channels=["meta", "both"],

    overlay_bg="none",
    badge_position="none",
    font_weight="light",
))


# 4. lifestyle_scene — บรรยากาศร้าน คนใช้งานจริง (แบบ candid lifestyle)
_reg(VisualStyle(
    key="lifestyle_scene",
    name_th="ไลฟ์สไตล์บรรยากาศ",
    description_th="คนในร้าน บรรยากาศจริง ความสุข ความอบอุ่น เหมาะกับ Win-Back, Welcome",

    base_prompt=(
        "Lifestyle restaurant photography, happy customers enjoying {product}, "
        "cozy café atmosphere, candid natural moment, warm ambient lighting, "
        "bokeh background showing restaurant interior, {photo_style}, "
        "genuine emotion, social dining scene, commercial lifestyle quality, "
        "no text, no watermark"
    ),
    negative_prompt="text, watermark, posed stiff, studio white background, empty restaurant, sad expressions",

    composition="right_half",
    aspect_ratios=["1:1", "4:5", "9:16"],
    image_safe_zone="Left or right third with darker/blurred area for text",

    lighting="Warm ambient interior light, golden hour or candle-warm tones",
    color_palette="Warm amber, skin tones, soft restaurant colors — human and inviting",
    mood="Warm, social, genuine, nostalgic, belonging",
    visual_language="People + food together, candid not staged, bokeh creates intimacy",

    action_types=["win_back", "welcome", "brand_story", "milestone_reward"],
    tiers=["casual", "cafe"],
    channels=["meta", "line_oa", "both"],

    overlay_bg="dark_gradient",
    badge_position="none",
    font_weight="bold",
))


# 5. minimal_premium — clean minimal สำหรับ premium tier (Korean café inspired)
_reg(VisualStyle(
    key="minimal_premium",
    name_th="Minimal พรีเมียม",
    description_th="Clean, minimalist, earthy tones — เหมาะกับ specialty café, premium brand",

    base_prompt=(
        "Minimalist premium product photography of {product}, "
        "centered on pure white or warm cream background, "
        "single dramatic side light creating soft shadow, "
        "Korean café aesthetic, {photo_style}, "
        "generous negative space, typographic composition, "
        "editorial magazine quality, no text, no watermark"
    ),
    negative_prompt="text, watermark, busy background, bright colors, promotional badges, clutter, warm yellow cast",

    composition="center_overlay",
    aspect_ratios=["1:1", "4:5", "9:16"],
    image_safe_zone="Large negative space top and bottom for typography",

    lighting="Single soft side light, clean directional shadow, natural window quality",
    color_palette="White, cream, warm grey, sage green, terracotta — muted and sophisticated",
    mood="Sophisticated, calm, premium, restrained, editorial",
    visual_language="Breathing room, single hero, typographic balance, no clutter",

    action_types=["vip_reward", "zero_cost", "upsell_nudge", "brand_story", "new_menu"],
    tiers=["cafe", "premium"],
    channels=["meta", "both"],

    overlay_bg="none",
    badge_position="none",
    font_weight="light",
))


# 6. urgent_countdown — ด่วน countdown urgency (แบบ Hours Left, Last Chance)
_reg(VisualStyle(
    key="urgent_countdown",
    name_th="ด่วน! นับถอยหลัง",
    description_th="High urgency, countdown feel, bold contrast — เหมาะกับ Rescue, Last Chance",

    base_prompt=(
        "Dark dramatic food advertisement, {product} dramatically lit against dark background, "
        "moody studio lighting with single spotlight, "
        "high contrast chiaroscuro effect, deep shadows, "
        "cinematic commercial quality, {photo_style}, "
        "no text, no watermark, intense and urgent mood"
    ),
    negative_prompt="text, watermark, bright pastel, cheerful, flat lighting, low contrast, blurry",

    composition="center_overlay",
    aspect_ratios=["1:1", "4:5", "9:16"],
    image_safe_zone="Center product — text floats on dark areas with high contrast",

    lighting="Single dramatic spotlight, deep shadows, cinematic contrast",
    color_palette="Deep black/dark background, product illuminated, red or brand accent for urgency",
    mood="Urgent, dramatic, now-or-never, high stakes",
    visual_language="Dark background + isolated lit product + bold countdown text overlay",

    action_types=["rescue", "reactivation"],
    tiers=["casual", "cafe"],
    channels=["meta", "both"],

    overlay_bg="dark_gradient",
    badge_position="top_right",
    font_weight="black",
))


# 7. seasonal_festive — เทศกาล เฉลิมฉลอง (สงกรานต์, ตรุษจีน, ปีใหม่)
_reg(VisualStyle(
    key="seasonal_festive",
    name_th="เทศกาล/ซีซั่น",
    description_th="Festive, celebratory — ผูก brand กับเทศกาล เหมาะกับ seasonal campaigns",

    base_prompt=(
        "Festive food advertisement, {product} surrounded by {season} decorations, "
        "celebratory atmosphere, vibrant seasonal colors, "
        "confetti or traditional Thai/Chinese decorative elements, "
        "joyful and warm, commercial poster quality, "
        "{photo_style}, no text, no watermark"
    ),
    negative_prompt="text, watermark, plain background, sad, dark, minimalist, boring",

    composition="bottom_third",
    aspect_ratios=["1:1", "4:5", "9:16"],
    image_safe_zone="Bottom third for text, decorative elements frame the edges",

    lighting="Bright, warm festive light, slightly elevated contrast",
    color_palette="Seasonal: Songkran=blues/yellows, CNY=red/gold, NewYear=gold/champagne, Valentine=pink/red",
    mood="Celebratory, joyful, shared happiness, festive occasion",
    visual_language="Seasonal props + food + festive color palette + celebratory energy",

    action_types=["seasonal", "new_menu", "welcome", "brand_story"],
    tiers=["casual", "cafe", "all"],
    channels=["meta", "line_oa", "both"],

    overlay_bg="dark_gradient",
    badge_position="top_right",
    font_weight="bold",
))


# 8. menu_grid — หลาย product ในเฟรมเดียว + ราคา (แบบ menu launch)
_reg(VisualStyle(
    key="menu_grid",
    name_th="เมนูหลายรายการ",
    description_th="หลาย product + ราคาในเฟรมเดียว เหมาะกับ menu launch, campaign หลาย action",

    base_prompt=(
        "Food menu layout photography, multiple dishes of {product} arranged neatly, "
        "each item clearly visible, clean restaurant style, "
        "uniform lighting across all items, "
        "overhead or slight 3/4 angle, white or brand-colored background, "
        "commercial menu photography quality, no text, no watermark"
    ),
    negative_prompt="text, watermark, single item only, dark moody, overlapping food, messy",

    composition="full_bleed",
    aspect_ratios=["1:1", "4:5"],
    image_safe_zone="Items arranged with gaps — text/price overlays go between items",

    lighting="Even studio lighting across all items, no harsh shadows",
    color_palette="Clean neutral background + natural food colors — menu clarity over mood",
    mood="Appetizing, organized, abundance, good value",
    visual_language="Multiple items in organized grid, equal treatment of each product",

    action_types=["new_menu", "upsell_nudge", "bestseller"],
    tiers=["casual", "cafe"],
    channels=["meta", "line_oa", "both"],

    overlay_bg="none",
    badge_position="top_right",
    font_weight="bold",
))


# 9. korean_aesthetic — Korean café pastel aesthetic (กำลัง trend ในไทย)
_reg(VisualStyle(
    key="korean_aesthetic",
    name_th="Korean Aesthetic",
    description_th="Pastel, dreamy, clean — กำลัง trend ในกลุ่ม specialty café ไทย",

    base_prompt=(
        "Korean café aesthetic product photography of {product}, "
        "soft pastel color palette, dreamy soft-focus background, "
        "ceramic or glass vessel, clean composition, "
        "light airy feel, Scandinavian-Korean minimal style, "
        "Instagram-worthy, {photo_style}, "
        "natural window light, soft shadows, no text, no watermark"
    ),
    negative_prompt="text, watermark, dark, bold colors, street food feel, plastic, Thai casual vibe",

    composition="center_overlay",
    aspect_ratios=["1:1", "4:5", "9:16"],
    image_safe_zone="Soft top area and bottom for overlay — both have natural negative space",

    lighting="Soft diffused window light, airy, no hard shadows, slightly overexposed",
    color_palette="Soft pink, lavender, sage, cream, dusty blue — desaturated and gentle",
    mood="Dreamy, calm, aesthetic, Instagram-worthy, trendy",
    visual_language="Pastel world, ceramic vessels, impeccable styling, Korean café template",

    action_types=["vip_reward", "new_menu", "brand_story", "upsell_nudge"],
    tiers=["cafe", "premium"],
    channels=["meta", "both"],

    overlay_bg="none",
    badge_position="none",
    font_weight="light",
))


# 10. line_broadcast — LINE OA rich message format (horizontal banner)
_reg(VisualStyle(
    key="line_broadcast",
    name_th="LINE Broadcast Banner",
    description_th="Format สำหรับ LINE OA broadcast — horizontal banner 1040×585px",

    base_prompt=(
        "Food advertisement banner, {product} on the right two-thirds, "
        "left one-third with clean solid or gradient background for text, "
        "wide horizontal composition, warm inviting lighting, "
        "clear product hero shot, Thai restaurant quality, "
        "{photo_style}, no text, no watermark, banner format"
    ),
    negative_prompt="text, watermark, vertical composition, dark, complex background, overhead shot",

    composition="left_half",
    aspect_ratios=["1:1"],   # will be cropped to LINE format 1040×585 (16:9ish)
    image_safe_zone="Left 35% solid or gradient — right 65% for product hero",

    lighting="Bright, clean, inviting — optimized for small mobile screen readability",
    color_palette="Brand primary + white/light — high readability on mobile",
    mood="Clear, direct, action-oriented — optimized for LINE message",
    visual_language="Horizontal split: color band left + product right, very clear CTA area",

    action_types=["win_back", "welcome", "rescue", "vip_reward", "seasonal"],
    tiers=["casual", "cafe", "all"],
    channels=["line_oa"],

    overlay_bg="solid_band",
    badge_position="top_right",
    font_weight="bold",
))


# ---------------------------------------------------------------------------
# Action → Style mapping
# ---------------------------------------------------------------------------

ACTION_STYLE_MAP: dict[str, list[str]] = {
    "win_back":         ["warm_hero", "lifestyle_scene", "bold_promo"],
    "early_winback":    ["warm_hero", "bold_promo"],
    "rescue":           ["urgent_countdown", "bold_promo"],
    "reactivation":     ["urgent_countdown", "bold_promo"],
    "vip_reward":       ["minimal_premium", "korean_aesthetic", "lifestyle_scene"],
    "welcome":          ["warm_hero", "lifestyle_scene", "seasonal_festive"],
    "milestone_reward": ["lifestyle_scene", "flat_lay", "warm_hero"],
    "upsell_nudge":     ["minimal_premium", "korean_aesthetic", "menu_grid"],
    "zero_cost":        ["minimal_premium", "warm_hero"],
    "new_menu":         ["flat_lay", "warm_hero", "menu_grid", "korean_aesthetic"],
    "bestseller":       ["warm_hero", "menu_grid", "bold_promo"],
    "brand_story":      ["lifestyle_scene", "minimal_premium", "korean_aesthetic"],
    "seasonal":         ["seasonal_festive", "flat_lay", "warm_hero"],
    "default":          ["warm_hero", "bold_promo"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_style(key: str) -> VisualStyle:
    """Get a style by key. Raises KeyError if not found."""
    if key not in STYLES:
        raise KeyError(f"Style '{key}' not found. Available: {list(STYLES.keys())}")
    return STYLES[key]


def get_styles_for_action(action: str) -> list[VisualStyle]:
    """Return ordered list of recommended styles for a campaign action."""
    action_lower = action.lower().replace(" ", "_").replace("-", "_")
    # Fuzzy match
    for key, styles in ACTION_STYLE_MAP.items():
        if key in action_lower or action_lower in key:
            return [STYLES[s] for s in styles if s in STYLES]
    return [STYLES[s] for s in ACTION_STYLE_MAP["default"] if s in STYLES]


def build_image_prompt(
    style: VisualStyle,
    product: str,
    photo_style: str = "warm natural light",
    brand_color_desc: str = "warm earthy tones",
    season: str = "",
) -> tuple[str, str]:
    """Build final image prompt and negative prompt for a given style.

    Returns:
        (positive_prompt, negative_prompt)
    """
    positive = style.base_prompt.format(
        product=product,
        photo_style=photo_style,
        brand_color_desc=brand_color_desc,
        season=season or "festive",
    )
    return positive, style.negative_prompt


def list_styles() -> list[dict]:
    """Return summary of all styles for UI display."""
    return [
        {
            "key": s.key,
            "name_th": s.name_th,
            "description_th": s.description_th,
            "action_types": s.action_types,
            "tiers": s.tiers,
            "composition": s.composition,
        }
        for s in STYLES.values()
    ]


def auto_select_style(
    action: str,
    tier: str = "casual",
    channel: str = "meta",
) -> VisualStyle:
    """Automatically select the best style for action + tier + channel.

    Priority: action match → tier match → channel match → first available
    """
    candidates = get_styles_for_action(action)
    # Filter by tier
    tier_matches = [s for s in candidates if tier in s.tiers or "all" in s.tiers]
    if tier_matches:
        candidates = tier_matches
    # Filter by channel
    chan_matches = [s for s in candidates if channel in s.channels or "both" in s.channels]
    if chan_matches:
        candidates = chan_matches
    return candidates[0] if candidates else STYLES["warm_hero"]


# ---------------------------------------------------------------------------
# CLI preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Visual Style Library ===\n")
    for info in list_styles():
        print(f"/{info['key']:20s}  {info['name_th']:18s}  → {', '.join(info['action_types'][:3])}")

    print("\n--- Auto-select examples ---")
    for action, tier in [
        ("win_back", "casual"),
        ("vip_reward", "premium"),
        ("rescue", "casual"),
        ("new_menu", "cafe"),
        ("seasonal", "all"),
    ]:
        s = auto_select_style(action, tier)
        print(f"  {action:20s} [{tier:8s}] → /{s.key} ({s.name_th})")

    print("\n--- Prompt example (warm_hero + Latte) ---")
    s = get_style("warm_hero")
    pos, neg = build_image_prompt(s, product="Iced Latte", photo_style="cozy café natural light")
    print(f"Positive: {pos[:120]}...")
    print(f"Negative: {neg[:80]}...")
