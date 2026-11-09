"""Image composition engine — overlays Thai ad copy onto generated images.

Takes a PIL Image + ContentOutput metadata and produces channel-ready ad images.

Composition zones (from visual_style_library):
  bottom_third   → dark gradient bottom 35%, text in that zone
  top_third      → dark gradient top 35%
  right_half     → solid band left 48%, product right
  left_half      → solid band right 48%, product left
  center_overlay → semi-transparent center panel
  full_bleed     → text floats with shadow over full image
  split_bottom   → solid color band bottom 40%

Output sizes:
  meta_1x1   → 1080×1080
  meta_4x5   → 1080×1350
  line_oa    → 1040×1040

Usage:
    from image_composer import compose_ad, save_composed_ads
    composed = compose_ad(image, content, badge_text="ลด 15%")
    save_composed_ads(composed, output_dir="./output", prefix="winback")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from content_generator import ContentOutput


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_SIZES: dict[str, tuple[int, int]] = {
    "meta_1x1": (1080, 1080),
    "meta_4x5": (1080, 1350),
    "line_oa":  (1040, 1040),
}

# (headline_px, body_px, cta_px) per font_weight
_TEXT_SIZES: dict[str, tuple[int, int, int]] = {
    "light":   (38, 22, 28),
    "regular": (42, 24, 30),
    "bold":    (50, 26, 32),
    "black":   (58, 28, 36),
}

# Project-bundled fonts (priority)
_SRC_DIR = os.path.dirname(__file__)
_BUNDLED_FONTS = {
    "regular": os.path.join(_SRC_DIR, "fonts", "Sarabun-Regular.ttf"),
    "bold":    os.path.join(_SRC_DIR, "fonts", "Sarabun-Bold.ttf"),
    "black":   os.path.join(_SRC_DIR, "fonts", "Sarabun-Bold.ttf"),
    "light":   os.path.join(_SRC_DIR, "fonts", "Sarabun-Regular.ttf"),
}

# System Thai font fallbacks
_THAI_FONT_PATHS = [
    "/Library/Fonts/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/thai/Loma.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Thonburi.ttc",
]

_FALLBACK_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

def _load_font(size: int, weight: str = "regular"):
    """Load Thai-capable font at given size. Prefers bundled Sarabun."""
    from PIL import ImageFont

    # 1. Try bundled font
    bundled = _BUNDLED_FONTS.get(weight, _BUNDLED_FONTS["regular"])
    if os.path.exists(bundled):
        try:
            return ImageFont.truetype(bundled, size)
        except Exception:
            pass

    # 2. System Thai fonts
    for path in _THAI_FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    # 3. Generic fallback
    for path in _FALLBACK_FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Image resize + crop
# ---------------------------------------------------------------------------

def _resize_crop(image: "PILImage.Image", target_w: int, target_h: int) -> "PILImage.Image":
    """Scale image to cover target dimensions, then center-crop."""
    from PIL import Image

    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = image.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def _apply_overlay(
    image: "PILImage.Image",
    composition: str,
    overlay_bg: str,
) -> "PILImage.Image":
    """Draw overlay layer onto a copy of the image."""
    from PIL import Image, ImageDraw

    if overlay_bg == "none" or not overlay_bg:
        return image.copy()

    img = image.copy().convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if overlay_bg == "dark_gradient":
        # Gradient from transparent → dark at the text zone edge
        if composition in ("bottom_third", "split_bottom"):
            zone_h = int(h * 0.45)
            for y in range(zone_h):
                alpha = int(230 * (y / zone_h) ** 0.7)
                draw.line([(0, h - zone_h + y), (w, h - zone_h + y)],
                          fill=(0, 0, 0, alpha))
        elif composition == "top_third":
            zone_h = int(h * 0.40)
            for y in range(zone_h):
                alpha = int(200 * (1 - y / zone_h))
                draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))
        else:
            # center_overlay / full_bleed — moderate full cover
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 130))

    elif overlay_bg == "solid_band":
        if composition == "right_half":
            band_w = int(w * 0.48)
            draw.rectangle([(0, 0), (band_w, h)], fill=(20, 20, 20, 235))
        elif composition == "left_half":
            band_w = int(w * 0.48)
            draw.rectangle([(w - band_w, 0), (w, h)], fill=(20, 20, 20, 235))
        else:
            band_h = int(h * 0.42)
            draw.rectangle([(0, h - band_h), (w, h)], fill=(20, 20, 20, 235))

    elif overlay_bg == "blur_panel":
        # Approximate blur panel with semi-transparent rect
        if composition in ("bottom_third", "split_bottom"):
            zone_h = int(h * 0.38)
            draw.rectangle([(0, h - zone_h), (w, h)], fill=(0, 0, 0, 170))
        else:
            draw.rectangle([(0, 0), (w, h)], fill=(0, 0, 0, 140))

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


# ---------------------------------------------------------------------------
# Text zone
# ---------------------------------------------------------------------------

def _text_zone(composition: str, w: int, h: int) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) bounding box for the text zone."""
    pad = int(w * 0.065)

    if composition == "bottom_third":
        return (pad, int(h * 0.63), w - pad, h - int(h * 0.04))
    elif composition == "top_third":
        return (pad, int(h * 0.04), w - pad, int(h * 0.37))
    elif composition == "right_half":
        # Text on LEFT half
        return (pad, int(h * 0.12), int(w * 0.46), int(h * 0.88))
    elif composition == "left_half":
        # Text on RIGHT half
        return (int(w * 0.54), int(h * 0.12), w - pad, int(h * 0.88))
    elif composition in ("center_overlay", "full_bleed"):
        return (pad, int(h * 0.32), w - pad, int(h * 0.78))
    elif composition == "split_bottom":
        return (pad, int(h * 0.60), w - pad, h - int(h * 0.04))
    else:
        return (pad, int(h * 0.63), w - pad, h - int(h * 0.04))


def _wrap_text(draw, text: str, font, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test = f"{line} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) > max_width and line:
            lines.append(line)
            line = word
        else:
            line = test
    if line:
        lines.append(line)
    return lines


def _draw_text_block(
    draw,
    zone: tuple[int, int, int, int],
    headline: str,
    body: str,
    cta: str,
    font_weight: str,
) -> None:
    """Draw headline → body → CTA in the given zone."""
    x1, y1, x2, y2 = zone
    zone_w = x2 - x1
    sizes = _TEXT_SIZES.get(font_weight, _TEXT_SIZES["bold"])

    h_font = _load_font(sizes[0], font_weight)
    b_font = _load_font(sizes[1], "regular")
    c_font = _load_font(sizes[2], font_weight)

    y = y1
    gap = int(sizes[0] * 0.3)

    # Headline — white with black stroke (no alpha issues)
    if headline:
        draw.text(
            (x1, y), headline, font=h_font,
            fill=(255, 255, 255),
            stroke_width=2, stroke_fill=(0, 0, 0),
        )
        y += sizes[0] + gap

    # Body — word-wrap, light grey with stroke
    if body:
        for line in _wrap_text(draw, body, b_font, zone_w):
            draw.text(
                (x1, y), line, font=b_font,
                fill=(230, 230, 230),
                stroke_width=1, stroke_fill=(0, 0, 0),
            )
            y += sizes[1] + int(gap * 0.4)
        y += int(gap * 0.3)

    # CTA — white pill button
    if cta and y < y2 - sizes[2] - 10:
        cta_y = min(y, y2 - sizes[2] - 20)
        bbox = draw.textbbox((x1, cta_y), cta, font=c_font)
        px, py = 22, 10
        draw.rounded_rectangle(
            [bbox[0] - px, bbox[1] - py, bbox[2] + px, bbox[3] + py],
            radius=22, fill=(255, 255, 255),
        )
        draw.text((x1, cta_y), cta, font=c_font, fill=(25, 25, 25))


# ---------------------------------------------------------------------------
# Badge
# ---------------------------------------------------------------------------

def _draw_badge(
    image: "PILImage.Image",
    badge_position: str,
    badge_text: str,
) -> "PILImage.Image":
    """Draw a promo badge sticker on the image."""
    from PIL import ImageDraw

    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = _load_font(28)

    bbox = draw.textbbox((0, 0), badge_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    px, py = 18, 10
    badge_w = text_w + px * 2
    badge_h = text_h + py * 2

    margin = 24
    if badge_position == "top_right":
        bx, by = w - badge_w - margin, margin
    elif badge_position == "top_left":
        bx, by = margin, margin
    elif badge_position == "bottom_right":
        bx, by = w - badge_w - margin, h - badge_h - margin
    else:
        bx, by = w - badge_w - margin, margin

    draw.rounded_rectangle([bx, by, bx + badge_w, by + badge_h], radius=12,
                           fill=(210, 40, 40))
    draw.text((bx + px, by + py), badge_text, font=font, fill=(255, 255, 255))

    return img


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def compose_ad(
    image: "PILImage.Image",
    content: "ContentOutput",
    output_channels: list[str] | None = None,
    badge_text: str = "",
) -> dict[str, "PILImage.Image"]:
    """Compose finished ad images with text overlays for each output channel.

    Args:
        image:           PIL Image from image_generator
        content:         ContentOutput — provides headline, body, cta, composition_zone, etc.
        output_channels: subset of OUTPUT_SIZES keys — None auto-picks by content.channel
        badge_text:      promo badge e.g. "ลด 15%" — drawn if badge_position != "none"

    Returns:
        dict of channel_key → PIL Image (RGB)
    """
    from PIL import ImageDraw

    if output_channels is None:
        if content.channel == "line_oa":
            output_channels = ["line_oa"]
        elif content.channel == "meta":
            output_channels = ["meta_1x1", "meta_4x5"]
        else:
            output_channels = ["meta_1x1", "meta_4x5", "line_oa"]

    composition = content.composition_zone or "bottom_third"
    overlay_bg  = content.overlay_bg or "dark_gradient"
    font_weight = content.font_weight or "bold"
    badge_pos   = content.badge_position or "top_right"

    results: dict[str, "PILImage.Image"] = {}

    for channel_key in output_channels:
        if channel_key not in OUTPUT_SIZES:
            continue
        target_w, target_h = OUTPUT_SIZES[channel_key]

        # 1. Resize + crop
        base = _resize_crop(image, target_w, target_h)

        # 2. Overlay
        base = _apply_overlay(base, composition, overlay_bg)

        # 3. Text
        draw = ImageDraw.Draw(base)
        zone = _text_zone(composition, target_w, target_h)
        _draw_text_block(draw, zone, content.headline, content.body, content.cta, font_weight)

        # 4. Badge
        if badge_text and badge_pos != "none":
            base = _draw_badge(base, badge_pos, badge_text)

        results[channel_key] = base

    return results


def save_composed_ads(
    composed: dict[str, "PILImage.Image"],
    output_dir: str | Path,
    prefix: str = "ad",
) -> dict[str, str]:
    """Save composed images to disk as PNG.

    Returns:
        dict of channel_key → absolute file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for channel_key, img in composed.items():
        path = output_dir / f"{prefix}__{channel_key}.png"
        img.save(str(path))
        paths[channel_key] = str(path)

    return paths
