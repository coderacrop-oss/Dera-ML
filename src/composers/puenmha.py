"""เพื่อนหมา PET LOVER — Design System Compositor v2

Matches the reference design system exactly:
  ┌─────────────────────────────────────────────┐
  │ เพื่อนหมา PET LOVER        [CAFÉ · TYPE]   │  ← top bar
  │ ○ decorative circle (top-right bleed)        │
  │                                              │
  │ [campaign chip]                              │
  │                                              │
  │ Headline ใหญ่มาก                            │  ← center content
  │ body text สั้นๆ                             │
  │                                              │
  │ [CTA pill]              [ลด X% circle]      │
  │                         [mascot PNG]         │
  │                                              │
  │ [date chip] [action chip]                    │  ← bottom chips
  ├──────────────────────────────────────────────┤
  │ ทักไลน์ · @LINE ID    เงื่อนไข...           │  ← footer bar
  └──────────────────────────────────────────────┘

Usage:
    from composers.puenmha import compose_promo, compose_how_to, compose_last_call
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

_FONTS_DIR = Path(__file__).parent.parent / "fonts"

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
C = {
    "red":        (232, 68,  74),
    "red_light":  (238, 97,  102),
    "blue":       (143, 203, 240),
    "blue_light": (169, 216, 245),
    "cream":      (254, 246, 218),
    "cream_dark": (251, 235, 184),
    "black":      (17,  17,  17),
    "white":      (255, 255, 255),
    "grey":       (85,  85,  79),
    "pink":       (255, 227, 227),
    "paper":      (236, 232, 222),
}

THEMES = {
    "win_back":    {"bg": "red",   "text": "white", "circle": "red_light",  "chip_bg": "white",  "chip_txt": "red"},
    "rescue":      {"bg": "black", "text": "white", "circle": "red",        "chip_bg": "red",    "chip_txt": "white"},
    "new_service": {"bg": "blue",  "text": "black", "circle": "blue_light", "chip_bg": "red",    "chip_txt": "white"},
    "new_menu":    {"bg": "cream", "text": "black", "circle": "cream_dark", "chip_bg": "red",    "chip_txt": "white"},
    "vip_reward":  {"bg": "black", "text": "white", "circle": "red",        "chip_bg": "blue",   "chip_txt": "black"},
    "bestseller":  {"bg": "blue",  "text": "black", "circle": "blue_light", "chip_bg": "black",  "chip_txt": "white"},
    "default":     {"bg": "red",   "text": "white", "circle": "red_light",  "chip_bg": "white",  "chip_txt": "red"},
}

SIZE   = 1040
FOOTER = 90   # footer bar height
MARGIN = 60


# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------

def _font(size: int, weight: int = 400, family: str = "noto"):
    from PIL import ImageFont
    if family == "sriracha":
        p = _FONTS_DIR / "Sriracha-Regular.ttf"
        if p.exists():
            return ImageFont.truetype(str(p), size)
    vf = _FONTS_DIR / "NotoSansThai-VF.ttf"
    if vf.exists():
        try:
            f = ImageFont.truetype(str(vf), size)
            if weight >= 700:
                try: f.set_variation_by_axes({"wght": weight})
                except: pass
            return f
        except: pass
    fb = _FONTS_DIR / ("Sarabun-Bold.ttf" if weight >= 700 else "Sarabun-Regular.ttf")
    if fb.exists():
        return ImageFont.truetype(str(fb), size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _c(name): return C[name]

def _deco_circle(draw, canvas_h, color: str):
    """Large decorative circle bleeding top-right."""
    r = int(SIZE * 0.60)
    cx = SIZE - int(r * 0.30)
    cy = -int(r * 0.22)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=_c(color))

def _chip(draw, x, y, text, bg, txt, font):
    """Pill chip. Returns (right_x, bottom_y)."""
    bbox = draw.textbbox((0,0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    px, py = 22, 11
    w, h = tw+px*2, th+py*2
    draw.rounded_rectangle([x, y, x+w, y+h], radius=999, fill=_c(bg))
    draw.text((x+px, y+py), text, font=font, fill=_c(txt))
    return x+w, y+h

def _pill_button(draw, x, y, text, bg, txt, font):
    """CTA button (wider padding than chip)."""
    bbox = draw.textbbox((0,0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    px, py = 36, 16
    w, h = tw+px*2, th+py*2
    draw.rounded_rectangle([x, y, x+w, y+h], radius=999, fill=_c(bg))
    draw.text((x+px, y+py), text, font=font, fill=_c(txt))
    return x+w, y+h

def _wrap(draw, text, font, x, y, max_w, fill, lh=1.2):
    """Word-wrapped text. Returns y after last line."""
    words = text.split()
    lines, line = [], ""
    for w in words:
        t = f"{line} {w}".strip()
        if (draw.textbbox((0,0), t, font=font)[2]) > max_w and line:
            lines.append(line); line = w
        else: line = t
    if line: lines.append(line)
    for ln in lines:
        draw.text((x, y), ln, font=font, fill=_c(fill))
        h = draw.textbbox((0,0), ln, font=font)[3]
        y += int(h * lh)
    return y

def _footer_bar(img, draw, line_id: str = "@LINE ID"):
    """Dark footer strip at bottom."""
    from PIL import Image
    fy = SIZE - FOOTER
    draw.rectangle([0, fy, SIZE, SIZE], fill=_c("black"))
    f = _font(26, 400, "noto")
    fs = _font(22, 400, "sriracha")
    # Left: ทักไลน์จองโต๊ะ · @LINE ID
    left = f"ทักไลน์จองโต๊ะ · {line_id}"
    draw.text((MARGIN, fy + (FOOTER-26)//2), left, font=fs, fill=_c("white"))
    # Right: disclaimer
    disc = "เงื่อนไขเป็นไปตามที่ร้านกำหนด"
    db = draw.textbbox((0,0), disc, font=f)
    draw.text((SIZE - MARGIN - (db[2]-db[0]), fy + (FOOTER-22)//2), disc, font=f, fill=(160,160,155))

def _big_promo_circle(draw, discount_text: str, cx: int, cy: int, r: int = 130):
    """Big cream circle with red discount text — matches reference design."""
    # Outer cream circle
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=_c("cream"))
    # Thin red ring
    draw.ellipse([cx-r+6, cy-r+6, cx+r-6, cy+r-6], outline=_c("red"), width=3)
    f_pct = _font(72, 800, "noto")
    f_lbl = _font(28, 700, "noto")
    # "ลด" label in red
    lb = draw.textbbox((0,0), "ลด", font=f_lbl)
    draw.text((cx - (lb[2]-lb[0])//2, cy - r//2 - 10), "ลด", font=f_lbl, fill=_c("red"))
    # Discount number in red
    db = draw.textbbox((0,0), discount_text, font=f_pct)
    draw.text((cx - (db[2]-db[0])//2, cy - (db[3]-db[1])//2 + 10), discount_text, font=f_pct, fill=_c("red"))

def _mascot(img, mascot_path: str | None, x: int, y: int, height: int = 280):
    """Paste mascot PNG if available."""
    if not mascot_path or not Path(mascot_path).exists():
        return
    from PIL import Image
    mascot = Image.open(mascot_path).convert("RGBA")
    # Scale to target height
    ratio = height / mascot.height
    w = int(mascot.width * ratio)
    mascot = mascot.resize((w, height), Image.LANCZOS)
    img.paste(mascot, (x, y), mascot)


# ---------------------------------------------------------------------------
# Layout composers
# ---------------------------------------------------------------------------

def _top_bar(draw, brand_name, category, chip_label, t, f_brand, f_cat, bg_image=False):
    """Render top bar: brand name left, category badge right. Returns y after bar."""
    brand_color = "blue" if t["bg"] == "black" else ("white" if t["bg"] == "red" else "black")
    if bg_image: brand_color = "black"
    draw.text((MARGIN, 52), brand_name, font=f_brand, fill=_c(brand_color))
    cat_text = f"{category} · {chip_label.upper()}"
    cat_bg = "black" if t["bg"] != "black" else "red"
    cb = draw.textbbox((0,0), cat_text, font=f_cat)
    cw, ch = (cb[2]-cb[0]) + 32, (cb[3]-cb[1]) + 14
    cat_x = SIZE - MARGIN - cw
    draw.rounded_rectangle([cat_x, 52, cat_x+cw, 52+ch], radius=999, fill=_c(cat_bg))
    draw.text((cat_x+16, 59), cat_text, font=f_cat, fill=_c("white"))
    return 52 + ch + 24


def compose_promo(
    headline: str,
    body: str,
    cta: str,
    campaign_chip: str = "โปรพิเศษ",
    category: str = "CAFÉ",
    campaign_type: str = "win_back",
    discount: str = "30%",
    date_range: str = "",
    line_id: str = "@LINE ID",
    mascot_path: str | None = None,
    brand_name: str = "เพื่อนหมา PET LOVER",
    bg_image=None,
) -> "PIL.Image.Image":
    """Two-column layout: text left 55%, mascot right 45% full-height."""
    from PIL import Image, ImageDraw

    t = THEMES.get(campaign_type, THEMES["default"])
    img = Image.new("RGB", (SIZE, SIZE), _c(t["bg"]))
    draw = ImageDraw.Draw(img)

    # Decorative circle top-right bleed
    _deco_circle(draw, SIZE, t["circle"])

    f_brand    = _font(32, 400, "sriracha")
    f_cat      = _font(20, 700, "noto")
    f_chip     = _font(24, 700, "noto")
    f_headline = _font(68, 800, "noto")
    f_body     = _font(26, 400, "noto")
    f_cta      = _font(28, 700, "noto")
    f_date     = _font(22, 400, "noto")

    htxt = t["text"]
    btxt = t["text"]

    # --- Layout constants ---
    # Right column starts at 55% → mascot fills right side top-to-bottom
    LEFT_COL_W = int(SIZE * 0.56) - MARGIN   # max width for text
    CONTENT_TOP = 52                           # first element y
    CONTENT_BOTTOM = SIZE - FOOTER - 56       # last row y (chips)

    # --- Top bar ---
    y = _top_bar(draw, brand_name, category, campaign_chip, t, f_brand, f_cat)
    y += 8

    # --- Campaign chip ---
    chip_bg = t["chip_bg"]
    chip_tx = t["chip_txt"]
    _, cy2 = _chip(draw, MARGIN, y, campaign_chip, chip_bg, chip_tx, f_chip)
    y = cy2 + 32

    # --- Headline (left col only) ---
    y = _wrap(draw, headline, f_headline, MARGIN, y, LEFT_COL_W, htxt, 1.15)
    y += 18

    # --- Body ---
    y = _wrap(draw, body, f_body, MARGIN, y, LEFT_COL_W, btxt, 1.4)
    y += 40

    # --- CTA button ---
    cta_bg  = "white" if t["bg"] in ("red","black") else "black"
    cta_txt = "red" if t["bg"] == "red" else ("white" if cta_bg == "black" else "black")
    _pill_button(draw, MARGIN, y, cta, cta_bg, cta_txt, f_cta)

    # --- Date chip pinned to bottom-left ---
    if date_range:
        date_chip_bg = "white" if t["bg"] not in ("cream","blue") else "paper"
        _chip(draw, MARGIN, CONTENT_BOTTOM, date_range, date_chip_bg, "black", f_date)

    # --- RIGHT COLUMN: mascot fills full height, circle anchored lower-right ---
    right_col_x = int(SIZE * 0.52)                     # start x of right column
    mascot_top  = CONTENT_TOP + 10                      # align with top bar
    mascot_bot  = SIZE - FOOTER                         # flush above footer
    mascot_h    = mascot_bot - mascot_top               # fills full column height

    # Promo circle: lower-right corner of canvas, partially bleeding off edge
    if discount:
        c_r  = 130
        c_cx = SIZE - c_r + 20          # bleed right edge slightly
        c_cy = SIZE - FOOTER - c_r + 10 # sit just above footer
        _big_promo_circle(draw, discount, c_cx, c_cy, r=c_r)

    # Mascot: right column, centered horizontally, full height
    if mascot_path:
        from PIL import Image as _Img
        from pathlib import Path as _Path
        if _Path(mascot_path).exists():
            m = _Img.open(mascot_path).convert("RGBA")
            # Scale to fill right column height
            ratio = mascot_h / m.height
            mw = int(m.width * ratio)
            m = m.resize((mw, mascot_h), _Img.LANCZOS)
            # Center in right column
            mx = right_col_x + max(0, (SIZE - right_col_x - mw) // 2)
            img.paste(m, (mx, mascot_top), m)

    # --- Footer ---
    _footer_bar(img, draw, line_id)
    return img


def compose_last_call(
    headline: str,
    body: str,
    discount: str = "30%",
    date_end: str = "",
    line_id: str = "@LINE ID",
    mascot_path: str | None = None,
    brand_name: str = "เพื่อนหมา PET LOVER",
) -> "PIL.Image.Image":
    """Last call urgency post — same two-column principle: text left, mascot right full-height."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (SIZE, SIZE), _c("black"))
    draw = ImageDraw.Draw(img)

    # Subtle dark deco circle top-right
    r = int(SIZE * 0.55)
    draw.ellipse([SIZE-int(r*0.3)-r, -int(r*0.2)-r, SIZE-int(r*0.3)+r, -int(r*0.2)+r], fill=(35,35,35))

    f_brand = _font(32, 400, "sriracha")
    f_cat   = _font(20, 700, "noto")
    f_chip  = _font(24, 700, "noto")
    f_h     = _font(76, 800, "noto")
    f_body  = _font(26, 400, "noto")
    f_date  = _font(22, 400, "noto")

    LEFT_COL_W   = int(SIZE * 0.56) - MARGIN
    CONTENT_TOP  = 52
    CONTENT_BOT  = SIZE - FOOTER - 56

    # Top bar
    t_dummy = {"bg": "black", "text": "white"}
    y = 52
    draw.text((MARGIN, y), brand_name, font=f_brand, fill=_c("blue"))
    cb = draw.textbbox((0,0), "CAFÉ · LAST CALL", font=f_cat)
    cw = (cb[2]-cb[0]) + 32
    draw.rounded_rectangle([SIZE-MARGIN-cw, y, SIZE-MARGIN, y+(cb[3]-cb[1])+14], radius=999, fill=_c("red"))
    draw.text((SIZE-MARGIN-cw+16, y+7), "CAFÉ · LAST CALL", font=f_cat, fill=_c("white"))
    y += (cb[3]-cb[1]) + 14 + 24

    _chip(draw, MARGIN, y, "โค้งสุดท้าย", "white", "black", f_chip)
    y += 56

    y = _wrap(draw, headline, f_h, MARGIN, y, LEFT_COL_W, "white", 1.1)
    y += 18
    _wrap(draw, body, f_body, MARGIN, y, LEFT_COL_W, "grey", 1.4)

    # Bottom chips pinned
    cx2 = MARGIN
    if date_end:
        cx2, _ = _chip(draw, cx2, CONTENT_BOT, f"หมดเขต {date_end}", "paper", "black", f_date)
        cx2 += 12
    _chip(draw, cx2, CONTENT_BOT, "แสดงโพสต์นี้ที่ร้าน", "white", "black", f_date)

    # Right column: promo circle lower-right corner, mascot fills right column height
    right_col_x = int(SIZE * 0.52)
    mascot_top  = CONTENT_TOP + 10
    mascot_bot  = SIZE - FOOTER
    mascot_h    = mascot_bot - mascot_top

    c_r = 130
    _big_promo_circle(draw, discount, SIZE - c_r + 20, SIZE - FOOTER - c_r + 10, r=c_r)

    if mascot_path:
        from PIL import Image as _Img
        from pathlib import Path as _Path
        if _Path(mascot_path).exists():
            m = _Img.open(mascot_path).convert("RGBA")
            ratio = mascot_h / m.height
            mw = int(m.width * ratio)
            m = m.resize((mw, mascot_h), _Img.LANCZOS)
            mx = right_col_x + max(0, (SIZE - right_col_x - mw) // 2)
            img.paste(m, (mx, mascot_top), m)

    _footer_bar(img, draw, line_id)
    return img


def compose_vibe(
    headline: str,
    body: str,
    cta: str = "",
    date_range: str = "",
    location: str = "",
    campaign_type: str = "new_service",
    line_id: str = "@LINE ID",
    mascot_path: str | None = None,
    brand_name: str = "เพื่อนหมา PET LOVER",
    bg_image=None,
) -> "PIL.Image.Image":
    """Atmosphere/vibe post — photo background, text overlay. Matches 03-cafe-vibe."""
    from PIL import Image, ImageDraw

    if bg_image:
        img = bg_image.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
        # Light white overlay bottom half for readability
        ov = Image.new("RGBA", (SIZE, SIZE), (0,0,0,0))
        ovd = ImageDraw.Draw(ov)
        for yy in range(SIZE//2, SIZE - FOOTER):
            a = int(160 * (yy - SIZE//2) / (SIZE//2 - FOOTER))
            ovd.line([(0, yy), (SIZE, yy)], fill=(0,0,0,a))
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    else:
        img = Image.new("RGB", (SIZE, SIZE), _c("white"))

    draw = ImageDraw.Draw(img)

    f_brand = _font(30, 400, "sriracha")
    f_cat   = _font(20, 700, "noto")
    f_h     = _font(64, 800, "noto")
    f_body  = _font(26, 400, "noto")
    f_chip  = _font(22, 400, "noto")

    htxt = "black" if not bg_image else "white"
    btxt = htxt

    y = 52
    top_txt = "black" if not bg_image else "white"
    draw.text((MARGIN, y), brand_name, font=f_brand, fill=_c(top_txt))
    cb = draw.textbbox((0,0), "CAFÉ · VIBE", font=f_cat)
    cw = (cb[2]-cb[0]) + 32
    cat_bg = "black" if not bg_image else "white"
    cat_txt = "white" if cat_bg == "black" else "black"
    draw.rounded_rectangle([SIZE-MARGIN-cw, y, SIZE-MARGIN, y+(cb[3]-cb[1])+14], radius=999, fill=_c(cat_bg))
    draw.text((SIZE-MARGIN-cw+16, y+7), "CAFÉ · VIBE", font=f_cat, fill=_c(cat_txt))
    y += 72

    # On photo bg, headline goes near bottom
    if bg_image:
        hy = SIZE - FOOTER - 320
    else:
        hy = y + 100

    _wrap(draw, headline, f_h, MARGIN, hy, int(SIZE*0.72)-MARGIN, htxt, 1.15)
    _wrap(draw, body, f_body, MARGIN, hy + 160, int(SIZE*0.72)-MARGIN, btxt, 1.4)

    chip_y = SIZE - FOOTER - 60
    cx2 = MARGIN
    if date_range:
        cx2, _ = _chip(draw, cx2, chip_y, date_range, "white", "black", f_chip)
        cx2 += 12
    if location:
        _chip(draw, cx2, chip_y, location, "black", "white", f_chip)

    _mascot(img, mascot_path, SIZE - 320, SIZE - FOOTER - 300, height=260)
    _footer_bar(img, draw, line_id)
    return img


def compose_how_to(
    headline: str,
    steps: list[dict],
    campaign_chip: str = "รับส่วนลด 30%",
    date_range: str = "",
    line_id: str = "@LINE ID",
    mascot_path: str | None = None,
    brand_name: str = "เพื่อนหมา PET LOVER",
) -> "PIL.Image.Image":
    """How-to 3-step post — cream bg, matches 02-how-to."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (SIZE, SIZE), _c("cream"))
    draw = ImageDraw.Draw(img)

    # Deco circle cream-dark
    r = int(SIZE * 0.52)
    cx, cy = SIZE - int(r*0.25), -int(r*0.2)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=_c("cream_dark"))

    f_brand = _font(30, 400, "sriracha")
    f_cat   = _font(20, 700, "noto")
    f_chip  = _font(24, 700, "noto")
    f_h     = _font(68, 800, "noto")
    f_step  = _font(28, 700, "noto")
    f_sub   = _font(22, 400, "noto")
    f_date  = _font(22, 400, "noto")

    y = 52
    draw.text((MARGIN, y), brand_name, font=f_brand, fill=_c("black"))
    cb = draw.textbbox((0,0), "CAFÉ · HOW TO", font=f_cat)
    cw = (cb[2]-cb[0]) + 32
    draw.rounded_rectangle([SIZE-MARGIN-cw, y, SIZE-MARGIN, y+(cb[3]-cb[1])+14], radius=999, fill=_c("black"))
    draw.text((SIZE-MARGIN-cw+16, y+7), "CAFÉ · HOW TO", font=f_cat, fill=_c("white"))
    y += 72

    _chip(draw, MARGIN, y, campaign_chip, "red", "white", f_chip)
    y += 56

    y = _wrap(draw, headline, f_h, MARGIN, y, SIZE-MARGIN*2, "black", 1.1)
    y += 36

    # Step cards
    card_h = 90
    card_gap = 16
    card_w = SIZE - MARGIN*2

    for i, step in enumerate(steps[:3]):
        # Card background
        card_y = y + i*(card_h + card_gap)
        is_last = (i == len(steps)-1)
        card_bg = "black" if is_last else "white"
        txt_col = "white" if is_last else "black"
        draw.rounded_rectangle([MARGIN, card_y, MARGIN+card_w, card_y+card_h], radius=20, fill=_c(card_bg))

        # Step number circle
        nr = 24
        nx = MARGIN + 44
        ny = card_y + card_h//2
        draw.ellipse([nx-nr, ny-nr, nx+nr, ny+nr], fill=_c("red" if not is_last else "red"))
        nb = draw.textbbox((0,0), str(i+1), font=f_step)
        draw.text((nx-(nb[2]-nb[0])//2, ny-(nb[3]-nb[1])//2), str(i+1), font=f_step, fill=_c("white"))

        # Step text
        tx = MARGIN + 92
        draw.text((tx, card_y+18), step.get("title",""), font=f_step, fill=_c(txt_col))
        if step.get("sub"):
            draw.text((tx, card_y+52), step.get("sub",""), font=f_sub, fill=_c("grey" if not is_last else "blue"))

    y += 3*(card_h+card_gap) + 24

    # Date chip
    chip_y = SIZE - FOOTER - 60
    if date_range:
        _chip(draw, MARGIN, chip_y, date_range, "paper", "black", f_date)

    # Mascot: right column top, height covers top ~45% (doesn't overlap step cards)
    right_col_x = int(SIZE * 0.60)
    if mascot_path:
        from PIL import Image as _Img
        from pathlib import Path as _Path
        if _Path(mascot_path).exists():
            m = _Img.open(mascot_path).convert("RGBA")
            target_h = 340
            ratio = target_h / m.height
            mw = int(m.width * ratio)
            m = m.resize((mw, target_h), _Img.LANCZOS)
            mx = right_col_x + max(0, (SIZE - right_col_x - mw) // 2)
            img.paste(m, (mx, 50), m)
    _footer_bar(img, draw, line_id)
    return img


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save(img, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return str(path)
