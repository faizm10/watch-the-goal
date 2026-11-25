"""Shared, local PNG artwork for the player and FFmpeg compositor."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TITLE_SECONDS = 2
END_SECONDS = 3
FADE_SECONDS = 0.25  # dip to black between recap segments
# A caption slides in from the left and fades, shortly after its clip starts.
CAPTION_IN, CAPTION_EASE, CAPTION_HOLD = 0.4, 0.35, 4.0
KINDS = ("title", "board", "caption", "end")
FONT = Path(__file__).parent / "assets" / "Barlow-SemiBold.ttf"


def clean_text(value, limit):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"Use at most {limit} characters.")
    return " ".join(value.split())


def board_text(details):
    teams = "  /  ".join(t for t in (details.get("home"), details.get("away")) if t)
    return "    ".join(t for t in (teams, details.get("score")) if t)


def caption_window(duration: float) -> tuple[float, float]:
    """When a clip's caption is fully shown: (start, end) in seconds into the clip."""
    a = min(CAPTION_IN, duration * 0.2)
    b = min(duration - CAPTION_EASE - 0.2, a + CAPTION_EASE + CAPTION_HOLD)
    return a, max(b, a + CAPTION_EASE)


def _font(draw, text, size, max_width):
    font = ImageFont.truetype(str(FONT), size)
    width = draw.textlength(text, font=font)
    if width > max_width:
        font = ImageFont.truetype(str(FONT), max(1, int(size * max_width / width)))
    return font


@lru_cache(maxsize=32)
def artwork(kind: str, size: tuple[int, int], text: str) -> bytes:
    w, h = size
    card = kind in ("title", "end")
    image = Image.new("RGBA", size, "#101e1a" if card else (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    unit = min(w, h)
    margin, pad = round(unit * .045), round(unit * .018)
    max_width = w - 2 * (margin + pad)
    if card:
        font = _font(draw, text, round(unit * .096), max_width)
        y = h // 2
        if kind == "end":
            small = _font(draw, "Thanks for watching", round(unit * .036), max_width)
            draw.text((margin, y + unit * .1), "Thanks for watching", font=small, anchor="lm", fill="#9aa7a0")
        draw.rectangle((margin, y - unit * .09, margin + unit * .088, y - unit * .082), fill="#42de93")
        draw.text((margin, y), text, font=font, anchor="lm", fill="#f4f6ef")
    elif kind == "caption" and text:
        # A lower third, clear of the phone apps' buttons along the bottom of portrait video.
        font = _font(draw, text, round(unit * .052), max_width - pad)
        bottom = h - round(h * (.17 if h > w else .08))
        top = bottom - round(unit * .092)
        right = margin + 3 * pad + draw.textlength(text, font=font)
        draw.rounded_rectangle((margin, top, right, bottom), radius=10, fill="#101e1a")
        draw.rectangle((margin, top + 9, margin + 6, bottom - 9), fill="#42de93")
        draw.text((margin + 2 * pad, (top + bottom) / 2), text, font=font, anchor="lm", fill="#f4f6ef")
    elif text:
        font = _font(draw, text, round(unit * .036), max_width)
        right = margin + 2 * pad + draw.textlength(text, font=font)
        bottom = margin + round(unit * .066)
        draw.rounded_rectangle((margin, margin, right, bottom), radius=8, fill="#101e1a")
        draw.rectangle((margin, margin + 7, margin + 4, bottom - 7), fill="#42de93")
        draw.text((margin + pad, (margin + bottom) / 2), text, font=font, anchor="lm", fill="#f4f6ef")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
