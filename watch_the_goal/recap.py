"""Shared, local PNG artwork for the player and FFmpeg compositor."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TITLE_SECONDS = 2
FONT = Path(__file__).parent / "assets" / "Barlow-SemiBold.ttf"


def clean_text(value, limit):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"Use at most {limit} characters.")
    return " ".join(value.split())


def board_text(details):
    teams = "  /  ".join(t for t in (details.get("home"), details.get("away")) if t)
    return "    ".join(t for t in (teams, details.get("score")) if t)


@lru_cache(maxsize=32)
def artwork(kind: str, size: tuple[int, int], text: str) -> bytes:
    w, h = size
    title = kind == "title"
    image = Image.new("RGBA", size, "#101e1a" if title else (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    unit = min(w, h)
    margin, pad = round(unit * .045), round(unit * .018)
    font_size = round(unit * (.096 if title else .036))
    font = ImageFont.truetype(str(FONT), font_size)
    width = draw.textlength(text, font=font)
    max_width = w - 2 * (margin + pad)
    if width > max_width:
        font = ImageFont.truetype(str(FONT), max(1, int(font_size * max_width / width)))
    if title:
        y = h // 2
        draw.rectangle((margin, y - unit * .09, margin + unit * .088, y - unit * .082), fill="#42de93")
        draw.text((margin, y), text, font=font, anchor="lm", fill="#f4f6ef")
    elif text:
        right = margin + 2 * pad + draw.textlength(text, font=font)
        bottom = margin + round(unit * .066)
        draw.rounded_rectangle((margin, margin, right, bottom), radius=8, fill="#101e1a")
        draw.rectangle((margin, margin + 7, margin + 4, bottom - 7), fill="#42de93")
        draw.text((margin + pad, (margin + bottom) / 2), text, font=font, anchor="lm", fill="#f4f6ef")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
