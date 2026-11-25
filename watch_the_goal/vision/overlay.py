"""Debug overlays: what the person detector saw at a moment."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .players import GOAL_MARGIN

YELLOW, CYAN, WHITE, BLACK = (0, 220, 255), (255, 200, 0), (255, 255, 255), (0, 0, 0)


def label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.8) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 2, cv2.LINE_AA)


def draw(frame: np.ndarray, boxes: np.ndarray, goal_areas: list[dict] | None = None,
         caption: str = "", show_scores: bool = False) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    thick = max(1, round(max(h, w) / 800))
    for a in goal_areas or []:
        x0, y0 = int(a["x"] * w), int(a["y"] * h)
        x1, y1 = int((a["x"] + a["w"]) * w), int((a["y"] + a["h"]) * h)
        cv2.rectangle(out, (x0, y0), (x1, y1), CYAN, thick + 1)
        mx, my = GOAL_MARGIN * a["w"] * w, GOAL_MARGIN * a["h"] * h
        cv2.rectangle(out, (int(x0 - mx), int(y0 - my)), (int(x1 + mx), int(y1 + my)), CYAN, thick)
    for x0, y0, x1, y1, s in boxes:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), YELLOW, thick)
        if show_scores:
            label(out, f"{s:.2f}", (int(x0), max(12, int(y0) - 3)), 0.4)
    if caption:
        label(out, caption, (12, 34))
    return out


def save(path: Path, image: np.ndarray, max_side: int = 1280) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = max_side / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path
