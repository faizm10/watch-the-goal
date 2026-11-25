"""Turn per-frame person boxes into evidence about candidate moments.

This module measures; it doesn't decide. For each candidate window it reports
how many players were near a marked goal, how tightly players were bunched,
and whether someone right in front of the lens was covering a goal box. The
scoring in :mod:`watch_the_goal.detect` decides what that means.

All counts are compared with the same recording's typical values, sampled
across the whole file, so a crowded indoor dome and a quiet park aren't held
to the same number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .detectors import PersonDetector
from .frames import sample_frames, window_times

WINDOW_BEFORE = 6.0  # seconds before the anchor to look at
WINDOW_AFTER = 2.0
WINDOW_FPS = 1.0
BASELINE_FRAMES = 40
GOAL_MARGIN = 0.5  # widen each goal box by this share of its size on every side
BUNCH_RADIUS = 2.0  # body heights
LENS_HEIGHT = 0.35  # a person taller than this share of the frame is next to the camera
LENS_COVER = 0.3  # share of a goal box such a person must cover


@dataclass
class FrameDetections:
    t: float
    width: int
    height: int
    boxes: np.ndarray  # (N, 5) x0, y0, x1, y1, score


@dataclass
class PlayerEvidence:
    near_goal_peak: int = 0
    near_goal_t: float | None = None
    near_goal_label: str = ""
    near_goal_typical: float = 0.0
    bunch_peak: int = 0
    bunch_t: float | None = None
    bunch_typical: float = 0.0
    lens_blocked: bool = False
    frames: int = 0
    detections: list[FrameDetections] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "near_goal_peak": self.near_goal_peak,
            "near_goal_t": self.near_goal_t,
            "near_goal_label": self.near_goal_label,
            "near_goal_typical": round(self.near_goal_typical, 2),
            "bunch_peak": self.bunch_peak,
            "bunch_t": self.bunch_t,
            "bunch_typical": round(self.bunch_typical, 2),
            "lens_blocked": self.lens_blocked,
            "frames": self.frames,
        }


def feet(boxes: np.ndarray) -> np.ndarray:
    """Bottom-centre of each box: where the player stands."""
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1) if len(boxes) else np.zeros((0, 2))


def near_goal(boxes: np.ndarray, area: dict, width: int, height: int, margin: float = GOAL_MARGIN) -> int:
    """Players standing in or just around a goal box (normalised x, y, w, h)."""
    if not len(boxes):
        return 0
    x0 = (area["x"] - margin * area["w"]) * width
    x1 = (area["x"] + (1 + margin) * area["w"]) * width
    y0 = (area["y"] - margin * area["h"]) * height
    y1 = (area["y"] + (1 + margin) * area["h"]) * height
    f = feet(boxes)
    return int(((f[:, 0] >= x0) & (f[:, 0] <= x1) & (f[:, 1] >= y0) & (f[:, 1] <= y1)).sum())


def bunch(boxes: np.ndarray, radius: float = BUNCH_RADIUS) -> int:
    """Largest number of players within ``radius`` body heights of one player."""
    if len(boxes) < 2:
        return len(boxes)
    f = feet(boxes)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    dist = np.linalg.norm(f[:, None, :] - f[None, :, :], axis=2)
    reach = radius * np.maximum(h[:, None], h[None, :])
    return int((dist <= reach).sum(axis=1).max())


def lens_blocked(boxes: np.ndarray, area: dict, width: int, height: int) -> bool:
    """Someone right in front of the camera covers this goal box."""
    if not len(boxes):
        return False
    gx0, gy0 = area["x"] * width, area["y"] * height
    gx1, gy1 = gx0 + area["w"] * width, gy0 + area["h"] * height
    goal_area = max((gx1 - gx0) * (gy1 - gy0), 1.0)
    for x0, y0, x1, y1, _ in boxes:
        if (y1 - y0) < LENS_HEIGHT * height:
            continue
        ix = max(0.0, min(x1, gx1) - max(x0, gx0))
        iy = max(0.0, min(y1, gy1) - max(y0, gy0))
        if ix * iy / goal_area >= LENS_COVER:
            return True
    return False


def detect_frames(path: str, times: list[float], detector: PersonDetector) -> list[FrameDetections]:
    out = []
    for t, frame in sample_frames(path, times):
        h, w = frame.shape[:2]
        out.append(FrameDetections(t, w, h, detector.detect(frame)))
    return out


def typical_levels(frames: list[FrameDetections], goal_areas: list[dict]) -> tuple[float, float]:
    if not frames:
        return 0.0, 0.0
    near = [max((near_goal(f.boxes, a, f.width, f.height) for a in goal_areas), default=0) for f in frames]
    return float(np.median(near)), float(np.median([bunch(f.boxes) for f in frames]))


def measure(
    path: str,
    duration: float,
    anchors: list[float],
    detector: PersonDetector,
    goal_areas: list[dict] | None = None,
    area_labels: list[str] | None = None,
    progress=None,
) -> tuple[list[PlayerEvidence], dict]:
    """Player evidence for each anchor, plus timing and typical levels."""
    goal_areas = goal_areas or []
    area_labels = area_labels or [f"goal area {i + 1}" for i in range(len(goal_areas))]
    t0 = time.time()
    step = duration / (BASELINE_FRAMES + 1)
    base_times = [round(step * (i + 1), 2) for i in range(BASELINE_FRAMES)] if duration > 60 else []
    baseline = detect_frames(path, base_times, detector)
    near_typ, bunch_typ = typical_levels(baseline, goal_areas)
    if progress:
        progress(0.1)

    results = []
    for i, anchor in enumerate(anchors):
        times = window_times(max(0.0, anchor - WINDOW_BEFORE), min(duration, anchor + WINDOW_AFTER), WINDOW_FPS)
        frames = detect_frames(path, times, detector)
        ev = PlayerEvidence(near_goal_typical=near_typ, bunch_typical=bunch_typ, frames=len(frames), detections=frames)
        for f in frames:
            for a, label in zip(goal_areas, area_labels, strict=True):
                count = near_goal(f.boxes, a, f.width, f.height)
                if count > ev.near_goal_peak:
                    ev.near_goal_peak, ev.near_goal_t, ev.near_goal_label = count, f.t, label
                # Only the moment itself matters for a blocked lens.
                if abs(f.t - anchor) <= 2.0 and lens_blocked(f.boxes, a, f.width, f.height):
                    ev.lens_blocked = True
            b = bunch(f.boxes)
            if b > ev.bunch_peak:
                ev.bunch_peak, ev.bunch_t = b, f.t
        results.append(ev)
        if progress:
            progress(0.1 + 0.9 * (i + 1) / max(1, len(anchors)))
    stats = {
        "frames": len(baseline) + sum(r.frames for r in results),
        "seconds": round(time.time() - t0, 2),
        "near_goal_typical": near_typ,
        "bunch_typical": bunch_typ,
    }
    return results, stats
