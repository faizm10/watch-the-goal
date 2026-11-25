"""Frame sampling with OpenCV.

OpenCV's FFmpeg backend applies the rotation stored by phones, so portrait
clips come out upright. Seeking is by frame index, which decodes forward from
the previous keyframe and lands on the exact frame.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import cv2
import numpy as np

from ..media import MediaError

# Jumps shorter than this are decoded through instead of seeking.
SEQUENTIAL_SECONDS = 2.0


def sample_frames(path: str, times: Iterable[float], max_side: int | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(t, BGR frame)`` for each requested time, in ascending order."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise MediaError(f"OpenCV couldn't open {path}.")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pos = 0
    try:
        for t in sorted(set(times)):
            target = max(0, int(round(t * fps)))
            if total:
                target = min(target, total - 1)
            if target < pos or target - pos > SEQUENTIAL_SECONDS * fps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                pos = target
            while pos < target:
                if not cap.grab():
                    break
                pos += 1
            ok, frame = cap.read()
            if not ok:
                continue
            pos += 1
            if max_side and max(frame.shape[:2]) > max_side:
                scale = max_side / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            yield t, frame
    finally:
        cap.release()


def window_times(start: float, end: float, fps: float) -> list[float]:
    """Evenly spaced sample times covering [start, end]."""
    if end <= start:
        return [round(start, 3)]
    n = max(1, int((end - start) * fps)) + 1
    return [round(float(t), 3) for t in np.linspace(start, end, n)]
