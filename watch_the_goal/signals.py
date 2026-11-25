"""Turn a video file into compact time series the detector can reason about.

Everything here streams decoded data out of ffmpeg so memory stays flat even
for a 90-minute recording:

* audio loudness: RMS level in dBFS every ``AUDIO_HOP`` seconds
* motion: share of pixels that changed between consecutive low-res frames,
  for the whole frame and for each marked goal area
* pan: how far the camera turned between those frames, from the shift of the
  background behind the pitch (zero for a camera on a tripod)
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

AUDIO_RATE = 8000
AUDIO_HOP = 0.25  # seconds per loudness sample
MOTION_FPS = 5.0
MOTION_LONG_SIDE = 192
PIXEL_CHANGE_THRESHOLD = 14  # grey levels (0-255) before a pixel counts as "moving"
PAN_BAND = (0.12, 0.42)  # rows of the frame used to measure pan: trees, buildings and fences behind the pitch
PAN_MIN_RESPONSE = 0.15  # weaker matches (a dark or featureless frame) count as no pan

ProgressFn = Callable[[float], None]


@dataclass
class Signals:
    duration: float
    audio_hop: float = AUDIO_HOP
    audio_db: np.ndarray | None = None  # None when the file has no audio
    motion_hop: float = 1.0 / MOTION_FPS
    motion: np.ndarray = field(default_factory=lambda: np.zeros(0))
    goal_motion: list[np.ndarray] = field(default_factory=list)
    # Horizontal camera turn per motion frame, in frame widths; positive turns left.
    pan: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _analysis_size(width: int, height: int) -> tuple[int, int]:
    scale = MOTION_LONG_SIDE / max(width, height, 1)
    w = max(16, int(round(width * scale / 2)) * 2)
    h = max(16, int(round(height * scale / 2)) * 2)
    return w, h


def _area_mask(area: dict, w: int, h: int) -> np.ndarray:
    """Boolean mask for a normalised {x, y, w, h} rectangle."""
    x0 = int(np.clip(area["x"], 0, 1) * w)
    y0 = int(np.clip(area["y"], 0, 1) * h)
    x1 = int(np.ceil(np.clip(area["x"] + area["w"], 0, 1) * w))
    y1 = int(np.ceil(np.clip(area["y"] + area["h"], 0, 1) * h))
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = True
    return mask


def audio_loudness(path: str, duration: float, progress: ProgressFn | None = None) -> np.ndarray:
    hop = int(AUDIO_RATE * AUDIO_HOP)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(AUDIO_RATE),
         "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    levels: list[float] = []
    leftover = np.zeros(0, dtype=np.float32)
    chunk_bytes = hop * 2 * 400
    assert proc.stdout is not None
    while True:
        raw = proc.stdout.read(chunk_bytes)
        if not raw:
            break
        samples = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
        samples = np.concatenate([leftover, samples])
        n = len(samples) // hop
        if n:
            frames = samples[: n * hop].reshape(n, hop)
            rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
            levels.extend((20 * np.log10(rms)).tolist())
        leftover = samples[n * hop:]
        if progress and duration > 0:
            progress(min(1.0, len(levels) * AUDIO_HOP / duration))
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"audio decode failed: {err.decode(errors='replace').strip()[-300:]}")
    return np.asarray(levels, dtype=np.float32)


def motion_series(
    path: str, width: int, height: int, duration: float,
    goal_areas: list[dict] | None = None, progress: ProgressFn | None = None,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    import cv2

    w, h = _analysis_size(width, height)
    masks = [_area_mask(a, w, h) for a in (goal_areas or [])]
    top, bottom = int(h * PAN_BAND[0]), max(int(h * PAN_BAND[1]), int(h * PAN_BAND[0]) + 8)
    window = cv2.createHanningWindow((w, bottom - top), cv2.CV_32F)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-an",
         "-vf", f"fps={MOTION_FPS},scale={w}:{h}:flags=area,format=gray",
         "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    frame_bytes = w * h
    motion: list[float] = []
    goal: list[list[float]] = [[] for _ in masks]
    pan: list[float] = []
    prev: np.ndarray | None = None
    expected = max(1, int(duration * MOTION_FPS))
    assert proc.stdout is not None
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w).astype(np.int16)
        if prev is None:
            changed = np.zeros((h, w), dtype=bool)
            pan.append(0.0)
        else:
            changed = np.abs(frame - prev) > PIXEL_CHANGE_THRESHOLD
            (dx, _), response = cv2.phaseCorrelate(prev[top:bottom].astype(np.float32),
                                                   frame[top:bottom].astype(np.float32), window)
            pan.append(float(dx) / w if response > PAN_MIN_RESPONSE else 0.0)
        prev = frame
        motion.append(float(changed.mean()))
        for i, mask in enumerate(masks):
            goal[i].append(float(changed[mask].mean()))
        if progress and len(motion) % 25 == 0:
            progress(min(1.0, len(motion) / expected))
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"video decode failed: {err.decode(errors='replace').strip()[-300:]}")
    return (np.asarray(motion, dtype=np.float32), [np.asarray(g, dtype=np.float32) for g in goal],
            np.asarray(pan, dtype=np.float32))


def extract(
    path: str, width: int, height: int, duration: float, has_audio: bool,
    goal_areas: list[dict] | None = None, progress: ProgressFn | None = None,
) -> Signals:
    """Decode audio and video concurrently and return their time series."""
    parts = {"audio": 0.0 if has_audio else 1.0, "video": 0.0}
    lock = threading.Lock()

    def report(key: str):
        def inner(value: float) -> None:
            with lock:
                parts[key] = value
                if progress:
                    # Video decoding dominates the runtime, so weight it heavily.
                    progress(0.2 * parts["audio"] + 0.8 * parts["video"])
        return inner

    result: dict = {}

    def run_audio() -> None:
        try:
            result["audio"] = audio_loudness(path, duration, report("audio"))
        except Exception as exc:  # surfaced below
            result["audio_error"] = exc

    thread = None
    if has_audio:
        thread = threading.Thread(target=run_audio, daemon=True)
        thread.start()
    motion, goal, pan = motion_series(path, width, height, duration, goal_areas, report("video"))
    if thread:
        thread.join()
        if "audio_error" in result:
            raise result["audio_error"]
    return Signals(
        duration=duration,
        audio_db=result.get("audio"),
        motion=motion,
        goal_motion=goal,
        pan=pan,
    )


def waveform(path: str, start: float, end: float) -> list[float]:
    """A bounded envelope of the visible source range. Never decode video or hold a full recording."""
    bins = 512
    samples_per_bin = max(1, int(np.ceil((end - start) * AUDIO_RATE / bins)))
    peaks = np.zeros(bins, dtype=np.int32)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", str(start), "-i", path,
           "-t", str(end - start), "-map", "0:a:0", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "-"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        pos = 0
        while raw := proc.stdout.read(65536):
            samples = np.abs(np.frombuffer(raw, dtype="<i2").astype(np.int32))
            indexes = np.minimum((pos + np.arange(samples.size)) // samples_per_bin, bins - 1)
            np.maximum.at(peaks, indexes, samples)
            pos += samples.size
        if proc.wait():
            raise ValueError("Could not read this audio range.")
    return (peaks / 32768).round(4).tolist()


def save(sig: Signals, path, key: str = "") -> None:
    """Cache signals as .npz. ``key`` names what they were extracted with
    (e.g. the goal areas), so :func:`load` can tell a stale cache."""
    arrays = {f"goal{i:02d}": g for i, g in enumerate(sig.goal_motion)}
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(tmp, duration=sig.duration, audio_hop=sig.audio_hop, motion_hop=sig.motion_hop,
                        audio=sig.audio_db if sig.audio_db is not None else np.zeros(0, np.float32),
                        has_audio=sig.audio_db is not None, motion=sig.motion, pan=sig.pan, key=key, **arrays)
    os.replace(tmp, path)


def load(path, key: str = "") -> Signals | None:
    """Cached signals, or None if missing, unreadable, extracted with another
    ``key``, or from before pan was measured."""
    try:
        d = np.load(path, allow_pickle=False)
        if str(d["key"]) != key or "pan" not in d.files:
            return None
        return Signals(
            duration=float(d["duration"]), audio_hop=float(d["audio_hop"]), motion_hop=float(d["motion_hop"]),
            audio_db=d["audio"] if bool(d["has_audio"]) else None, motion=d["motion"],
            goal_motion=[d[k] for k in sorted(d.files) if k.startswith("goal")], pan=d["pan"],
        )
    except (OSError, KeyError, ValueError):
        return None
