"""Analysis pipeline for one recording, shared by the app, CLI and benchmarks.

1. signals      audio loudness and motion (always; no models)
2. candidates   explained moments from those signals (``detect``), with this
                recording's thresholds if the user's calls moved them
3. calls        candidates that resemble the user's calls move up or down (``coach``)
4. players      optional person detector on a bounded pool of candidates
5. select       rank, cap and merge

AI review (:mod:`watch_the_goal.review`) runs afterwards as its own step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import detect, signals

DETECTORS = ("none", "yolox", "rfdetr-nano")
POOL_FACTOR = 3  # candidates measured by the person detector, relative to the cap


@dataclass
class Options:
    detector: str = "none"
    tiles: tuple[int, int] = (1, 1)
    pre_roll: float = detect.PRE_ROLL
    post_roll: float = detect.POST_ROLL
    max_moments: int | None = None
    debug_dir: Path | None = None  # overlays and candidate timestamps go here when set
    tuning: detect.Tuning | None = None  # this recording's thresholds (None = defaults)
    calls: list[dict] = field(default_factory=list)  # the user's calls: {"t", "label"}
    play: dict | None = None  # kick-off, full time and the break (detect.play_spans); None = all of it


@dataclass
class Result:
    moments: list[detect.Moment]
    signals: signals.Signals
    stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def analyze(
    path: str,
    info: dict,
    goal_areas: list[dict] | None = None,
    options: Options | None = None,
    progress: Callable[[float, str], None] | None = None,
    sig: signals.Signals | None = None,
) -> Result:
    """Suggested moments for a long recording."""
    options = options or Options()
    goal_areas = goal_areas or []
    report = progress or (lambda f, m: None)
    stats: dict = {"detector": options.detector, "duration": info["duration"]}
    notes: list[str] = []

    t0 = time.time()
    if sig is None:
        report(0.0, "Listening and watching")
        weight = 0.7 if options.detector != "none" else 0.95
        sig = signals.extract(path, info["width"], info["height"], info["duration"], info["has_audio"],
                              goal_areas, progress=lambda f: report(f * weight, "Listening and watching"))
    stats["signals_seconds"] = round(time.time() - t0, 2)

    spans = detect.play_spans(options.play, sig.duration)
    moments = detect.candidate_moments(sig, goal_areas, options.pre_roll, options.post_roll, options.tuning,
                                       spans)
    stats["candidates"] = len(moments)
    if options.calls:
        from . import coach

        stats["moved_by_calls"] = coach.resemble(coach.Evidence(sig, goal_areas), moments, options.calls)
    cap = options.max_moments or detect.default_cap(detect.play_seconds(spans))
    evidence: list = []

    if options.detector != "none" and moments:
        from .vision import detectors, models, players

        try:
            det = detectors.load(options.detector, tiles=options.tiles)
        except models.ModelUnavailable as exc:
            notes.append(f"Player detector off: {exc}")
            det = None
        if det is not None:
            pool = moments[: cap * POOL_FACTOR]
            labels = [detect.area_label(a, i) for i, a in enumerate(goal_areas)]
            t1 = time.time()
            evidence, pstats = players.measure(
                path, sig.duration, [m.anchor for m in pool], det, goal_areas, labels,
                progress=lambda f: report(0.7 + 0.28 * f, f"Looking for players ({options.detector})"))
            stats["players_seconds"] = round(time.time() - t1, 2)
            stats["player_frames"] = pstats["frames"]
            stats["players_typical"] = {"near_goal": pstats["near_goal_typical"], "bunch": pstats["bunch_typical"]}
            kept = detect.apply_player_evidence(pool, evidence, options.post_roll, sig.duration)
            stats["dropped_by_players"] = len(pool) - len(kept)  # e.g. someone right at the lens
            moments = kept + moments[len(pool):]
    selected = detect.select(moments, cap)
    stats["moments"] = len(selected)
    stats["total_seconds"] = round(time.time() - t0, 2)

    if options.debug_dir:
        save_debug(Path(options.debug_dir), path, selected, moments, evidence, goal_areas, stats)
    report(1.0, "Done")
    return Result(selected, sig, stats, notes)


def save_debug(out: Path, path: str, selected, candidates, evidence, goal_areas, stats) -> None:
    """Candidate timestamps as JSON, and an overlay of the busiest frame per suggestion."""
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "video": path,
        "stats": stats,
        "suggestions": [m.to_dict() for m in selected],
        "candidates": [m.to_dict() | {"strength": round(m.strength, 3)} for m in candidates],
        "evidence": [e.to_dict() for e in evidence],
    }
    (out / "candidates.json").write_text(json.dumps(record, indent=1))
    if not evidence:
        return
    from .vision import overlay

    for m in selected:
        # The frame with the most players near the goal (or bunched), else the anchor.
        ev = next((e for e in evidence if e.detections and
                   min(f.t for f in e.detections) <= m.anchor <= max(f.t for f in e.detections) + 0.01), None)
        if ev is None:
            continue
        t = ev.near_goal_t if ev.near_goal_t is not None else ev.bunch_t
        f = min(ev.detections, key=lambda d: abs(d.t - (t if t is not None else m.anchor)))
        frame = next((img for _, img in _one_frame(path, f.t)), None)
        if frame is None:
            continue
        caption = f"#{m.rank} {f.t:.1f}s  " + "; ".join(m.reasons)[:110]
        overlay.save(out / f"rank{m.rank:02d}_{f.t:08.1f}s.jpg", overlay.draw(frame, f.boxes, goal_areas, caption))


def _one_frame(path: str, t: float):
    from .vision.frames import sample_frames

    return sample_frames(path, [t])
