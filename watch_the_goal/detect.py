"""Baseline highlight detector.

Works on the time series from :mod:`signals` and returns *candidate* moments,
each with the human-readable reasons that produced it. It never invents a
confidence score: ``strength`` is only used internally to rank and cap.

See docs/DETECTION.md for the reasoning and ideas for improving it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .signals import Signals

STEP = 0.5  # seconds per sample on the common analysis grid
BASELINE_WINDOW = 60.0  # seconds of context for "normal" levels
PRE_ROLL = 12.0  # default buildup kept before the moment
MAX_PRE_ROLL = 20.0  # a move that is still in full flow at the cut may pull the start back this far
SURGE = 1.3  # whole-frame motion above this multiple of typical counts as play in full flow
POST_ROLL = 5.0  # default time kept after the moment (up to twice this for a long reaction)
MERGE_GAP = 1.0  # suggestions closer than this are the same moment
CLUSTER_WINDOW = 6.0  # signals this close together describe one moment

AUDIO_Z = 3.5
AUDIO_MIN_DB = 6.0
AUDIO_MIN_SECONDS = 0.75
GOAL_Z = 3.5
GOAL_MIN_SHARE = 0.02
LULL_RATIO = 0.6
CAMERA_WINDOW = 180.0  # seconds of context for where an auto-panning camera usually points
CAMERA_MIN_SWEEP = 0.15  # frame widths; a camera that turns less than this isn't following play
ATTACK_END = 0.75  # how far towards one end of its sweep the camera must point (1 = the end)
ATTACK_SECONDS = 3.0  # held at least this long
ATTACK_SUPPORT = (-4.0, 14.0)  # an attack ending this close to noise or goal-box motion is part of that moment


@dataclass(frozen=True)
class Tuning:
    """Thresholds for one recording. The defaults are the hand-picked constants
    above; the user's calls on a recording (:mod:`watch_the_goal.coach`) may
    move them for that recording only."""

    audio_min_db: float = AUDIO_MIN_DB
    audio_z: float = AUDIO_Z
    goal_z: float = GOAL_Z
    goal_min_share: float = GOAL_MIN_SHARE

    @classmethod
    def from_dict(cls, d: dict | None) -> "Tuning":
        d = d or {}
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if d.get(k) is not None})

    def to_dict(self) -> dict:
        return {k: round(getattr(self, k), 3) for k in self.__dataclass_fields__}


DEFAULT_TUNING = Tuning()


@dataclass
class Moment:
    anchor: float
    start: float
    end: float
    reasons: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    strength: float = 0.0
    rank: int | None = None  # 1 = strongest suggestion in its recording

    def to_dict(self) -> dict:
        return {
            "anchor": round(self.anchor, 2),
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "reasons": list(self.reasons),
            "kinds": list(self.kinds),
            "rank": self.rank,
        }


# --------------------------------------------------------------------------
# Small numeric helpers


def to_grid(series: np.ndarray, hop: float, n: int, how: str = "max") -> np.ndarray:
    """Resample ``series`` (one value every ``hop`` s) onto ``n`` STEP-sized bins."""
    out = np.zeros(n, dtype=np.float64)
    if len(series) == 0 or n == 0:
        return out
    idx = np.minimum((np.arange(len(series)) * hop / STEP).astype(int), n - 1)
    if how == "max":
        out.fill(-np.inf)
        np.maximum.at(out, idx, series)
        out[np.isinf(out)] = np.nan
    else:
        sums = np.bincount(idx, weights=series, minlength=n)
        counts = np.bincount(idx, minlength=n)
        with np.errstate(invalid="ignore"):
            out = sums / counts
    # Fill gaps (e.g. hop > STEP) from the previous value.
    mask = np.isnan(out)
    if mask.all():
        return np.zeros(n)
    if mask.any():
        valid = np.where(~mask, np.arange(n), 0)
        np.maximum.accumulate(valid, out=valid)
        out = out[valid]
        out[np.isnan(out)] = np.nanmin(out)
    return out


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return x
    window = max(1, min(window | 1, (len(x) // 2) * 2 + 1))
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(view, axis=1)


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0 or window <= 1:
        return x.astype(np.float64)
    kernel = np.ones(window) / window
    return np.convolve(np.pad(x, (window // 2, window - 1 - window // 2), mode="edge"), kernel, "valid")


def robust_z(x: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    resid = x - baseline
    scale = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    return resid / max(scale, 1e-6)


def runs(mask: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    """Index ranges [a, b] where ``mask`` is true, bridging gaps <= max_gap."""
    out: list[tuple[int, int]] = []
    for i in np.flatnonzero(mask):
        if out and i - out[-1][1] <= max_gap + 1:
            out[-1] = (out[-1][0], int(i))
        else:
            out.append((int(i), int(i)))
    return out


def area_label(area: dict, index: int) -> str:
    if area.get("label"):
        return str(area["label"])
    if "x" in area:
        return "left goal" if area["x"] + area.get("w", 0) / 2 < 0.5 else "right goal"
    return f"goal area {index + 1}"


# --------------------------------------------------------------------------
# Signal events


@dataclass
class Event:
    kind: str  # "audio" | "goal" | "attack" | "lull"
    t: float  # best estimate of when the moment happened
    strength: float
    reason: str
    end: float | None = None  # when the evidence stops (for sustained noise)


def audio_levels(sig: Signals) -> tuple[np.ndarray, np.ndarray] | None:
    """Loudness above the local level, in dB and in robust deviations, one
    value per ``sig.audio_hop``. None without audio."""
    if sig.audio_db is None or len(sig.audio_db) == 0:
        return None
    level = rolling_mean(sig.audio_db.astype(np.float64), 2)
    base = rolling_median(level, int(BASELINE_WINDOW / sig.audio_hop))
    return level - base, robust_z(level, base)


def audio_events(sig: Signals, n: int, tuning: Tuning = DEFAULT_TUNING) -> list[Event]:
    """Sustained loudness well above the local level: cheers, shouting, celebrations.

    Works on the raw loudness series so a ball strike or a single call, which
    lasts a fraction of a second, isn't mistaken for a celebration.
    """
    levels = audio_levels(sig)
    if levels is None:
        return []
    hop = sig.audio_hop
    delta, z = levels
    hot = (z > tuning.audio_z) & (delta > tuning.audio_min_db)
    events = []
    for a, b in runs(hot, max_gap=1):
        if (b - a + 1) * hop < AUDIO_MIN_SECONDS:
            continue
        peak = a + int(np.argmax(delta[a:b + 1]))
        events.append(Event(
            kind="audio",
            t=max(0.0, a * hop - 0.5),  # the play usually happens just before the noise
            strength=float(delta[peak]) / 6.0,
            reason=f"crowd noise +{delta[peak]:.0f} dB above the surrounding level for {(b - a + 1) * hop:.1f}s",
            end=b * hop,
        ))
    return events


def goal_levels(sig: Signals, n: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    """Per goal area on the STEP grid: share of the box moving above its local
    level, the same in robust deviations, and the raw share. None for an
    empty series."""
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    for series in sig.goal_motion:
        if len(series) == 0:
            out.append(None)
            continue
        share = rolling_mean(to_grid(series, sig.motion_hop, n, "mean"), 2)
        base = rolling_median(share, int(BASELINE_WINDOW / STEP))
        out.append((share - base, robust_z(share, base), share))
    return out


def goal_events(sig: Signals, n: int, goal_areas: list[dict], tuning: Tuning = DEFAULT_TUNING) -> list[Event]:
    events = []
    for i, levels in enumerate(goal_levels(sig, n)):
        if levels is None:
            continue
        extra, z, share = levels
        hot = (z > tuning.goal_z) & (extra > tuning.goal_min_share)
        label = area_label(goal_areas[i] if i < len(goal_areas) else {}, i)
        for a, b in runs(hot, max_gap=4):
            if (b - a + 1) * STEP < 1.0:
                continue  # a single flicker is usually compression noise or a passer-by
            peak = a + int(np.argmax(share[a:b + 1]))
            events.append(Event(
                kind="goal",
                t=peak * STEP,
                # Ranking stays on the default scale, so tuning changes what
                # counts, not how strong it looks.
                strength=1.0 + min(3.0, float(z[peak]) / GOAL_Z),
                reason=f"motion near the marked {label}",
            ))
    return events


def flow(sig: Signals, n: int) -> np.ndarray | None:
    """Whole-frame motion on the STEP grid, smoothed over 2 s."""
    if len(sig.motion) == 0:
        return None
    return rolling_mean(to_grid(sig.motion, sig.motion_hop, n, "mean"), int(2 / STEP))


def lull_events(sig: Signals, n: int) -> list[Event]:
    """Burst of play followed by a clear drop in movement (ball out, goal, restart)."""
    m = flow(sig, n)
    if m is None:
        return []
    before_w, gap, after_w = int(8 / STEP), int(3 / STEP), int(10 / STEP)
    if n < before_w + gap + after_w + 1:
        return []
    typical = float(np.median(m))
    csum = np.concatenate([[0.0], np.cumsum(m)])
    t = np.arange(before_w, n - gap - after_w)
    before = (csum[t] - csum[t - before_w]) / before_w
    after = (csum[t + gap + after_w] - csum[t + gap]) / after_w
    ratio = after / np.maximum(before, 1e-6)
    hot = (ratio < LULL_RATIO) & (before > typical * 0.9) & (before > 1e-4)
    events = []
    for a, b in runs(hot, max_gap=4):
        best = a + int(np.argmin(ratio[a:b + 1]))
        events.append(Event(
            kind="lull",
            t=float(t[best]) * STEP,
            strength=0.6,
            reason="play stopped after a burst of action",
        ))
    return events


# --------------------------------------------------------------------------
# Moments


def reason_kind(reason: str) -> str:
    """The kind of evidence a reason describes, without its numbers."""
    return re.sub(r"[-+]?\d+(\.\d+)?", "#", reason)


def merge_overlapping(moments: list[Moment], gap: float = MERGE_GAP) -> list[Moment]:
    """Combine moments whose windows overlap (or nearly touch) into one."""
    merged: list[Moment] = []
    for m in sorted(moments, key=lambda m: (m.start, m.end)):
        if merged and m.start <= merged[-1].end + gap:
            cur = merged[-1]
            cur.end = max(cur.end, m.end)
            cur.anchor = min(cur.anchor, m.anchor)
            cur.strength = max(cur.strength, m.strength)
            for r in m.reasons:
                # One reason per kind of evidence ("crowd noise +12 dB..." and
                # "crowd noise +9 dB..." are the same kind).
                if reason_kind(r) not in {reason_kind(x) for x in cur.reasons}:
                    cur.reasons.append(r)
            for k in m.kinds:
                if k not in cur.kinds:
                    cur.kinds.append(k)
            if m.rank is not None:
                cur.rank = m.rank if cur.rank is None else min(cur.rank, m.rank)
        else:
            merged.append(Moment(m.anchor, m.start, m.end, list(m.reasons), list(m.kinds), m.strength, m.rank))
    return merged


def clamp_window(start: float, end: float, duration: float, min_len: float = 1.0) -> tuple[float, float]:
    """Clamp [start, end] into the file while keeping at least ``min_len`` seconds."""
    min_len = min(min_len, duration)
    start = max(0.0, min(start, duration - min_len))
    end = min(duration, max(end, start + min_len))
    return round(start, 3), round(end, 3)


def lead_in(motion: np.ndarray | None, anchor: float, pre_roll: float, max_pre_roll: float = MAX_PRE_ROLL) -> float:
    """Start of the clip: ``pre_roll`` before the moment, or earlier if play
    is in full flow at that cut, back to where the surge began."""
    start = anchor - pre_roll
    if motion is None or len(motion) == 0 or max_pre_roll <= pre_roll:
        return start
    typical = float(np.median(motion))
    k = int(start / STEP)
    if typical <= 0 or not 0 < k < len(motion) or motion[k] < typical * SURGE:
        return start
    limit = max(0, int(math.ceil((anchor - max_pre_roll) / STEP)))
    while k > limit and motion[k - 1] >= typical * SURGE:
        k -= 1
    return k * STEP


def camera_offset(sig: Signals, n: int) -> np.ndarray | None:
    """Where an auto-panning camera (Veo, Trace, Pixellot...) points, on the
    STEP grid: 0 at the middle of its sweep over the surrounding few minutes,
    about +1 at its left end and -1 at its right. Such cameras follow the
    ball, so this says which end play is at. None for a fixed camera."""
    if len(sig.pan) == 0 or n < 4:
        return None
    pos = to_grid(np.cumsum(sig.pan.astype(np.float64)), sig.motion_hop, n, "mean")
    half = int(CAMERA_WINDOW / STEP) // 2
    at = np.arange(0, n, 4)
    windows = [pos[max(0, i - half):i + half + 1] for i in at]
    lo = np.interp(np.arange(n), at, [np.percentile(w, 5) for w in windows])
    hi = np.interp(np.arange(n), at, [np.percentile(w, 95) for w in windows])
    if float(np.median(hi - lo)) < CAMERA_MIN_SWEEP:
        return None
    return (pos - (lo + hi) / 2) / np.maximum((hi - lo) / 2, CAMERA_MIN_SWEEP / 2)


def attack_events(sig: Signals, n: int) -> list[Event]:
    """An auto-panning camera held at one end while play was busy: play in a
    final third. Each event's time is when the camera left that end. On a
    phone or tripod the camera doesn't move, so this never fires."""
    offset = camera_offset(sig, n)
    m = flow(sig, n)
    if offset is None or m is None:
        return []
    hot = (np.abs(offset) > ATTACK_END) & (m > float(np.median(m)) * 0.9)
    events = []
    for a, b in runs(hot, max_gap=4):
        held = (b - a + 1) * STEP
        end = b * STEP
        if held < ATTACK_SECONDS:
            continue
        side = "left" if offset[a:b + 1].mean() > 0 else "right"
        events.append(Event(
            kind="attack",
            t=end,
            strength=0.5,
            reason=f"the camera followed play to the {side} end for {held:.0f}s",
        ))
    return events


def all_events(sig: Signals, goal_areas: list[dict] | None = None,
               tuning: Tuning = DEFAULT_TUNING) -> tuple[list[Event], list[Event], list[Event]]:
    """Audio, goal-area and lull events, before they are grouped into moments."""
    n = int(math.ceil(sig.duration / STEP))
    if n < 4:
        return [], [], []
    return audio_events(sig, n, tuning), goal_events(sig, n, goal_areas or [], tuning), lull_events(sig, n)


def play_spans(play: dict | None, duration: float) -> list[tuple[float, float]]:
    """The parts of a recording that are the match: kick-off to full time,
    without the halftime break. No ``play`` means all of it."""
    if not play:
        return [(0.0, duration)]
    start = min(max(0.0, float(play.get("start") or 0.0)), duration)
    end = min(duration, float(play["end"])) if play.get("end") is not None else duration
    spans = [(start, end)]
    if play.get("break"):
        a, b = (min(max(float(x), start), end) for x in play["break"])
        spans = [(start, a), (b, end)]
    return [(a, b) for a, b in spans if b > a]


def play_seconds(spans: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in spans)


def candidate_moments(
    sig: Signals,
    goal_areas: list[dict] | None = None,
    pre_roll: float = PRE_ROLL,
    post_roll: float = POST_ROLL,
    tuning: Tuning | None = None,
    spans: list[tuple[float, float]] | None = None,
) -> list[Moment]:
    """Every explained moment, strongest evidence first, before capping or merging.

    ``spans`` (see play_spans) keeps only moments during play, and keeps each
    clip inside the half it happened in."""
    goal_areas = goal_areas or []
    n = int(math.ceil(sig.duration / STEP))
    if n < 4:
        return []
    audio, goals, lulls = all_events(sig, goal_areas, tuning or DEFAULT_TUNING)
    attacks = attack_events(sig, n)
    smooth = flow(sig, n)

    # Primary evidence starts a moment; lulls only support one, unless the
    # footage has no audio, no goal areas and a fixed camera, where they are all we have.
    primary = audio + goals
    if not primary and not attacks and sig.audio_db is None and not goal_areas:
        primary = lulls
        lulls = []

    clusters: list[list[Event]] = []
    for ev in sorted(primary, key=lambda e: e.t):
        if clusters and ev.t - clusters[-1][0].t <= CLUSTER_WINDOW:
            clusters[-1].append(ev)
        else:
            clusters.append([ev])
    # An attack joins the noise or goal-box motion it led to. On its own it is
    # a moment only if play stopped right after (a shot out, a save, a goal).
    for ev in attacks:
        home = next((c for c in clusters
                     if c[0].t + ATTACK_SUPPORT[0] <= ev.t <= c[-1].t + ATTACK_SUPPORT[1]), None)
        if home is not None:
            home.append(ev)
        else:
            clusters.append([ev])
    clusters.sort(key=lambda c: min(e.t for e in c))

    moments: list[Moment] = []
    for cluster in clusters:
        goal_hits = [e for e in cluster if e.kind == "goal"]
        strongest = max(cluster, key=lambda e: e.strength)
        # The finish: motion in the goal mouth if we saw it, otherwise just
        # before the loudest reaction.
        anchor = min(e.t for e in goal_hits) if goal_hits else strongest.t
        half = next(((a, b) for a, b in spans if a <= anchor <= b), None) if spans is not None else (0.0, sig.duration)
        if half is None:
            continue  # warm-up, halftime or after the final whistle
        support = [e for e in lulls if anchor - 2 <= e.t <= anchor + 14]
        evidence = cluster + support[:1]
        # With goals marked, noise with no action near either goal and no
        # stoppage afterwards is more likely an argument than a goal.
        if goal_areas and not goal_hits and not support:
            continue
        # Play reaches a final third every minute or so; on its own that is a
        # moment only if play stopped right after (a shot out, a save, a goal).
        if all(e.kind == "attack" for e in cluster) and not support:
            continue
        # One reason per kind of evidence, from its strongest instance.
        reasons: list[str] = []
        for kind in ("goal", "audio", "attack", "lull"):
            of_kind = [e for e in evidence if e.kind == kind]
            if kind == "goal":
                labels = [e.reason for e in of_kind]
                reasons += [r for i, r in enumerate(labels) if r not in labels[:i]]
            elif of_kind:
                reasons.append(max(of_kind, key=lambda e: e.strength).reason)
        kinds = sorted({e.kind for e in evidence})
        strength = len(kinds) + max(e.strength for e in evidence)
        # Keep the reaction too, but don't let a long celebration swamp the reel.
        last = max(max(e.t, e.end or e.t) for e in cluster)
        tail = min(max(anchor + post_roll, last + 1.0), anchor + 2 * post_roll)
        start, end = clamp_window(lead_in(smooth, anchor, pre_roll), tail, sig.duration)
        start, end = max(start, half[0]), min(end, half[1])
        moments.append(Moment(anchor, start, end, reasons, kinds, strength))
    return sorted(moments, key=lambda m: -m.strength)


def default_cap(duration: float) -> int:
    return max(4, math.ceil(duration / 90))


def select(moments: list[Moment], max_moments: int) -> list[Moment]:
    """Keep the strongest ``max_moments``, number them by strength, merge overlaps."""
    ranked = sorted(moments, key=lambda m: -m.strength)[:max_moments]
    for i, m in enumerate(ranked):
        m.rank = i + 1
    return merge_overlapping(ranked)


def find_moments(
    sig: Signals,
    goal_areas: list[dict] | None = None,
    pre_roll: float = PRE_ROLL,
    post_roll: float = POST_ROLL,
    max_moments: int | None = None,
    tuning: Tuning | None = None,
) -> list[Moment]:
    moments = candidate_moments(sig, goal_areas, pre_roll, post_roll, tuning)
    return select(moments, default_cap(sig.duration) if max_moments is None else max_moments)


# --------------------------------------------------------------------------
# Player evidence (from the optional person detector)

PLAYERS_NEAR_GOAL_MIN = 2  # at least this many players near a marked goal...
PLAYERS_NEAR_GOAL_EXTRA = 2  # ...and this many more than usual
BUNCH_MIN = 4
BUNCH_EXTRA = 3


def apply_player_evidence(moments: list[Moment], evidence: list, post_roll: float = POST_ROLL,
                          duration: float | None = None) -> list[Moment]:
    """Fold per-moment player evidence (see :mod:`watch_the_goal.vision.players`)
    into reasons and strength. ``evidence[i]`` belongs to ``moments[i]``; None
    means it wasn't measured. Returns the moments that survive."""
    out = []
    for m, ev in zip(moments, evidence, strict=True):
        if ev is None:
            out.append(m)
            continue
        if ev.lens_blocked and "goal" in m.kinds:
            # The goal box "moved" because someone walked past the camera.
            m.reasons = [r for r in m.reasons if not r.startswith("motion near the marked")]
            m.kinds = [k for k in m.kinds if k != "goal"]
            m.strength -= 2.0
            if "audio" not in m.kinds:
                continue
        if (ev.near_goal_label and ev.near_goal_peak >= PLAYERS_NEAR_GOAL_MIN
                and ev.near_goal_peak >= ev.near_goal_typical + PLAYERS_NEAR_GOAL_EXTRA):
            m.reasons.append(f"{ev.near_goal_peak} players near the {ev.near_goal_label} "
                             f"(usually {round(ev.near_goal_typical)})")
            m.strength += 1.0 + min(1.0, (ev.near_goal_peak - ev.near_goal_typical) / 4)
            if "players" not in m.kinds:
                m.kinds.append("players")
            # Make sure the clip runs past the goalmouth action it found.
            if ev.near_goal_t is not None and ev.near_goal_t + post_roll / 2 > m.end:
                m.end = round(min(duration or float("inf"), ev.near_goal_t + post_roll / 2), 3)
        elif ev.bunch_peak >= BUNCH_MIN and ev.bunch_peak >= ev.bunch_typical + BUNCH_EXTRA:
            m.reasons.append(f"{ev.bunch_peak} players bunched together (usually {round(ev.bunch_typical)})")
            m.strength += 0.5 + min(1.0, (ev.bunch_peak - ev.bunch_typical) / 6)
            if "players" not in m.kinds:
                m.kinds.append("players")
        out.append(m)
    return out


def suggest_trim(sig: Signals, goal_areas: list[dict] | None = None,
                 pre_roll: float = 8.0, post_roll: float = 4.0,
                 min_saving: float = 3.0) -> Moment | None:
    """For a short clip: the strongest moment, if trimming to it saves time.

    Only audio or goal-area evidence counts here. A handheld phone pans and
    zooms, so a drop in whole-frame motion says little about where the goal is,
    and a wrong trim would cut the very moment the clip was filmed for.
    """
    moments = [m for m in find_moments(sig, goal_areas, pre_roll, post_roll, max_moments=1)
               if "audio" in m.kinds or "goal" in m.kinds]
    if not moments:
        return None
    best = moments[0]
    if sig.duration - (best.end - best.start) < min_saving:
        return None
    return best
