"""Learn from the user's calls on one recording.

A *call* is the user saying what really happened at a moment: a goal, a save,
a near miss, or not a highlight. Calls are made on a suggestion, or at the
playhead for something detection missed. From them this module:

* reads what the signals showed around any moment and says why it was or
  wasn't suggested (:meth:`Evidence.explain`);
* moves one recording's thresholds just far enough to catch highlights it
  missed, or to drop false alarms without losing a confirmed highlight
  (:func:`retune`);
* ranks candidates that resemble a call up or down, and names the call they
  resemble (:func:`resemble`);
* gives the live view a per-half-second readout of every signal next to the
  bar it has to clear (:meth:`Evidence.timeline`).

Like the detector, it never states a confidence. Every change comes with the
numbers behind it, so the user can check them against the footage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import detect
from .detect import DEFAULT_TUNING, STEP, Tuning
from .signals import Signals

LABELS = ("goal", "save", "near_miss", "chance", "tackle", "other")
HIGHLIGHTS = ("goal", "save", "near_miss", "chance", "tackle")
LABEL_TEXT = {"goal": "goal", "save": "save", "near_miss": "near miss", "chance": "chance",
              "tackle": "tackle", "other": "not a highlight"}
# What a reel is made of: a save and a near miss are both shots.
KIND = {"goal": "goal", "save": "shot", "near_miss": "shot", "chance": "chance", "tackle": "tackle"}
# A tackle is over in a moment and needs little buildup; the rest use the project's pre/post roll.
TAG_ROLL = {"tackle": (6.0, 4.0)}

# Where a moment's evidence is looked for, relative to its time. The crowd
# reacts after the play, and people mark a missed goal a little late.
AUDIO_WINDOW = (-4.0, 8.0)
GOAL_WINDOW = (-4.0, 3.0)
LULL_WINDOW = (-2.0, 14.0)  # same as the detector's "supporting stoppage" rule
LOUD_DB = 3.0  # "loud seconds" counts time at least this far above the local level
GOAL_RUN_SECONDS = 1.0  # goal-box motion must last this long (as in detect.goal_events)

# How far calls may move a recording's thresholds, as a share of the defaults.
MIN_SCALE = 0.5
MAX_SCALE = 2.0
MARGIN = 0.95  # a lowered bar sits a little under the evidence it must catch
RAISE_STEP = 1.03  # a raised bar sits a little over the false alarm it must drop

RESEMBLE_RADIUS = 1.0  # robust-standardised distance that still counts as "like"
RESEMBLE_NEIGHBOURS = 3
RESEMBLE_BONUS = 1.5  # strength added or removed; about one extra kind of evidence
NEAR_CALL = 15.0  # candidates this close to a call are that call's moment, not a lookalike


def fmt_time(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60}:{t % 60:02d}" if t < 3600 else f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


def _best_run(ratio: np.ndarray, k: int) -> float:
    """Highest value held for ``k`` consecutive samples."""
    if len(ratio) == 0:
        return 0.0
    if len(ratio) < k:
        return float(ratio.min())
    return float(np.lib.stride_tricks.sliding_window_view(ratio, k).min(axis=1).max())


@dataclass
class Readout:
    """What the signals showed around one moment, against default thresholds.

    A ratio is the best run's evidence as a share of the bar it had to clear
    (1.0 = exactly at the bar), so scaling a bar by ``s`` catches the moment
    when ``ratio >= s``."""

    t: float
    audio_db: float | None = None  # rise above the local level, held for AUDIO_MIN_SECONDS
    audio_peak_db: float | None = None
    audio_needs_db: float | None = None  # what counted at default thresholds, in dB
    audio_ratio: float | None = None
    loud_seconds: float = 0.0
    goal_ratios: list[float] = field(default_factory=list)  # per marked area
    lull: bool = False
    flow_after: float | None = None  # motion after / before (lower = play stopped)

    @property
    def goal_ratio(self) -> float | None:
        return max(self.goal_ratios) if self.goal_ratios else None

    def features(self) -> np.ndarray:
        """What :func:`resemble` compares: loudness, its length, goal-mouth
        movement and the drop in play afterwards."""
        return np.array([self.audio_peak_db or 0.0, self.loud_seconds, self.goal_ratio or 0.0,
                         self.flow_after if self.flow_after is not None else 1.0])

    def to_dict(self) -> dict:
        r = lambda v, d=2: None if v is None else round(float(v), d)  # noqa: E731
        return {"t": round(self.t, 2), "audio_db": r(self.audio_db, 1), "audio_peak_db": r(self.audio_peak_db, 1),
                "audio_needs_db": r(self.audio_needs_db, 1), "audio_ratio": r(self.audio_ratio),
                "loud_seconds": round(self.loud_seconds, 2), "goal_ratios": [round(g, 2) for g in self.goal_ratios],
                "lull": self.lull, "flow_after": r(self.flow_after)}


class Evidence:
    """The detector's signal levels for one recording, computed once."""

    def __init__(self, sig: Signals, goal_areas: list[dict] | None = None):
        self.sig = sig
        self.goal_areas = goal_areas or []
        self.n = int(math.ceil(sig.duration / STEP))
        self.audio = detect.audio_levels(sig)
        self.goals = [g for g in detect.goal_levels(sig, self.n) if g is not None] if self.n >= 4 else []
        self.flow = detect.flow(sig, self.n) if self.n >= 4 else None
        self.labels = [detect.area_label(a, i) for i, a in enumerate(self.goal_areas)][: len(self.goals)]
        self._lulls = detect.lull_events(sig, self.n) if self.n >= 4 else []

    # ----------------------------------------------------------- thresholds

    def audio_needs_db(self, tuning: Tuning = DEFAULT_TUNING) -> float | None:
        """Rise in dB that counts as crowd noise in this recording. Both rules
        apply (dB above the local level and robust deviations), so it's the
        stricter of the two."""
        if self.audio is None:
            return None
        delta = self.audio[0]
        med = float(np.median(delta))
        scale = 1.4826 * float(np.median(np.abs(delta - med)))
        return max(tuning.audio_min_db, med + tuning.audio_z * max(scale, 1e-6))

    def _audio_ratio(self, tuning: Tuning) -> np.ndarray:
        delta, z = self.audio
        return np.minimum(delta / tuning.audio_min_db, z / tuning.audio_z)

    def _goal_ratio(self, i: int, tuning: Tuning) -> np.ndarray:
        extra, z, _ = self.goals[i]
        return np.minimum(z / tuning.goal_z, extra / tuning.goal_min_share)

    # -------------------------------------------------------------- readout

    def readout(self, t: float, tuning: Tuning = DEFAULT_TUNING) -> Readout:
        out = Readout(t=t)
        if self.audio is not None:
            hop = self.sig.audio_hop
            delta, _ = self.audio
            a = max(0, int((t + AUDIO_WINDOW[0]) / hop))
            b = min(len(delta), int((t + AUDIO_WINDOW[1]) / hop) + 1)
            k = max(1, int(math.ceil(detect.AUDIO_MIN_SECONDS / hop)))
            if b > a:
                out.audio_db = _best_run(delta[a:b], k)
                out.audio_peak_db = float(delta[a:b].max())
                out.audio_ratio = max(0.0, _best_run(self._audio_ratio(tuning)[a:b], k))
                out.loud_seconds = float((delta[a:b] > LOUD_DB).sum() * hop)
            out.audio_needs_db = self.audio_needs_db(tuning)
        a = max(0, int((t + GOAL_WINDOW[0]) / STEP))
        b = min(self.n, int((t + GOAL_WINDOW[1]) / STEP) + 1)
        k = max(1, int(math.ceil(GOAL_RUN_SECONDS / STEP)))
        out.goal_ratios = [max(0.0, _best_run(self._goal_ratio(i, tuning)[a:b], k)) if b > a else 0.0
                           for i in range(len(self.goals))]
        out.lull = any(t + LULL_WINDOW[0] <= e.t <= t + LULL_WINDOW[1] for e in self._lulls)
        if self.flow is not None:
            i = int(t / STEP)
            before = self.flow[max(0, i - int(8 / STEP)):i + 1]
            after = self.flow[min(self.n, i + int(3 / STEP)):min(self.n, i + int(13 / STEP))]
            if len(before) and len(after) and before.mean() > 0:
                out.flow_after = float(after.mean() / before.mean())
        return out

    def explain(self, t: float, tuning: Tuning, clips: list[dict], candidates: list[detect.Moment],
                cap: int, short: bool = False, label: str | None = None) -> dict:
        """What the detector saw at ``t`` and why it did or didn't suggest it,
        in words. With the user's ``label``, says whether that agrees."""
        r = self.readout(t, tuning)
        lines = []
        if self.audio is None:
            lines.append("No audio in this file, so crowd noise can't help.")
        elif r.audio_db is not None:
            held = f"{detect.AUDIO_MIN_SECONDS:g} s"
            lines.append(f"Crowd noise rose {r.audio_db:+.1f} dB (held {held}, peak {r.audio_peak_db:+.1f} dB). "
                         f"Here it needs {r.audio_needs_db:+.1f} dB.")
        if not self.goals:
            if not short:
                lines.append("No goals marked, so movement in the goal mouth wasn't measured. "
                             "For a fixed camera, Mark goals helps.")
        else:
            for label, ratio in zip(self.labels, r.goal_ratios):
                lines.append(f"Movement at the {label} reached {ratio:.0%} of what counts.")
        if self.flow is not None:
            lines.append("Play stopped afterwards." if r.lull else "Play didn't clearly stop afterwards.")

        # The verdict is what the detector does with the current bars, whether
        # or not a clip the user kept, skipped or added already covers the moment.
        verdict, rank = "missed", None
        shown = [c for c in clips if c["start"] <= t <= c["end"] and c.get("origin") == "detected"
                 and c.get("status") == "candidate"]
        hit = next(((i, m) for i, m in enumerate(candidates) if m.start <= t <= m.end), None)
        if shown:
            verdict, rank = "suggested", min(c.get("rank") or 1 << 30 for c in shown)
            rank = None if rank == 1 << 30 else rank
            lines.append(f"Suggested{f' as #{rank}' if rank else ''}.")
        elif hit and hit[0] < cap:
            verdict, rank = "suggested", hit[0] + 1
            lines.append(f"The detector finds this (#{rank} by strength).")
        elif hit:
            verdict, rank = "below_cap", hit[0] + 1
            lines.append(f"It's a candidate, but #{rank} by strength. Only the strongest {cap} are suggested.")
        elif (r.audio_ratio or 0) >= 1 and self.goals:
            verdict = "filtered"
            lines.append("The noise cleared the bar, but with goals marked, noise also needs goal-mouth "
                         "movement or a stoppage after it.")
        else:
            lines.append("Nothing here cleared the bars"
                         + (", which agrees with your call." if label == "other" else ", so it was missed."))
        if label == "other" and verdict == "suggested":
            lines[-1] += " Re-tune can raise the bar past it, if a highlight you confirm is clearer."
        return {"t": round(t, 2), "verdict": verdict, "rank": rank, "lines": lines, "readout": r.to_dict()}

    # ------------------------------------------------------------- timeline

    def timeline(self, tuning: Tuning = DEFAULT_TUNING) -> dict:
        """Every signal on the STEP grid next to its bar, for the live view."""
        def grid(series: np.ndarray, hop: float) -> np.ndarray:
            return detect.to_grid(series.astype(np.float64), hop, self.n, "max")

        out: dict = {"step": STEP, "duration": self.sig.duration, "tuning": tuning.to_dict(),
                     "default_tuning": DEFAULT_TUNING.to_dict(), "surge": detect.SURGE}
        if self.audio is not None:
            out["audio_db"] = np.round(grid(self.audio[0], self.sig.audio_hop), 1).tolist()
            ratio = grid(self._audio_ratio(tuning), self.sig.audio_hop)
            out["audio_ratio"] = np.round(np.clip(ratio, -9, 9), 2).tolist()
            out["audio_needs_db"] = round(self.audio_needs_db(tuning), 1)
        out["goals"] = [{"label": label, "ratio": np.round(np.clip(self._goal_ratio(i, tuning), -9, 9), 2).tolist()}
                        for i, label in enumerate(self.labels)]
        if self.flow is not None:
            typical = float(np.median(self.flow)) or 1e-6
            out["flow"] = np.round(self.flow / typical, 2).tolist()
        audio, goals, lulls = detect.all_events(self.sig, self.goal_areas, tuning)
        out["events"] = [{"kind": e.kind, "t": round(e.t, 2), "end": round(e.end, 2) if e.end else None,
                          "reason": e.reason} for e in sorted(audio + goals + lulls, key=lambda e: e.t)]
        return out


# --------------------------------------------------------------------------
# Re-tuning from calls


@dataclass
class Retune:
    tuning: Tuning
    changes: list[str] = field(default_factory=list)  # what moved, and which call moved it
    notes: list[str] = field(default_factory=list)  # calls it couldn't learn from, and why


def _scale(name: str, readouts: list[tuple[dict, float | None]], noun: str,
           any_highlights: bool, unreviewed: list[float] | None = None) -> tuple[float, list[str], list[str]]:
    """One bar's scale (1.0 = default) from (call, ratio) pairs, with reasons.

    Lower the bar until it catches the weakest highlight that has any of this
    evidence; with no highlight left to catch, raise it past false alarms,
    but never past a confirmed highlight. Raising needs at least one
    confirmed highlight, so a run of "not a highlight" calls alone can't
    silence the recording, and it may drop at most one suggestion the user
    hasn't reviewed (``unreviewed``: their ratios) per false alarm it drops.
    Lowering only adds suggestions to review, so it has no such limit."""
    changes: list[str] = []
    notes: list[str] = []
    highs = [(c, r) for c, r in readouts if c["label"] in HIGHLIGHTS and r is not None]
    falses = [(c, r) for c, r in readouts if c["label"] == "other" and r is not None]
    learnable = [(c, r) for c, r in highs if r * MARGIN >= MIN_SCALE]
    upper = min([r * MARGIN for _, r in learnable], default=MAX_SCALE)
    upper = min(MAX_SCALE, upper)
    missed = [(c, r) for c, r in learnable if r < 1]
    if missed:
        c, r = min(missed, key=lambda cr: cr[1])
        scale = max(MIN_SCALE, upper)
        line = (f"{name} bar lowered to {scale:.0%} of the default so your {LABEL_TEXT[c['label']]} "
                f"at {fmt_time(c['t'])} counts.")
        readmitted = [fc for fc, fr in falses if scale <= fr < 1]
        if readmitted:
            line += (" It also lets in " + ", ".join(fmt_time(fc["t"]) for fc in readmitted[:3])
                     + ", which you called not a highlight.")
        changes.append(line)
        return scale, changes, notes
    caught_false = sorted(r for _, r in falses if r >= 1)
    if caught_false and not highs:
        if not any_highlights:
            notes.append("Confirm at least one highlight too, so it knows how far it can raise a bar "
                         "past your not-a-highlight calls.")
        return 1.0, changes, notes
    unreviewed = unreviewed or []
    blocked = 0
    for scale in sorted((r * RAISE_STEP for r in caught_false if r * RAISE_STEP <= upper), reverse=True):
        dropped = [c for c, r in falses if 1 <= r < scale]
        collateral = sum(1 for r in unreviewed if r < scale)
        if collateral > len(dropped):
            blocked = collateral  # ends on the smallest raise tried
            continue
        times = ", ".join(fmt_time(c["t"]) for c in dropped[:3])
        line = (f"{name} bar raised to {scale:.0%} of the default, so {noun} like {times} "
                f"(not a highlight) no longer counts. Your {len(highs)} highlight"
                f"{'s' if len(highs) != 1 else ''} still clear it.")
        if collateral:
            line += f" It also drops {collateral} suggestion{'s' if collateral > 1 else ''} you hadn't reviewed."
        changes.append(line)
        return scale, changes, notes
    if blocked:
        notes.append(f"Raising the {noun} bar past your not-a-highlight calls would also drop {blocked} "
                     "suggestions you haven't reviewed, so it stays. Call more of them to move it.")
    elif caught_false:
        notes.append(f"Your not-a-highlight calls had as much {noun} as a highlight you confirmed, "
                     f"so the {noun} bar stays.")
    return 1.0, changes, notes


def retune(ev: Evidence, calls: list[dict], candidates: list[detect.Moment] | None = None) -> Retune:
    """This recording's thresholds, learned from ``calls`` (each ``{"t", "label"}``).

    ``candidates`` are the moments found at default thresholds; the ones no
    call covers limit how far a bar may be raised. Always computed from the
    defaults, so the same calls give the same answer and changes never compound."""
    readouts = [(c, ev.readout(float(c["t"]))) for c in calls if c.get("label") in LABELS]
    unreviewed = [(m, ev.readout(m.anchor)) for m in candidates or []
                  if not any(m.start - NEAR_CALL <= float(c["t"]) <= m.end + NEAR_CALL for c in calls)]
    out = Retune(DEFAULT_TUNING)
    # Each highlight is carried by the signal that saw more of it, relative to
    # its bar, and only constrains that bar. False alarms count against both.
    carried: dict[int, str] = {}
    for i, (c, r) in enumerate(readouts):
        if c["label"] not in HIGHLIGHTS:
            continue
        a, g = r.audio_ratio or 0.0, r.goal_ratio or 0.0
        carried[i] = "goal" if ev.goals and (g >= a or ev.audio is None) else "audio"
        best = max(a, g)
        if best < 1 and best * MARGIN < MIN_SCALE:
            what = "goal-mouth movement" if carried[i] == "goal" else "crowd noise"
            out.notes.append(f"Your {LABEL_TEXT[c['label']]} at {fmt_time(float(c['t']))} had too little {what} "
                             f"to learn from ({best:.0%} of the default bar). Keep it as a clip; the bars stay.")

    def pairs(signal: str, ratio) -> list[tuple[dict, float | None]]:
        return [(c, ratio(r) if c["label"] == "other" or carried.get(i) == signal else None)
                for i, (c, r) in enumerate(readouts)]

    goal_scale = audio_scale = 1.0
    if ev.goals:
        # Raising one bar only drops a suggestion the other signal doesn't also carry.
        goal_scale, ch, notes = _scale("Goal-mouth movement", pairs("goal", lambda r: r.goal_ratio),
                                       "goal-mouth movement", bool(carried),
                                       [r.goal_ratio or 0.0 for m, r in unreviewed if "audio" not in m.kinds])
        out.changes += ch
        out.notes += notes
    if ev.audio is not None:
        audio_scale, ch, notes = _scale("Crowd-noise", pairs("audio", lambda r: r.audio_ratio), "crowd noise",
                                        bool(carried),
                                        [r.audio_ratio or 0.0 for m, r in unreviewed if "goal" not in m.kinds])
        out.changes += ch
        out.notes += notes
    out.tuning = replace(DEFAULT_TUNING,
                         audio_min_db=DEFAULT_TUNING.audio_min_db * audio_scale,
                         audio_z=DEFAULT_TUNING.audio_z * audio_scale,
                         goal_z=DEFAULT_TUNING.goal_z * goal_scale,
                         goal_min_share=DEFAULT_TUNING.goal_min_share * goal_scale)
    if not readouts:
        out.notes.append("Make a few calls first: mark goals it missed, and say which suggestions were or weren't "
                         "highlights.")
    elif not out.changes and not out.notes:
        out.notes.append("Your calls already agree with the default bars.")
    return out


# --------------------------------------------------------------------------
# Resemblance


def resemble(ev: Evidence, moments: list[detect.Moment], calls: list[dict]) -> int:
    """Move candidates that look like a call up (highlight) or down (not a
    highlight), naming the call. Returns how many moved.

    "Look like" compares four numbers from the signals (loudest rise, seconds
    of noise, goal-mouth movement, the drop in play afterwards), each scaled
    by how much it varies across this recording's candidates."""
    calls = [c for c in calls if c.get("label") in LABELS]
    if not calls or not moments:
        return 0
    cand = np.array([ev.readout(m.anchor).features() for m in moments])
    examples = np.array([ev.readout(float(c["t"])).features() for c in calls])
    pool = np.vstack([cand, examples])
    spread = 1.4826 * np.median(np.abs(pool - np.median(pool, axis=0)), axis=0)
    spread = np.maximum(spread, [1.0, 0.5, 0.25, 0.1])  # dB, seconds, share of the goal bar, ratio
    moved = 0
    for m, f in zip(moments, cand):
        if any(m.start - NEAR_CALL <= float(c["t"]) <= m.end + NEAR_CALL for c in calls):
            continue
        d = np.linalg.norm((examples - f) / spread, axis=1)
        near = [i for i in np.argsort(d)[:RESEMBLE_NEIGHBOURS] if d[i] <= RESEMBLE_RADIUS]
        if not near:
            continue
        vote = sum((1 if calls[i]["label"] in HIGHLIGHTS else -1) / (d[i] + 0.25) for i in near)
        if vote == 0:
            continue
        side = [i for i in near if (calls[i]["label"] in HIGHLIGHTS) == (vote > 0)]
        c = calls[side[0]]
        if vote > 0:
            m.strength += RESEMBLE_BONUS
            m.reasons.append(f"like your {LABEL_TEXT[c['label']]} at {fmt_time(float(c['t']))}")
        else:
            m.strength -= RESEMBLE_BONUS
            m.reasons.append(f"like {fmt_time(float(c['t']))}, which you called not a highlight")
        if "calls" not in m.kinds:
            m.kinds.append("calls")
        moved += 1
    moments.sort(key=lambda m: -m.strength)
    return moved


# --------------------------------------------------------------------------
# Export


def labels_file(src: dict, clips: list[dict]) -> dict:
    """The recording's calls in the format scripts/eval_highlights.py reads."""
    by_id = {c["id"]: c for c in clips}
    goals, moments = [], []
    for call in sorted(src.get("calls", []), key=lambda c: c["t"]):
        clip = by_id.get(call.get("clip_id"))
        if call["label"] == "goal":
            g = {"t": call["t"]}
            if clip:
                g.update(buildup_start=clip["start"], reaction_end=clip["end"])
            goals.append(g)
        else:
            moments.append({"t": call["t"], "label": call["label"]})
    return {
        "video": src["path"],
        "note": f"Calls made in Watch the Goal on {src['name']}.",
        "goal_moments": goals,
        "moments": moments,
        "goal_areas": src.get("goal_areas", []),
        "use_goal_areas": bool(src.get("goal_areas")),
        "fully_labeled": False,
        "labelling": "Calls made while reviewing in Watch the Goal. Only moments you called are labelled.",
    }
