"""Project state: imported sources, clips and reel order, saved as JSON.

A project lives in a workspace folder:

    <workspace>/project.json   state
    <workspace>/media/         copies of files dropped into the browser
    <workspace>/cache/         preview proxies
    <workspace>/exports/       rendered reels

Files imported by path are referenced in place and only ever read.
"""

from __future__ import annotations

from collections import deque
import json
import math
import threading
import time
import uuid
from pathlib import Path

from .detect import POST_ROLL, PRE_ROLL, Moment, clamp_window, merge_overlapping
from .media import MediaInfo
from .review import DEFAULTS as REVIEW_DEFAULTS

LONG_RECORDING_SECONDS = 120.0
MIN_CLIP_SECONDS = 1.0
TARGET_MIN, TARGET_MAX = 30.0, 60.0
STATUSES = ("candidate", "kept", "skipped")
CALL_LABELS = ("goal", "save", "near_miss", "chance", "tackle", "other")  # see coach.py
MIN_PLAY_SECONDS = 60.0  # a trimmed match shorter than this is a mistake, not a match
AUTO_OVERSHOOT = 1.25  # an auto-filled reel may run this far past its target rather than stop short
SAME_CALL_SECONDS = 3.0  # a second call this close to one with no clip replaces it


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class ProjectError(Exception):
    pass


class Project:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).expanduser().resolve()
        self.lock = threading.RLock()
        for sub in ("media", "cache", "exports"):
            (self.workspace / sub).mkdir(parents=True, exist_ok=True)
        self.file = self.workspace / "project.json"
        self.sources: list[dict] = []
        self.clips: dict[str, dict] = {}
        self.order: list[str] = []
        self.exports: list[dict] = []
        self.settings = {"template": "plain", "title": "", "end_card": "", "layout": "auto", "pre_roll": PRE_ROLL, "post_roll": POST_ROLL,
                         "detector": "none", "ai_review": dict(REVIEW_DEFAULTS)}
        self.history = deque(maxlen=60)
        self.load()

    # ------------------------------------------------------------------ io

    def load(self) -> None:
        if not self.file.exists():
            return
        data = json.loads(self.file.read_text())
        self.sources = data.get("sources", [])
        self.clips = {c["id"]: c for c in data.get("clips", [])}
        self.order = [cid for cid in data.get("order", []) if cid in self.clips]
        self.exports = data.get("exports", [])
        saved = data.get("settings", {})
        self.settings.update({k: v for k, v in saved.items() if k != "ai_review"})
        # Projects from before AI review existed get its defaults (off).
        self.settings["ai_review"] = {**REVIEW_DEFAULTS, **(saved.get("ai_review") or {})}
        # Work that was interrupted by a restart is no longer running.
        for s in self.sources:
            s.setdefault("calls", [])  # projects from before calls existed
            s.setdefault("tuning", None)
            s.setdefault("play", None)  # projects from before match trimming existed
            if s.get("analysis", {}).get("status") == "running":
                s["analysis"] = {"status": "interrupted", "message": "Analysis was interrupted. Run it again."}

    def save(self) -> None:
        with self.lock:
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.snapshot(), indent=1))
            tmp.replace(self.file)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "sources": self.sources,
                "clips": [self.clips[cid] for cid in self.order],
                "order": list(self.order),
                "exports": self.exports,
                "settings": self.settings,
            }

    def reset(self) -> None:
        with self.lock:
            self.sources, self.clips, self.order, self.exports = [], {}, [], []
            self.history.clear()
            self.settings.update(template="plain", title="", end_card="")
            self.save()

    # ------------------------------------------------------------- sources

    def source(self, source_id: str) -> dict:
        for s in self.sources:
            if s["id"] == source_id:
                return s
        raise ProjectError(f"Unknown source {source_id}")

    def add_source(self, info: MediaInfo, name: str, copied: bool = False) -> dict:
        with self.lock:
            for s in self.sources:
                if s["path"] == info.path:
                    return s  # importing the same file twice is a no-op
            kind = "long" if info.duration > LONG_RECORDING_SECONDS else "short"
            src = {
                "id": new_id("src"),
                "name": name,
                "path": info.path,
                "copied": copied,
                "kind": kind,
                "info": info.to_dict(),
                "goal_areas": [],
                "calls": [],  # the user's calls: what really happened at a moment
                "tuning": None,  # thresholds learned from those calls (None = defaults)
                "play": None,  # {"start", "end", "break": [a, b] | None}: the match itself (None = all of it)
                "analysis": {"status": "pending"},
                "preview": "original" if info.browser_safe else "pending",
                "imported_at": time.time(),
            }
            self.sources.append(src)
            if kind == "short":
                # Short phone clips are the highlight: include them right away.
                self._insert_clip(self._make_clip(src, 0.0, info.duration, "kept", "short-clip",
                                                  ["short clip, included whole"]))
            self.save()
            return src

    def remove_source(self, source_id: str) -> None:
        with self.lock:
            self.source(source_id)
            self.sources = [s for s in self.sources if s["id"] != source_id]
            for cid in [c for c in self.order if self.clips[c]["source_id"] == source_id]:
                self.order.remove(cid)
                del self.clips[cid]
            self.save()

    def set_play(self, source_id: str, play: dict | None) -> dict:
        """Where the match is in a recording: kick-off, full time and an optional
        halftime break, in seconds. None (or the whole recording) clears it."""
        with self.lock:
            src = self.source(source_id)
            duration = src["info"]["duration"]
            if play:
                try:
                    start = float(play.get("start") or 0.0)
                    end = float(play["end"]) if play.get("end") is not None else duration
                    brk = [float(x) for x in play["break"]] if play.get("break") else None
                except (TypeError, ValueError, KeyError):
                    raise ProjectError("Match times must be numbers of seconds.") from None
                start, end = max(0.0, start), min(duration, end)
                if end - start < MIN_PLAY_SECONDS:
                    raise ProjectError("Full time must be at least a minute after kick-off.")
                if brk and not (start < brk[0] < brk[1] < end):
                    raise ProjectError("Halftime must start after kick-off and end before full time.")
                play = {"start": round(start, 2), "end": round(end, 2),
                        "break": [round(brk[0], 2), round(brk[1], 2)] if brk else None}
                if play["start"] == 0 and play["end"] >= round(duration, 2) and not brk:
                    play = None
            src["play"] = play or None
            self.save()
            return src

    def set_goal_areas(self, source_id: str, areas: list[dict]) -> dict:
        clean = []
        for a in areas[:4]:
            x, y, w, h = (float(a[k]) for k in ("x", "y", "w", "h"))
            x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
            w, h = min(max(w, 0.0), 1.0 - x), min(max(h, 0.0), 1.0 - y)
            if w > 0.005 and h > 0.005:
                clean.append({"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)})
        with self.lock:
            src = self.source(source_id)
            src["goal_areas"] = clean
            self.save()
            return src

    def update_source(self, source_id: str, **fields) -> dict:
        with self.lock:
            src = self.source(source_id)
            src.update(fields)
            self.save()
            return src

    # --------------------------------------------------------------- clips

    def _make_clip(self, src: dict, start: float, end: float, status: str, origin: str,
                   reasons: list[str], anchor: float | None = None) -> dict:
        start, end = clamp_window(start, end, src["info"]["duration"], MIN_CLIP_SECONDS)
        return {
            "id": new_id("clip"),
            "source_id": src["id"],
            "start": start,
            "end": end,
            "status": status,
            "origin": origin,
            "reasons": reasons,
            "anchor": anchor,
            "edited": False,
            "suggested_trim": None,
            "rank": None,
            "review": None,
        }

    def _sort_key(self, clip: dict) -> tuple:
        idx = next((i for i, s in enumerate(self.sources) if s["id"] == clip["source_id"]), 1 << 30)
        return (idx, clip["start"])

    def _insert_clip(self, clip: dict) -> None:
        """Insert in source-then-time order, after anything the user arranged before it."""
        self.clips[clip["id"]] = clip
        key = self._sort_key(clip)
        pos = len(self.order)
        for i, cid in enumerate(self.order):
            if self._sort_key(self.clips[cid]) > key:
                pos = i
                break
        self.order.insert(pos, clip["id"])

    def clip(self, clip_id: str) -> dict:
        try:
            return self.clips[clip_id]
        except KeyError:
            raise ProjectError(f"Unknown clip {clip_id}") from None

    def add_manual_clip(self, source_id: str, start: float, end: float, reasons: list[str] | None = None,
                        anchor: float | None = None) -> dict:
        with self.lock:
            src = self.source(source_id)
            clip = self._make_clip(src, start, end, "kept", "manual", reasons or ["added by you"], anchor)
            clip["edited"] = True
            self._insert_clip(clip)
            self.save()
            return clip

    def update_clip(self, clip_id: str, **fields) -> dict:
        with self.lock:
            clip = self.clip(clip_id)
            src = self.source(clip["source_id"])
            original = clip
            clip = dict(original)  # validate the whole edit before publishing any field
            before = (clip["start"], clip["end"])
            if "expected" in fields and list(before) != fields["expected"]:
                raise ProjectError("This clip changed. Request a new proposal.")
            if "crop" in fields:
                crop = fields["crop"]
                if crop is not None and (not isinstance(crop, (int, float)) or not 0 <= crop <= 1):
                    raise ProjectError("Crop position must be between 0 and 1.")
                clip["crop"] = crop
                clip["edited"] = True
            if "status" in fields:
                if fields["status"] not in STATUSES:
                    raise ProjectError(f"Status must be one of {', '.join(STATUSES)}")
                clip["status"] = fields["status"]
            if "start" in fields or "end" in fields:
                start = float(fields.get("start", clip["start"]))
                end = float(fields.get("end", clip["end"]))
                if not math.isfinite(start) or not math.isfinite(end):
                    raise ProjectError("Invalid clip time.")
                if end < start:
                    start, end = end, start
                clip["start"], clip["end"] = clamp_window(start, end, src["info"]["duration"], MIN_CLIP_SECONDS)
                clip["edited"] = True
            if fields.get("apply_trim") and clip.get("suggested_trim"):
                trim = clip["suggested_trim"]
                clip["start"], clip["end"] = clamp_window(trim["start"], trim["end"], src["info"]["duration"], MIN_CLIP_SECONDS)
                clip["edited"] = True
            after = (clip["start"], clip["end"])
            original.update(clip)
            if before != after:
                self.history.append(("trim", clip_id, before, after))
            self.save()
            return original

    def undo(self):
        with self.lock:
            while self.history:
                kind, cid, before, after = self.history.pop()
                if kind == "trim":
                    clip = self.clips.get(cid)
                    if not clip or (clip["start"], clip["end"]) != after:
                        continue
                    clip["start"], clip["end"] = before
                elif kind == "status":
                    # Only clips still as the edit left them; the user may have changed some since.
                    still = [c for c in before if c in self.clips and self.clips[c]["status"] == after[c]]
                    if not still:
                        continue
                    for c in still:
                        self.clips[c]["status"] = before[c]
                else:
                    # Preserve newly added clips and ignore clips since deleted.
                    present = set(before).intersection(self.order)
                    restored = iter(i for i in before if i in present)
                    self.order = [next(restored) if i in present else i for i in self.order]
                self.save()
                return {"kind": kind, "clip_id": cid}
            raise ProjectError("No edit to undo.")

    def delete_clip(self, clip_id: str) -> None:
        with self.lock:
            clip = self.clip(clip_id)
            del self.clips[clip_id]
            self.order.remove(clip_id)
            # The call still says what happened at that moment.
            for call in self.source(clip["source_id"]).get("calls", []):
                if call.get("clip_id") == clip_id:
                    call["clip_id"] = None
            self.save()

    # --------------------------------------------------------------- calls

    def add_call(self, source_id: str, t: float, label: str, clip_id: str | None = None) -> dict:
        """Record what really happened at ``t``. One call per clip, and a call
        near an earlier one without a clip replaces it."""
        if label not in CALL_LABELS:
            raise ProjectError(f"Call must be one of {', '.join(CALL_LABELS)}")
        t = float(t)
        with self.lock:
            src = self.source(source_id)
            if not math.isfinite(t) or not 0 <= t <= src["info"]["duration"]:
                raise ProjectError("That time isn't in this recording.")
            if clip_id is not None and self.clip(clip_id)["source_id"] != source_id:
                raise ProjectError("That clip belongs to another recording.")
            calls = src.setdefault("calls", [])
            same = [c for c in calls if (clip_id and c.get("clip_id") == clip_id)
                    or (not c.get("clip_id") and not clip_id and abs(c["t"] - t) < SAME_CALL_SECONDS)]
            call = same[0] if same else {"id": new_id("call")}
            call.update(t=round(t, 2), label=label, clip_id=clip_id, at=time.time())
            if not same:
                calls.append(call)
            calls.sort(key=lambda c: c["t"])
            self.save()
            return call

    def remove_call(self, source_id: str, call_id: str) -> None:
        with self.lock:
            src = self.source(source_id)
            calls = src.get("calls", [])
            if not any(c["id"] == call_id for c in calls):
                raise ProjectError(f"Unknown call {call_id}")
            src["calls"] = [c for c in calls if c["id"] != call_id]
            self.save()

    def reorder(self, ids: list[str]) -> None:
        """Set the order of the given clips; clips not listed keep their relative place after them."""
        with self.lock:
            ids = list(dict.fromkeys(i for i in ids if i in self.clips))
            rest = [i for i in self.order if i not in ids]
            if self.order != ids + rest:
                self.history.append(("order", None, self.order.copy(), ids + rest))
            self.order = ids + rest
            self.save()

    def apply_detection(self, source_id: str, moments: list[Moment]) -> list[dict]:
        """Replace untouched suggestions for a source with fresh ones."""
        with self.lock:
            src = self.source(source_id)
            if src["kind"] == "short":
                clip = next((self.clips[c] for c in self.order
                             if self.clips[c]["source_id"] == source_id and self.clips[c]["origin"] == "short-clip"), None)
                if clip is not None:
                    clip["suggested_trim"] = moments[0].to_dict() if moments else None
                self.save()
                return [clip] if clip else []

            stale = [cid for cid in self.order
                     if self.clips[cid]["source_id"] == source_id
                     and self.clips[cid]["origin"] == "detected"
                     and self.clips[cid]["status"] == "candidate"
                     and not self.clips[cid]["edited"]]
            for cid in stale:
                self.order.remove(cid)
                del self.clips[cid]
            # Don't re-suggest a moment the user already kept, skipped or edited.
            existing = [self.clips[c] for c in self.order if self.clips[c]["source_id"] == source_id]
            added = []
            for m in merge_overlapping(moments):
                if any(c["start"] < m.end and m.start < c["end"] for c in existing):
                    continue
                clip = self._make_clip(src, m.start, m.end, "candidate", "detected", m.reasons, m.anchor)
                clip["rank"] = m.rank
                self._insert_clip(clip)
                added.append(clip)
            self.save()
            return added

    def auto_reel(self, target: float | None) -> dict:
        """Keep the strongest untouched suggestions until the reel runs about
        ``target`` seconds (None: keep every suggestion).

        Strongest means rank, so AI review and the user's calls already count;
        with several recordings their #1s come first, then their #2s. Clips the
        user kept or skipped stay as they are, and a suggestion the AI reviewer
        called "not a highlight" is left out. One undo puts it all back."""
        with self.lock:
            index = {s["id"]: i for i, s in enumerate(self.sources)}
            total = sum(c["end"] - c["start"] for c in self.reel())
            pool = [self.clips[cid] for cid in self.order
                    if self.clips[cid]["origin"] == "detected" and self.clips[cid]["status"] == "candidate"
                    and not ((self.clips[cid].get("review") or {}).get("status") == "done"
                             and self.clips[cid]["review"].get("label") == "other")]
            pool.sort(key=lambda c: (c.get("rank") is None, c.get("rank") or 0,
                                     index.get(c["source_id"], 0), c["start"]))
            kept: list[str] = []
            for c in pool:
                length = c["end"] - c["start"]
                if target is not None:
                    if total >= target:
                        break
                    if total + length > target * AUTO_OVERSHOOT and (kept or total):
                        continue  # too long to fit; a shorter, weaker one may
                c["status"] = "kept"
                kept.append(c["id"])
                total += length
            if kept:
                self.history.append(("status", None, {c: "candidate" for c in kept}, {c: "kept" for c in kept}))
                self.save()
            left = sum(1 for c in pool if c["status"] == "candidate")
            return {"kept": kept, "seconds": round(total, 1), "left": left}

    def reel(self) -> list[dict]:
        return [self.clips[cid] for cid in self.order if self.clips[cid]["status"] == "kept"]

    def add_export(self, record: dict) -> None:
        with self.lock:
            self.exports.insert(0, record)
            self.save()
