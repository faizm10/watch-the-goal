"""Optional AI review of suggested moments with a local vision model.

Off by default. When enabled, the strongest untouched suggestions of a
recording are sent, as timestamped frames, to a vision model served by Ollama
on this machine. The model answers with a label, a clip it would cut inside
the window it was shown, and a reason. That answer is validated before it is
used, and it only ever *proposes*: suggestions are reordered and get a
suggested trim the user can apply. Clips the user has kept, skipped or edited
are never touched.

Footage stays local: only loopback endpoints are accepted.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import math
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

from .detect import fit_window

LABELS = ("likely_goal", "save", "near_miss", "other", "uncertain")
LABEL_TEXT = {"likely_goal": "likely goal", "save": "save", "near_miss": "near miss",
              "other": "not a highlight", "uncertain": "uncertain"}
# Lower sorts first when suggestions are reordered after review.
PRIORITY = {"likely_goal": 0, "near_miss": 1, "save": 1, "uncertain": 2, "other": 3}
UNREVIEWED = 2

DEFAULTS = {
    "enabled": False,
    "endpoint": "http://127.0.0.1:11434",
    "model": "qwen3.5:9b",
    "max_candidates": 6,
    "frames": 12,
    "timeout": 180,
}
CONTEXT_BEFORE = 20.0  # seconds of buildup shown before the detected moment
CONTEXT_AFTER = 8.0
MAX_WINDOW = 40.0
MIN_CLIP = 3.0
MAX_REASON = 240
FRAME_SIDE = 768

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "start": {"type": "number"},
        "end": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["label", "start", "end", "reason"],
}

PROMPT = """You are helping pick highlights from amateur soccer footage.
The {n} images are frames in time order from one camera, between {w0:.1f} s and {w1:.1f} s into the recording. Each frame has its time in seconds written in the top-left corner: {stamps}.

1. Say what happens in this window:
   - likely_goal: the ball goes in the net, or players celebrate, walk back, or restart from the centre right after a shot
   - save: a goalkeeper stops a shot
   - near_miss: a shot or chance that goes wide, over, or off the post
   - other: ordinary play, a stoppage, people near the camera, or nothing to do with the match
   - uncertain: the frames don't show enough to tell
2. Choose the clip that tells it best. Start where the move that leads to the finish begins, so the pass or dribble before the shot is included. End a few seconds after the finish so the immediate reaction is in. Both times must be between {w0:.1f} and {w1:.1f}.
3. Give one short sentence describing what you can actually see, citing frame times.

Answer with JSON only, like {{"label": "near_miss", "start": {ex0:.1f}, "end": {ex1:.1f}, "reason": "..."}}."""


class ReviewError(Exception):
    """The reviewer couldn't be used; baseline suggestions stay as they were."""


class Unavailable(ReviewError):
    pass


class Timeout(ReviewError):
    pass


class InvalidResponse(ReviewError):
    pass


@dataclass
class Verdict:
    label: str
    start: float
    end: float
    reason: str

    def to_dict(self) -> dict:
        return {"label": self.label, "start": self.start, "end": self.end, "reason": self.reason}


# --------------------------------------------------------------------------
# Config and validation


def config(settings: dict | None) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (settings or {}).items() if k in DEFAULTS})
    cfg["max_candidates"] = int(min(max(int(cfg["max_candidates"]), 1), 20))
    cfg["frames"] = int(min(max(int(cfg["frames"]), 4), 24))
    cfg["timeout"] = float(min(max(float(cfg["timeout"]), 5.0), 900.0))
    return cfg


def require_local(endpoint: str) -> str:
    """Only talk to a model on this machine."""
    url = urlparse(endpoint)
    if url.scheme not in ("http", "https") or not url.hostname:
        raise Unavailable(f"AI review endpoint must be an http URL, got {endpoint!r}.")
    host = url.hostname
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except ValueError:
            raise Unavailable(
                f"AI review only talks to a model on this computer (localhost), not {host}. "
                "Footage never leaves your machine.") from None
    return endpoint.rstrip("/")


def window_for(start: float, end: float, anchor: float | None, duration: float) -> tuple[float, float]:
    """What the reviewer is shown: the suggestion plus room for buildup and reaction."""
    a = anchor if anchor is not None else (start + end) / 2
    w0 = max(0.0, min(start, a - CONTEXT_BEFORE))
    w1 = min(duration, max(end, a + CONTEXT_AFTER))
    if w1 - w0 > MAX_WINDOW:
        # A long suggestion: centre the window on the moment instead.
        w0 = max(0.0, a - (MAX_WINDOW - CONTEXT_AFTER))
        w1 = min(duration, w0 + MAX_WINDOW)
    return round(w0, 2), round(w1, 2)


def frame_times(w0: float, w1: float, n: int) -> list[float]:
    if n <= 1 or w1 <= w0:
        return [round(w0, 2)]
    return [round(w0 + (w1 - w0) * i / (n - 1), 2) for i in range(n)]


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidResponse(f"expected a number, got {value!r}")
    if not math.isfinite(value):
        raise InvalidResponse("timestamps must be finite")
    return float(value)


def validate(obj, w0: float, w1: float) -> Verdict:
    """Check the model's answer. Anything off means the answer is not used."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            raise InvalidResponse("the answer wasn't JSON") from None
    if not isinstance(obj, dict):
        raise InvalidResponse("the answer wasn't a JSON object")
    missing = [k for k in ("label", "start", "end", "reason") if k not in obj]
    if missing:
        raise InvalidResponse(f"the answer is missing {', '.join(missing)}")
    label = str(obj["label"]).strip().lower().replace(" ", "_").replace("-", "_")
    if label not in LABELS:
        raise InvalidResponse(f"unknown label {obj['label']!r}")
    start, end = _number(obj["start"]), _number(obj["end"])
    slack = 0.5  # rounding by the model
    if not (w0 - slack <= start < end <= w1 + slack):
        raise InvalidResponse(f"clip {start:.1f}-{end:.1f}s is outside the window {w0:.1f}-{w1:.1f}s")
    start, end = max(w0, start), min(w1, end)
    if end - start < MIN_CLIP:
        raise InvalidResponse(f"clip is shorter than {MIN_CLIP:.0f}s")
    reason = obj["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise InvalidResponse("the answer has no reason")
    reason = " ".join(reason.split())
    if len(reason) > MAX_REASON:
        reason = reason[: MAX_REASON - 1].rstrip() + "…"
    return Verdict(label, round(start, 2), round(end, 2), reason)


# --------------------------------------------------------------------------
# Frames


def render_frames(path: str, times: list[float], max_side: int = FRAME_SIDE) -> list[bytes]:
    """JPEGs of each time with the timestamp burned in."""
    import cv2

    from .vision.frames import sample_frames
    from .vision.overlay import label

    out = []
    for t, frame in sample_frames(path, times, max_side=max_side):
        label(frame, f"{t:.1f}s", (10, 30), 0.9)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out.append(buf.tobytes())
    return out


# --------------------------------------------------------------------------
# Ollama client


class OllamaClient:
    def __init__(self, endpoint: str, model: str, timeout: float):
        self.endpoint = require_local(endpoint)
        self.model = model
        self.timeout = timeout

    def _post(self, path: str, body: dict, timeout: float) -> dict:
        req = urllib.request.Request(self.endpoint + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except (TimeoutError, socket.timeout):
            raise Timeout(f"the model took longer than {timeout:.0f}s") from None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:200]
            raise Unavailable(f"Ollama returned {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise Timeout(f"the model took longer than {timeout:.0f}s") from None
            raise Unavailable(f"Ollama isn't reachable at {self.endpoint} ({exc.reason}). "
                              "Start it with `ollama serve`.") from None
        except json.JSONDecodeError:
            raise InvalidResponse("Ollama sent something that isn't JSON") from None

    def check(self) -> dict:
        """Is the model there, and can it see images?"""
        info = self._post("/api/show", {"model": self.model}, min(self.timeout, 15))
        caps = info.get("capabilities") or []
        if caps and "vision" not in caps:
            raise Unavailable(f"{self.model} can't read images. Choose a vision model, e.g. qwen3.5:9b.")
        return {"model": self.model, "capabilities": caps,
                "license": (info.get("license") or "").strip().splitlines()[0:1]}

    def review(self, prompt: str, images: list[bytes]) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": SCHEMA,
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_ctx": 16384},
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(i).decode() for i in images]}],
        }
        data = self._post("/api/chat", body, self.timeout)
        content = (data.get("message") or {}).get("content")
        if not content:
            raise InvalidResponse("the model returned an empty answer")
        return content


def review_window(client, path: str, w0: float, w1: float, n_frames: int, instruction: str = "") -> Verdict:
    times = frame_times(w0, w1, n_frames)
    images = render_frames(path, times)
    if not images:
        raise InvalidResponse("couldn't read frames for this window")
    mid = (w0 + w1) / 2
    prompt = PROMPT.format(n=len(images), w0=w0, w1=w1, stamps=", ".join(f"{t:.1f}" for t in times),
                           ex0=max(w0, mid - 8), ex1=min(w1, mid + 4))
    return validate(client.review(prompt + instruction, images), w0, w1)


# --------------------------------------------------------------------------
# Applying results to a project


def untouched(clip: dict) -> bool:
    return clip["origin"] == "detected" and clip["status"] == "candidate" and not clip.get("edited")


def pick(clips: list[dict], limit: int) -> list[dict]:
    """The strongest untouched suggestions, by detector rank."""
    pool = [c for c in clips if untouched(c)]
    pool.sort(key=lambda c: (c.get("rank") is None, c.get("rank") or 0, c["start"]))
    return pool[:limit]


def rerank(clips: list[dict]) -> None:
    """Order detected suggestions by review label, then by detector rank."""
    detected = [c for c in clips if c["origin"] == "detected"]
    for c in detected:
        c.setdefault("detector_rank", c.get("rank"))

    def key(c):
        r = c.get("review") or {}
        p = PRIORITY.get(r.get("label"), UNREVIEWED) if r.get("status") == "done" else UNREVIEWED
        dr = c.get("detector_rank")
        return (p, dr is None, dr or 0, c["start"])

    for i, c in enumerate(sorted(detected, key=key)):
        c["rank"] = i + 1


def review_source(project, source_id: str, settings: dict | None,
                  progress: Callable[[float, str], None] | None = None, client=None) -> dict:
    """Review a recording's strongest untouched suggestions. Never raises for
    reviewer problems: they are recorded and the baseline stays."""
    cfg = config(settings)
    report = progress or (lambda f, m: None)
    src = project.source(source_id)
    with project.lock:
        clips = [project.clips[c] for c in project.order if project.clips[c]["source_id"] == source_id]
        chosen = [dict(c) for c in pick(clips, cfg["max_candidates"])]
    summary = {"status": "done", "model": cfg["model"], "reviewed": 0, "failed": 0, "labels": {},
               "started": time.time()}
    try:
        client = client or OllamaClient(cfg["endpoint"], cfg["model"], cfg["timeout"])
        report(0.02, f"Checking {cfg['model']}")
        client.check()
    except ReviewError as exc:
        summary.update(status="unavailable", message=str(exc))
        project.update_source(source_id, review=summary)
        return summary
    if not chosen:
        summary.update(status="done", message="No untouched suggestions to review.")
        project.update_source(source_id, review=summary)
        return summary

    duration = src["info"]["duration"]
    for i, snap in enumerate(chosen):
        report(0.05 + 0.9 * i / len(chosen), f"AI review {i + 1}/{len(chosen)} ({cfg['model']})")
        w0, w1 = window_for(snap["start"], snap["end"], snap.get("anchor"), duration)
        t0 = time.time()
        record = {"window": [w0, w1], "model": cfg["model"]}
        try:
            verdict = review_window(client, src["path"], w0, w1, cfg["frames"])
            record.update(status="done", **verdict.to_dict())
        except Unavailable as exc:
            # The server went away: stop, keep what we have.
            summary.update(status="partial", message=str(exc))
            break
        except ReviewError as exc:
            record.update(status="timeout" if isinstance(exc, Timeout) else "invalid", message=str(exc))
        except Exception as exc:  # frames unreadable, etc.: this suggestion just isn't reviewed
            record.update(status="invalid", message=str(exc) or exc.__class__.__name__)
        record["seconds"] = round(time.time() - t0, 1)
        _apply(project, snap["id"], record)
        if record["status"] == "done":
            summary["reviewed"] += 1
            summary["labels"][record["label"]] = summary["labels"].get(record["label"], 0) + 1
        else:
            summary["failed"] += 1

    with project.lock:
        rerank([project.clips[c] for c in project.order if project.clips[c]["source_id"] == source_id])
        summary["seconds"] = round(time.time() - summary.pop("started"), 1)
        project.update_source(source_id, review=summary)
    report(1.0, "AI review done")
    return summary


def widen(clip: dict, verdict: dict) -> tuple[float, float]:
    """The AI's start and end, but only where they make the clip longer.

    In benchmarks the reviewer's trims cut into the buildup or the reaction as
    often as they extended them, and keeping both is the point of the app. So
    the detector's clip is the floor: the AI can add buildup or aftermath,
    never remove it, though the result still fits in a short reel clip."""
    anchor = clip["anchor"] if clip.get("anchor") is not None else (clip["start"] + clip["end"]) / 2
    start, end = fit_window(min(clip["start"], verdict["start"]), max(clip["end"], verdict["end"]), anchor, float("inf"))
    return min(start, clip["start"]), max(end, clip["end"])


def _apply(project, clip_id: str, record: dict) -> None:
    """Attach a review to a clip, if the user hasn't acted on it meanwhile."""
    with project.lock:
        clip = project.clips.get(clip_id)
        if clip is None:
            return
        clip["review"] = record
        if record.get("status") == "done" and untouched(clip):
            start, end = widen(clip, record)
            changed = abs(start - clip["start"]) > 0.5 or abs(end - clip["end"]) > 0.5
            clip["suggested_trim"] = {
                "start": start, "end": end, "anchor": clip.get("anchor"),
                "reasons": [f"AI review: {record['reason']}"], "kinds": ["ai"], "source": "ai",
            } if changed else None
        project.save()

def propose_edit(project, clip_id: str, request: str) -> dict:
    with project.lock:
        cfg = config(project.settings.get("ai_review"))
        clip = dict(project.clip(clip_id))
        src = project.source(clip["source_id"])
    if not cfg["enabled"]:
        raise Unavailable("Enable local AI review in Analysis first.")
    start, end = clip["start"], clip["end"]
    anchor = clip.get("anchor")
    if anchor is None:
        anchor = (clip.get("suggested_trim") or {}).get("anchor")
    if request == "buildup":
        w0, w1 = max(0, start - 15), min(end, start + 20)
        instruction = "Include more of the buildup. Find an earlier start."
    elif request == "sooner":
        w0, w1 = max(start, end - MAX_WINDOW), end
        instruction = "Cut sooner after the goal. Find an earlier end after the finish."
    else:
        raise InvalidResponse("Unknown edit request.")
    if w1 - w0 < MIN_CLIP or (request == "buildup" and start <= 0):
        raise InvalidResponse("No room for this edit.")
    client = OllamaClient(cfg["endpoint"], cfg["model"], cfg["timeout"])
    client.check()
    context = f" Current source range: {start:.2f}–{end:.2f}s. "
    if anchor is not None:
        context += f"Suggested finish: {anchor:.2f}s. "
    verdict = review_window(client, src["path"], w0, w1, cfg["frames"], context + instruction)
    new_start = verdict.start if request == "buildup" else start
    new_end = verdict.end if request == "sooner" else end
    if not (0 <= new_start < new_end <= src["info"]["duration"] and new_end - new_start >= 1
            and (new_start < start or new_end < end)
            and (request != "sooner" or anchor is None or new_end > anchor)):
        raise InvalidResponse("The reviewer found no usable change in that direction.")
    return {"expected": [start, end], "start": new_start, "end": new_end, "reason": verdict.reason}
