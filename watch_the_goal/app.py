"""Application services shared by the web server and the CLI."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np

from . import coach, detect, pipeline, review, signals, writer
from .export import ExportClip, export_reel
from .jobs import Job, JobRunner
from .media import VIDEO_EXTENSIONS, MediaError, make_preview_proxy, probe
from .project import Project, ProjectError
from .recap import FADE_SECONDS, board_text


def safe_name(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip() or "video"


class App:
    def __init__(self, workspace: Path):
        self.project = Project(workspace)
        self.jobs = JobRunner()
        self._evidence: dict[str, tuple[tuple, coach.Evidence]] = {}
        self._live_lock = threading.Lock()  # one OpenCV net, one frame at a time
        self._live: tuple[str, object] | None = None
        self._keys: dict[str, str] = {}  # provider -> API key pasted this session; never written to disk

    # -------------------------------------------------------------- import

    def import_paths(self, paths: list[str]) -> dict:
        """Import files in the order given; a folder's videos come in recording order."""
        added, errors = [], []
        probed = []
        for raw in paths:
            p = Path(str(raw).strip().strip("'\"")).expanduser()
            group = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS) if p.is_dir() else [p]
            if p.is_dir() and not group:
                errors.append(f"No videos found in {p}")
            infos = []
            for f in group:
                try:
                    infos.append((f, probe(f)))
                except MediaError as exc:
                    errors.append(str(exc))
            if p.is_dir():
                infos.sort(key=lambda fi: (fi[1].recorded_at, fi[0].name))
            probed.extend(infos)
        for f, info in probed:
            src = self.project.add_source(info, f.name, copied=False)
            added.append(src)
            self.start_source_jobs(src)
        return {"added": added, "errors": errors}

    def import_upload(self, name: str, stream, length: int) -> dict:
        """Save an uploaded file into the workspace (the original stays on the user's device)."""
        name = safe_name(name)
        if Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
            raise MediaError(f"{name} doesn't look like a video. Supported: MP4, MOV, M4V, MKV, WebM.")
        dest = self.project.workspace / "media" / f"{int(time.time() * 1000)}-{name}"
        remaining = length
        with open(dest, "wb") as fh:
            while remaining > 0:
                chunk = stream.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
        if remaining:
            dest.unlink(missing_ok=True)
            raise MediaError(f"Upload of {name} was interrupted.")
        try:
            info = probe(dest)
        except MediaError:
            dest.unlink(missing_ok=True)
            raise
        src = self.project.add_source(info, name, copied=True)
        self.start_source_jobs(src)
        return src

    def start_source_jobs(self, src: dict) -> None:
        if src["preview"] == "pending":
            self.jobs.start("preview", lambda job, s=src: self._build_preview(job, s), src["id"])
        if src["analysis"].get("status") in ("pending", "interrupted"):
            self.analyze(src["id"])

    def _build_preview(self, job: Job, src: dict) -> dict:
        job.update(0.05, f"Building a browser preview for {src['name']}")
        out = self.project.workspace / "cache" / f"{src['id']}-preview.mp4"
        try:
            make_preview_proxy(src["path"], out)
        except MediaError:
            self.project.update_source(src["id"], preview="failed")
            raise
        self.project.update_source(src["id"], preview="proxy", preview_path=str(out))
        return {"preview": str(out)}

    # ------------------------------------------------------------ analysis

    def analyze(self, source_id: str) -> Job:
        self.project.update_source(source_id, analysis={"status": "running"})
        return self.jobs.start("analyze", lambda job: self._analyze(job, source_id), source_id)

    def _analyze(self, job: Job, source_id: str) -> dict:
        src = self.project.source(source_id)
        info = src["info"]
        settings = self.project.settings
        t0 = time.time()
        job.update(0.0, f"Listening and watching {src['name']}")
        notes: list[str] = []
        stats: dict = {}
        try:
            pre, post = settings["pre_roll"], settings["post_roll"]
            if src["kind"] == "short":
                sig = signals.extract(
                    src["path"], info["width"], info["height"], info["duration"], info["has_audio"],
                    src["goal_areas"], progress=lambda f: job.update(f * 0.95),
                )
                trim = detect.suggest_trim(sig, src["goal_areas"])
                moments = [trim] if trim else []
            else:
                options = pipeline.Options(
                    detector=settings.get("detector", "none"), pre_roll=pre, post_roll=post,
                    debug_dir=self.project.workspace / "cache" / "debug" / source_id,
                    tuning=detect.Tuning.from_dict(src.get("tuning")) if src.get("tuning") else None,
                    calls=list(src.get("calls", [])),
                    play=src.get("play"),
                )
                # Re-analysing (after a call, a new detector or new tuning) reuses the decoded signals.
                cached = self.signals_for(src)
                result = pipeline.analyze(src["path"], info, src["goal_areas"], options,
                                          progress=lambda f, m: job.update(f * 0.98, f"{m}: {src['name']}"),
                                          sig=cached)
                sig, moments, notes, stats = result.signals, result.moments, result.notes, result.stats
                if cached is None:
                    signals.save(sig, self._signals_file(source_id), self._signal_key(src))
        except Exception as exc:
            self.project.update_source(source_id, analysis={"status": "error", "message": str(exc)})
            raise
        added = self.project.apply_detection(source_id, moments)
        self._save_activity(source_id, sig)
        detector = stats.get("detector", "none") if "players_seconds" in stats else "none"
        summary = {
            "status": "done",
            "found": len(moments),
            "seconds": round(time.time() - t0, 1),
            "used": [k for k, on in (("audio", sig.audio_db is not None),
                                     ("motion", True),
                                     ("camera pan", detect.camera_offset(sig, int(sig.duration / detect.STEP)) is not None),
                                     ("goal areas", bool(src["goal_areas"])),
                                     (f"players ({detector})", detector != "none")) if on],
            "notes": notes,
            "tuned": bool(src.get("tuning")),
            "moved_by_calls": stats.get("moved_by_calls", 0),
            "players_typical": stats.get("players_typical"),
        }
        self.project.update_source(source_id, analysis=summary)
        if src["kind"] == "long" and settings.get("ai_review", {}).get("enabled"):
            self.review(source_id)
        return {"found": len(moments), "added": len(added)}

    # ----------------------------------------------------------- AI review

    def review(self, source_id: str) -> Job:
        self.project.source(source_id)
        self.project.update_source(source_id, review={"status": "running"})

        def run(job: Job) -> dict:
            return review.review_source(self.project, source_id, self.project.settings.get("ai_review"),
                                        progress=lambda f, m: job.update(f, m))

        return self.jobs.start("review", run, source_id)

    def download_model(self, name: str) -> Job:
        from .vision import models

        models.model_path(name)  # unknown names fail here, before a job starts

        def run(job: Job) -> dict:
            path = models.download(name, progress=lambda f: job.update(f, f"Downloading {name}"))
            return {"path": str(path)}

        return self.jobs.start("download", run, name)

    # ------------------------------------------------------ calls and coaching

    def _signal_key(self, src: dict) -> str:
        return json.dumps(src["goal_areas"], sort_keys=True)

    def _signals_file(self, source_id: str) -> Path:
        return self.project.workspace / "cache" / f"{source_id}-signals.npz"

    def signals_for(self, src: dict) -> signals.Signals | None:
        """Signals from this recording's last analysis, if its goal areas haven't changed since."""
        return signals.load(self._signals_file(src["id"]), self._signal_key(src))

    def evidence(self, source_id: str) -> coach.Evidence | None:
        src = self.project.source(source_id)
        key = (self._signal_key(src), src.get("analysis", {}).get("seconds"))
        hit = self._evidence.get(source_id)
        if hit and hit[0] == key:
            return hit[1]
        sig = self.signals_for(src)
        if sig is None:
            return None
        ev = coach.Evidence(sig, src["goal_areas"])
        self._evidence[source_id] = (key, ev)
        return ev

    def timeline(self, source_id: str) -> dict:
        ev = self.evidence(source_id)
        if ev is None:
            return {}
        return ev.timeline(detect.Tuning.from_dict(self.project.source(source_id).get("tuning")))

    def _explainer(self, source_id: str):
        """explain(t, label) for one recording, sharing one candidate list."""
        src = self.project.source(source_id)
        ev = self.evidence(source_id)
        if ev is None:
            return lambda t, label=None: {"t": t, "verdict": None, "rank": None, "lines": [
                "Analyse this recording to see what the detector measured here."]}
        tuning = detect.Tuning.from_dict(src.get("tuning"))
        settings = self.project.settings
        spans = detect.play_spans(src.get("play"), ev.sig.duration)
        candidates = detect.candidate_moments(ev.sig, src["goal_areas"], settings["pre_roll"],
                                              settings["post_roll"], tuning, spans)
        coach.resemble(ev, candidates, src.get("calls", []))  # the same order analysis uses
        cap = detect.default_cap(detect.play_seconds(spans))
        with self.project.lock:
            clips = [dict(c) for c in self.project.clips.values() if c["source_id"] == source_id]

        def explain(t, label=None):
            out = ev.explain(t, tuning, clips, candidates, cap, src["kind"] == "short", label)
            if not any(a <= t <= b for a, b in spans):
                out["verdict"], out["rank"] = "outside", None
                out["lines"][-1] = ("This is outside the match you trimmed to (warm-up, halftime or after "
                                    "full time), so nothing here is suggested.")
            return out

        return explain

    def explain(self, source_id: str, t: float, label: str | None = None) -> dict:
        return self._explainer(source_id)(t, label)

    def make_call(self, source_id: str, label: str, t: float | None = None, clip_id: str | None = None) -> dict:
        """Record the user's call. On a clip, or at ``t`` for something detection
        missed: a highlight there gets a kept clip with its buildup. A suggestion
        called a highlight is kept, and one called not a highlight is skipped."""
        p = self.project
        with p.lock:
            src = p.source(source_id)
            if clip_id is None and t is not None:
                covering = [c for c in p.clips.values() if c["source_id"] == source_id and c["start"] <= t <= c["end"]]
                if covering:
                    clip_id = min(covering, key=lambda c: abs((c["anchor"] if c["anchor"] is not None
                                                               else c["start"]) - t))["id"]
                elif label in coach.HIGHLIGHTS:
                    pre, post = coach.TAG_ROLL.get(label, (p.settings["pre_roll"], p.settings["post_roll"]))
                    clip = p.add_manual_clip(source_id, t - pre, t + post,
                                             [f"your call: {coach.LABEL_TEXT[label]}"], anchor=round(t, 2))
                    clip_id = clip["id"]
            if clip_id is not None:
                clip = p.clip(clip_id)
                if t is None:
                    t = clip["anchor"] if clip["anchor"] is not None else (clip["start"] + clip["end"]) / 2
            if t is None:
                raise ProjectError("A call needs a time or a clip.")
            call = p.add_call(source_id, t, label, clip_id)
            if clip_id is not None:
                clip = p.clip(clip_id)
                if clip["origin"] == "manual" and clip["reasons"][0].startswith("your call: "):
                    clip["reasons"][0] = f"your call: {coach.LABEL_TEXT[label]}"  # a changed mind
            if clip_id is not None and p.clip(clip_id)["status"] == "candidate":
                p.update_clip(clip_id, status="kept" if label in coach.HIGHLIGHTS else "skipped")
            src_calls = len(src["calls"])
        return {"call": call, "calls": src_calls, "explain": self.explain(source_id, call["t"], label)}

    def retune(self, source_id: str) -> Job:
        """Learn this recording's thresholds from its calls, then re-analyse."""
        src = self.project.source(source_id)
        if src["kind"] != "long":
            raise ProjectError("Re-tuning is for long recordings.")
        if self.evidence(source_id) is None:
            raise ProjectError("Analyse this recording first.")
        self.project.update_source(source_id, analysis={**src["analysis"], "status": "running"})

        def run(job: Job) -> dict:
            job.update(0.0, "Learning from your calls")
            before = self._suggestion_count(source_id)
            used = [dict(c) for c in self.project.source(source_id)["calls"]]
            try:
                ev = self.evidence(source_id)
                settings = self.project.settings
                found = detect.candidate_moments(ev.sig, src["goal_areas"], settings["pre_roll"], settings["post_roll"],
                                                 spans=detect.play_spans(src.get("play"), ev.sig.duration))
                result = coach.retune(ev, used, found)
            except Exception as exc:
                self.project.update_source(source_id, analysis={"status": "error", "message": str(exc)})
                raise
            tuned = result.tuning != detect.DEFAULT_TUNING
            self.project.update_source(source_id, tuning=result.tuning.to_dict() if tuned else None)
            out = self._analyze(job, source_id)
            explain = self._explainer(source_id)
            calls = [{"t": c["t"], "label": c["label"], "verdict": explain(c["t"], c["label"])["verdict"]}
                     for c in self.project.source(source_id)["calls"]]
            coaching = {"changes": result.changes, "notes": result.notes, "before": before,
                        "after": self._suggestion_count(source_id), "calls": calls, "at": time.time(),
                        "basis": calls_signature(used),
                        "moved": self.project.source(source_id)["analysis"].get("moved_by_calls", 0)}
            self.project.update_source(source_id, coaching=coaching)
            return {**out, **coaching}

        return self.jobs.start("analyze", run, source_id)

    def reset_tuning(self, source_id: str) -> Job:
        self.project.update_source(source_id, tuning=None, coaching=None)
        return self.analyze(source_id)

    def _suggestion_count(self, source_id: str) -> int:
        with self.project.lock:
            return sum(1 for c in self.project.clips.values()
                       if c["source_id"] == source_id and c["origin"] == "detected")

    def labels_file(self, source_id: str) -> dict:
        with self.project.lock:
            return coach.labels_file(self.project.source(source_id), list(self.project.clips.values()))

    # -------------------------------------------------------- live detection

    def live_detector_name(self) -> str | None:
        """The analysis detector if one is chosen and downloaded, else any downloaded one."""
        from .vision import models

        chosen = self.project.settings.get("detector", "none")
        for name in ([chosen] if chosen in models.MODELS else []) + list(models.MODELS):
            try:
                models.require(name)
                return name
            except models.ModelUnavailable:
                continue
        return None

    def live(self, source_id: str, jpeg: bytes, t: float) -> dict:
        """People in one frame the browser is showing, and what they mean for the marked goals."""
        import cv2

        from .vision import detectors, players

        src = self.project.source(source_id)
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Couldn't read that frame.")
        name = self.live_detector_name()
        if name is None:
            return {"t": t, "detector": None,
                    "message": "Download a player detector in Analysis to see players live."}
        with self._live_lock:
            if self._live is None or self._live[0] != name:
                self._live = (name, detectors.load(name))
            t0 = time.time()
            boxes = self._live[1].detect(frame)
            ms = (time.time() - t0) * 1000
        h, w = frame.shape[:2]
        areas = src["goal_areas"]
        return {
            "t": t, "detector": name, "ms": round(ms),
            "boxes": [[round(float(x0) / w, 4), round(float(y0) / h, 4), round(float(x1) / w, 4),
                       round(float(y1) / h, 4), round(float(sc), 2)] for x0, y0, x1, y1, sc in boxes],
            "people": len(boxes),
            "near": [{"label": detect.area_label(a, i), "count": players.near_goal(boxes, a, w, h),
                      "blocked": players.lens_blocked(boxes, a, w, h)} for i, a in enumerate(areas)],
            "bunch": players.bunch(boxes),
            "typical": src.get("analysis", {}).get("players_typical"),
        }

    def _save_activity(self, source_id: str, sig: signals.Signals) -> None:
        """Store ~1 value per second of audio and motion for the timeline sparkline."""
        n = max(1, int(np.ceil(sig.duration)))

        def per_second(series, hop):
            if series is None or len(series) == 0:
                return None
            idx = np.minimum((np.arange(len(series)) * hop).astype(int), n - 1)
            out = np.full(n, -np.inf)
            np.maximum.at(out, idx, series)
            out[np.isinf(out)] = np.nan
            lo, hi = np.nanpercentile(out, 5), np.nanpercentile(out, 99.5)
            scaled = np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)
            return [None if np.isnan(v) else round(float(v), 3) for v in scaled]

        data = {
            "audio": per_second(sig.audio_db, sig.audio_hop),
            "motion": per_second(sig.motion, sig.motion_hop),
        }
        (self.project.workspace / "cache" / f"{source_id}-activity.json").write_text(json.dumps(data))

    def activity(self, source_id: str) -> dict:
        f = self.project.workspace / "cache" / f"{source_id}-activity.json"
        return json.loads(f.read_text()) if f.exists() else {}

    # ---------------------------------------------------------- reel text

    def writers(self) -> list[dict]:
        """Who can write the reel text, for the recap dialog. Keys themselves never leave the server."""
        return [{"id": "detection", "label": "tags and detection", "ready": True}] + [
            {"id": pid, "label": label, "model": model, "env": env[0],
             "ready": pid in self._keys or writer.has_key(pid)}
            for pid, (label, model, env, _) in writer.PROVIDERS.items()]

    def draft_text(self, provider: str, key: str = "", model: str = "") -> dict:
        """A proposal for the title, captions and end card. Saving it is a separate step."""
        if key and provider in writer.PROVIDERS:
            self._keys[provider] = key.strip()
        with self.project.lock:
            facts = writer.facts(self.project)
        return writer.draft(facts, provider, self._keys.get(provider), model.strip() or None)

    # -------------------------------------------------------------- export

    def export(self, layout: str = "auto") -> Job:
        with self.project.lock:
            recap = self.project.settings.get("template") == "recap"
            title = self.project.settings.get("title", "") if recap else ""
            end = self.project.settings.get("end_card", "") if recap else ""
            reel = self.project.reel()
            sources = {s["id"]: s for s in self.project.sources}
            clips = []
            for i, c in enumerate(reel):
                s = sources[c["source_id"]]
                clips.append(ExportClip(
                    path=s["path"], start=c["start"], end=c["end"],
                    has_audio=s["info"]["has_audio"], width=s["info"]["width"], height=s["info"]["height"],
                    crop=c.get("crop"), board=board_text(s.get("scoreboard", {})) if recap else "",
                    caption=c.get("caption", "") if recap else "", fade=FADE_SECONDS if recap else 0.0,
                    label=f"clip {i + 1} ({s['name']} {fmt_time(c['start'])})",
                ))
        out = self.project.workspace / "exports" / f"reel-{time.time_ns()}.mp4"

        def run(job: Job) -> dict:
            result = export_reel(clips, out, layout, self.project.workspace / "cache",
                                 progress=lambda f, m: job.update(f, m), title=title, end=end,
                                 fade=FADE_SECONDS if recap else 0.0)
            result["name"] = out.name
            result["created"] = time.time()
            result["layout"] = layout
            self.project.add_export(result)
            return result

        return self.jobs.start("export", run)


def calls_signature(calls: list[dict]) -> str:
    """Which calls a re-tune learned from, so the UI can tell when there are newer ones."""
    return ",".join(sorted(f"{c['id']}:{c['label']}" for c in calls))


def fmt_time(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 60}:{t % 60:02d}" if t < 3600 else f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"
