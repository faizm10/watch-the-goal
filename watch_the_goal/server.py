"""Local web UI: a small JSON API plus static files, standard library only."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import recap, review, signals
from .app import App
from .export import SIZES, ExportError
from .media import MediaError, extract_frame
from .pipeline import DETECTORS
from .project import ProjectError
from .vision.models import ModelUnavailable

STATIC = Path(__file__).parent / "static"
USER_ERRORS = (MediaError, ExportError, ProjectError, ModelUnavailable, review.ReviewError, ValueError, KeyError)
MAX_LIVE_FRAME = 8 << 20  # bytes; a 1080p JPEG is well under 1 MB


class Handler(BaseHTTPRequestHandler):
    app: App  # set by make_server
    server_version = "WatchTheGoal/0.1"

    def log_message(self, fmt, *args):  # keep the terminal quiet
        if os.environ.get("WTG_DEBUG"):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------- helpers

    def send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def send_file(self, path: Path, content_type: str | None = None, cache: bool = False) -> None:
        """Serve a file with HTTP Range support so <video> can seek."""
        if not path.exists():
            return self.send_error_json("File not found", 404)
        size = path.stat().st_size
        ctype = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        status = HTTPStatus.OK
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match and size:
            a, b = match.groups()
            if a:
                start = int(a)
                end = int(b) if b else size - 1
            elif b:
                start = max(0, size - int(b))
            end = min(end, size - 1)
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-cache")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    chunk = fh.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser stopped reading, e.g. after a seek

    def review_check(self) -> dict:
        cfg = review.config(self.app.project.settings.get("ai_review"))
        try:
            info = review.OllamaClient(cfg["endpoint"], cfg["model"], cfg["timeout"]).check()
            return {"ok": True, **info}
        except review.ReviewError as exc:
            return {"ok": False, "message": str(exc)}

    def state(self) -> dict:
        data = self.app.project.snapshot()
        data["can_undo"] = bool(self.app.project.history)
        data["jobs"] = self.app.jobs.active()
        data["workspace"] = str(self.app.project.workspace)
        return data

    # ------------------------------------------------------------- routing

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        path, query = url.path, parse_qs(url.query)
        try:
            if path in ("/", "/index.html"):
                return self.send_file(STATIC / "index.html", "text/html; charset=utf-8")
            if path.startswith("/static/"):
                target = (STATIC / path[len("/static/"):]).resolve()
                if STATIC.resolve() not in target.parents:
                    return self.send_error_json("Not found", 404)
                return self.send_file(target)
            if path == "/api/state":
                return self.send_json(self.state())
            if m := re.fullmatch(r"/api/jobs/(\w+)", path):
                job = self.app.jobs.get(m.group(1))
                return self.send_json(job.to_dict()) if job else self.send_error_json("Unknown job", 404)
            if m := re.fullmatch(r"/media/(\w+)", path):
                src = self.app.project.source(m.group(1))
                if src.get("preview") == "proxy" and src.get("preview_path"):
                    return self.send_file(Path(src["preview_path"]), "video/mp4")
                # Browsers play H.264 in a .mov fine when it is labelled as MP4.
                webm = src["path"].lower().endswith(".webm")
                return self.send_file(Path(src["path"]), "video/webm" if webm else "video/mp4")
            if m := re.fullmatch(r"/api/sources/(\w+)/frame", path):
                src = self.app.project.source(m.group(1))
                t = float(query.get("t", ["0"])[0])
                width = int(query.get("w", ["960"])[0])
                jpg = extract_frame(src["path"], t, min(max(width, 64), 1920))
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(jpg)
                return
            if m := re.fullmatch(r"/api/sources/(\w+)/activity", path):
                return self.send_json(self.app.activity(m.group(1)))
            if m := re.fullmatch(r"/api/sources/(\w+)/evidence", path):
                return self.send_json(self.app.timeline(m.group(1)))
            if m := re.fullmatch(r"/api/sources/(\w+)/explain", path):
                label = query.get("label", [None])[0]
                return self.send_json(self.app.explain(m.group(1), float(query["t"][0]), label))
            if m := re.fullmatch(r"/api/sources/(\w+)/labels\.json", path):
                src = self.app.project.source(m.group(1))
                body = json.dumps(self.app.labels_file(src["id"]), indent=1).encode()
                stem = re.sub(r"[^\w.-]+", "_", Path(src["name"]).stem)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Disposition", f'attachment; filename="{stem}.labels.json"')
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/live":
                return self.send_json({"detector": self.app.live_detector_name()})
            if m := re.fullmatch(r"/api/sources/(\w+)/waveform", path):
                src = self.app.project.source(m.group(1))
                start, end = (float(query[k][0]) for k in ("start", "end"))
                if not 0 <= start < end <= src["info"]["duration"] + .001:
                    raise ValueError("Invalid waveform range")
                peaks = signals.waveform(src["path"], start, end) if src["info"]["has_audio"] else []
                return self.send_json({"peaks": peaks})
            if path == "/api/artwork":
                kind = query["kind"][0]
                if kind not in ("title", "board"):
                    raise ValueError("Unknown artwork")
                text = query["text"][0]
                if len(text) > 128:
                    raise ValueError("Artwork text is too long")
                png = recap.artwork(kind, SIZES[query["layout"][0]], text)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(png)
                return
            if path == "/api/models":
                from .vision import models

                return self.send_json({"models": models.status()})
            if path == "/api/review/check":
                return self.send_json(self.review_check())
            if m := re.fullmatch(r"/exports/([\w.-]+\.mp4)", path):
                f = self.app.project.workspace / "exports" / m.group(1)
                return self.send_file(f, "video/mp4")
            return self.send_error_json("Not found", 404)
        except USER_ERRORS as exc:
            return self.send_error_json(str(exc))

    def do_PUT(self):
        url = urlparse(self.path)
        if url.path != "/api/upload":
            return self.send_error_json("Not found", 404)
        name = unquote(parse_qs(url.query).get("name", ["video.mp4"])[0])
        length = int(self.headers.get("Content-Length") or 0)
        free = shutil.disk_usage(self.app.project.workspace).free
        if length > free - (512 << 20):
            return self.send_error_json(
                f"Not enough disk space to copy {name} into the workspace. "
                "Free some space, or import it by path instead (the file is then read in place).", 507)
        try:
            src = self.app.import_upload(name, self.rfile, length)
        except USER_ERRORS as exc:
            return self.send_error_json(str(exc))
        return self.send_json({"source": src})

    def do_POST(self):
        url = urlparse(self.path)
        path = url.path
        try:
            if m := re.fullmatch(r"/api/sources/(\w+)/live", path):
                # A JPEG of the frame the browser is showing, not JSON.
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_LIVE_FRAME:
                    raise ValueError("Send one JPEG frame.")
                t = float(parse_qs(url.query).get("t", ["0"])[0])
                return self.send_json(self.app.live(m.group(1), self.rfile.read(length), t))
            body = self.read_json()
            p = self.app.project
            if path == "/api/import":
                return self.send_json(self.app.import_paths(body.get("paths", [])))
            if m := re.fullmatch(r"/api/sources/(\w+)/goals", path):
                p.set_goal_areas(m.group(1), body.get("areas", []))
                # Tuning was learned from the old evidence; the calls stay, so re-tuning is one click.
                p.update_source(m.group(1), tuning=None, coaching=None)
                job = self.app.analyze(m.group(1))
                return self.send_json({"job": job.to_dict()})
            if m := re.fullmatch(r"/api/sources/(\w+)/analyze", path):
                return self.send_json({"job": self.app.analyze(m.group(1)).to_dict()})
            if m := re.fullmatch(r"/api/sources/(\w+)/review", path):
                return self.send_json({"job": self.app.review(m.group(1)).to_dict()})
            if m := re.fullmatch(r"/api/sources/(\w+)/calls", path):
                t = body.get("t")
                return self.send_json(self.app.make_call(m.group(1), body.get("label"),
                                                         None if t is None else float(t), body.get("clip_id")))
            if m := re.fullmatch(r"/api/sources/(\w+)/retune", path):
                return self.send_json({"job": self.app.retune(m.group(1)).to_dict()})
            if m := re.fullmatch(r"/api/sources/(\w+)/tuning/reset", path):
                return self.send_json({"job": self.app.reset_tuning(m.group(1)).to_dict()})
            if m := re.fullmatch(r"/api/models/([\w-]+)/download", path):
                return self.send_json({"job": self.app.download_model(m.group(1)).to_dict()})
            if path == "/api/clips":
                clip = p.add_manual_clip(body["source_id"], float(body["start"]), float(body["end"]))
                return self.send_json({"clip": clip})
            if path == "/api/undo":
                return self.send_json(p.undo())
            if m := re.fullmatch(r"/api/clips/(\w+)/proposal", path):
                return self.send_json(review.propose_edit(p, m.group(1), body.get("request")))
            if path == "/api/recap":
                template = body.get("template")
                if template not in ("recap", "plain"):
                    raise ValueError("Unknown template")
                title = recap.clean_text(body.get("title", ""), 64)
                with p.lock:
                    updates = [(p.source(sid), {k: recap.clean_text(row.get(k, ""), limit)
                                for k, limit in (("home", 28), ("away", 28), ("score", 12))})
                               for sid, row in body.get("sources", {}).items()]
                    for src, board in updates:
                        src["scoreboard"] = board
                    p.settings.update(template=template, title=title)
                    p.save()
                return self.send_json({"ok": True})
            if path == "/api/reel/auto":
                target = body.get("target")
                if target is not None and (isinstance(target, bool) or not isinstance(target, (int, float))
                                           or not 1 <= target <= 600):
                    raise ValueError("Reel target must be between 1 and 600 seconds, or none for every suggestion.")
                return self.send_json(p.auto_reel(target))
            if path == "/api/order":
                p.reorder(body.get("ids", []))
                return self.send_json({"ok": True})
            if path == "/api/settings":
                if "layout" in body:
                    p.settings["layout"] = body["layout"]
                if "detector" in body:
                    if body["detector"] not in DETECTORS:
                        raise ProjectError(f"Detector must be one of {', '.join(DETECTORS)}")
                    p.settings["detector"] = body["detector"]
                if isinstance(body.get("ai_review"), dict):
                    cfg = review.config({**p.settings.get("ai_review", {}), **body["ai_review"]})
                    review.require_local(cfg["endpoint"])
                    cfg["enabled"] = bool(cfg["enabled"])
                    p.settings["ai_review"] = cfg
                p.save()
                return self.send_json({"settings": p.settings})
            if path == "/api/export":
                return self.send_json({"job": self.app.export(body.get("layout") or p.settings["layout"]).to_dict()})
            if path == "/api/reset":
                p.reset()
                return self.send_json({"ok": True})
            return self.send_error_json("Not found", 404)
        except USER_ERRORS as exc:
            return self.send_error_json(str(exc))

    def do_PATCH(self):
        path = urlparse(self.path).path
        try:
            if m := re.fullmatch(r"/api/clips/(\w+)", path):
                body = self.read_json()
                allowed = {k: body[k] for k in ("start", "end", "status", "apply_trim", "crop", "expected") if k in body}
                return self.send_json({"clip": self.app.project.update_clip(m.group(1), **allowed)})
            return self.send_error_json("Not found", 404)
        except USER_ERRORS as exc:
            return self.send_error_json(str(exc))

    def do_DELETE(self):
        path = urlparse(self.path).path
        try:
            if m := re.fullmatch(r"/api/clips/(\w+)", path):
                self.app.project.delete_clip(m.group(1))
                return self.send_json({"ok": True})
            if m := re.fullmatch(r"/api/sources/(\w+)/calls/(\w+)", path):
                self.app.project.remove_call(m.group(1), m.group(2))
                return self.send_json({"ok": True})
            if m := re.fullmatch(r"/api/sources/(\w+)", path):
                self.app.project.remove_source(m.group(1))
                return self.send_json({"ok": True})
            return self.send_error_json("Not found", 404)
        except USER_ERRORS as exc:
            return self.send_error_json(str(exc))


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
