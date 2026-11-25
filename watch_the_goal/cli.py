"""Command line entry point.

    watch-the-goal [VIDEO ...]            open the local web app
    watch-the-goal detect VIDEO [...]     print suggested highlights as JSON
    watch-the-goal models [download NAME] list or fetch optional detector models
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path

from . import __version__

DEFAULT_WORKSPACE = Path.home() / "WatchTheGoal"


def parse_area(text: str) -> dict:
    try:
        x, y, w, h = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "goal area must be x,y,w,h as fractions of the frame, e.g. 0,0.4,0.12,0.25") from None
    return {"x": x, "y": y, "w": w, "h": h}


def run_detect(argv: list[str]) -> int:
    from . import detect, pipeline
    from . import review as ai
    from .media import MediaError, probe

    ap = argparse.ArgumentParser(prog="watch-the-goal detect",
                                 description="Print suggested highlight moments for a video as JSON.")
    ap.add_argument("video")
    ap.add_argument("--goal", action="append", type=parse_area, default=[],
                    help="goal area as x,y,w,h fractions of the frame (repeatable)")
    ap.add_argument("--pre-roll", type=float, default=detect.PRE_ROLL)
    ap.add_argument("--post-roll", type=float, default=detect.POST_ROLL)
    ap.add_argument("--detector", choices=pipeline.DETECTORS, default="none",
                    help="optional person detector (needs `watch-the-goal models download NAME`)")
    ap.add_argument("--debug-dir", type=Path, help="save overlays and candidate timestamps here")
    ap.add_argument("--review", action="store_true", help="also ask a local Ollama vision model about each moment")
    ap.add_argument("--review-model", default=ai.DEFAULTS["model"])
    ap.add_argument("--review-max", type=int, default=ai.DEFAULTS["max_candidates"])
    args = ap.parse_args(argv)
    try:
        info = probe(args.video)
    except MediaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result = pipeline.analyze(info.path, info.to_dict(), args.goal, pipeline.Options(
        detector=args.detector, pre_roll=args.pre_roll, post_roll=args.post_roll, debug_dir=args.debug_dir))
    for note in result.notes:
        print(f"note: {note}", file=sys.stderr)
    moments = [m.to_dict() for m in result.moments]
    if args.review:
        client = ai.OllamaClient(ai.DEFAULTS["endpoint"], args.review_model, ai.DEFAULTS["timeout"])
        try:
            client.check()
        except ai.ReviewError as exc:
            print(f"note: AI review unavailable, moments unchanged: {exc}", file=sys.stderr)
            client = None
        for m in sorted(moments, key=lambda m: m["rank"] or 0)[: args.review_max] if client else []:
            w0, w1 = ai.window_for(m["start"], m["end"], m["anchor"], info.duration)
            try:
                m["review"] = {"status": "done", "window": [w0, w1],
                               **ai.review_window(client, info.path, w0, w1, ai.DEFAULTS["frames"]).to_dict()}
            except ai.ReviewError as exc:
                m["review"] = {"status": "failed", "message": str(exc)}
    json.dump({"video": info.path, "duration": info.duration, "stats": result.stats,
               "moments": moments}, sys.stdout, indent=2)
    print()
    return 0


def run_models(argv: list[str]) -> int:
    from .vision import models

    ap = argparse.ArgumentParser(prog="watch-the-goal models",
                                 description="List or download the optional person-detector models.")
    ap.add_argument("action", nargs="?", choices=["list", "download"], default="list")
    ap.add_argument("names", nargs="*", help=f"models to download: {', '.join(models.MODELS)}")
    args = ap.parse_args(argv)
    if args.action == "list":
        for m in models.status():
            state = "downloaded" if m["downloaded"] else "not downloaded"
            print(f"{m['name']:12} {m['size'] / 1e6:5.0f} MB  {m['license']:10}  {state}  {m['path']}")
            print(f"{'':12} {m['description']}; {m['source']}")
        return 0
    for name in args.names or list(models.MODELS):
        try:
            path = models.download(name, progress=lambda f, n=name: print(f"\r{n}: {f * 100:3.0f}%", end="", flush=True))
            print(f"\r{name}: ok, {path}")
        except models.ModelUnavailable as exc:
            print(f"\nerror: {exc}", file=sys.stderr)
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "detect":
        return run_detect(argv[1:])
    if argv and argv[0] == "models":
        return run_models(argv[1:])

    ap = argparse.ArgumentParser(prog="watch-the-goal",
                                 description="Turn soccer videos into one highlight reel. Opens a local web app.")
    ap.add_argument("videos", nargs="*", help="video files or folders to import on start")
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE,
                    help=f"where the project, previews and exported reels are kept (default: {DEFAULT_WORKSPACE})")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    ap.add_argument("--version", action="version", version=f"watch-the-goal {__version__}")
    args = ap.parse_args(argv)

    from .app import App
    from .media import MediaError, require_ffmpeg
    from .server import make_server

    try:
        require_ffmpeg()
    except MediaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    app = App(args.workspace)
    if args.videos:
        result = app.import_paths(args.videos)
        for err in result["errors"]:
            print(f"skipped: {err}", file=sys.stderr)
    try:
        server = make_server(app, args.host, args.port)
    except OSError as exc:
        print(f"error: can't listen on {args.host}:{args.port} ({exc.strerror}). "
              f"Try another port with --port.", file=sys.stderr)
        return 1
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Watch the Goal is running at {url}")
    print(f"Workspace: {app.project.workspace}")
    print("Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0
