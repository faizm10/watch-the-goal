"""Drive the real UI end to end with Playwright and export a reel.

    uv run --group e2e python scripts/e2e_ui.py LONG_RECORDING [PHONE_CLIP ...] \\
        [--goal SECONDS] [--goal-area x,y,w,h] [--review ok|fail|off] [--detector none|yolox|rfdetr-nano] \\
        [--channel chrome] [--out bench/e2e]

What it does, the way a person would:

1. starts the app on a free port with a fresh workspace
2. sets the player detector and AI review in the Analysis dialog
   (``--review fail`` points the reviewer at a closed port to exercise the fallback)
3. imports the long recording and the first phone clip by pasting a folder
   path, and uploads the remaining phone clips through the file picker
4. waits for analysis (and AI review), then checks every suggestion on the
   long recording has a reason and a rank
5. opens the suggestion containing ``--goal`` (or the strongest one), nudges
   its start 5 s earlier, undoes the change, drags the start back, and keeps it
6. keeps every phone clip, moves the first reel item down one place and undoes the reorder
7. optionally sets up the recap and checks preview frames, plays the export, then runs scripts/check_reel.py

Screenshots, the project state and the reel land in --out. Exit code 1 if any
step or reel check fails. Needs FFmpeg and a Chromium (``playwright install
chromium``) or ``--channel chrome``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def state(page: Page, base: str) -> dict:
    return page.request.get(f"{base}/api/state").json()


def wait_jobs(page: Page, base: str, timeout: float, kinds: tuple[str, ...] = ()) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = state(page, base)
        running = [j for j in st["jobs"] if j["status"] == "running" and (not kinds or j["kind"] in kinds)]
        if not running:
            return st
        time.sleep(1)
    raise TimeoutError(f"jobs still running after {timeout:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("long", type=Path)
    ap.add_argument("phone", nargs="*", type=Path)
    ap.add_argument("--goal", type=float, help="a goal time in the long recording; its suggestion is kept")
    ap.add_argument("--goal-area", help="normalised x,y,w,h of a fixed camera's goal (sent via the API)")
    ap.add_argument("--review", choices=["off", "ok", "fail"], default="off")
    ap.add_argument("--review-model", default="qwen3.5:9b")
    ap.add_argument("--review-max", type=int, default=3)
    ap.add_argument("--detector", default="none")
    ap.add_argument("--channel", help="browser channel, e.g. chrome; default is Playwright's Chromium")
    ap.add_argument("--layout", choices=["portrait", "landscape"], default="landscape")
    ap.add_argument("--recap", action="store_true", help="test the title, source scoreboard and selected framing")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds to wait for analysis/review/export")
    ap.add_argument("--out", type=Path, default=Path("bench/e2e"))
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    def check(name: str, ok: bool) -> None:
        results[name] = bool(ok)
        print(f"  [{'ok' if ok else 'FAIL'}] {name}", flush=True)

    ws = Path(tempfile.mkdtemp(prefix="wtg-e2e-"))
    folder = ws / "import"
    folder.mkdir()
    # One path import: a folder holding the long recording and the first phone clip.
    for f in [args.long, *args.phone[:1]]:
        os.symlink(f.resolve(), folder / f.name)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = subprocess.Popen([str(Path(sys.executable).parent / "watch-the-goal"), "--workspace", str(ws / "workspace"),
                               "--port", str(port), "--no-browser"], cwd=ROOT,
                              stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel, headless=not args.headed)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(base)
            page.wait_for_load_state("networkidle")

            # 2. Analysis settings through the dialog.
            page.click("#settings-btn")
            expect(page.locator("#settings-dialog")).to_be_visible()
            options = page.locator("#detector-select option").all_inner_texts()
            check("settings: detector menu lists none, YOLOX and RF-DETR",
                  len(options) == 3)
            page.select_option("#detector-select", args.detector)
            if args.review != "off":
                page.check("#review-enabled")
                page.fill("#review-model", args.review_model)
                page.fill("#review-endpoint", "http://127.0.0.1:11434" if args.review == "ok" else "http://127.0.0.1:9")
                page.fill("#review-max", str(args.review_max))
            page.screenshot(path=str(out / "1-settings.png"))
            page.click("#settings-save")
            expect(page.locator("#settings-dialog")).to_be_hidden()
            settings = state(page, base)["settings"]
            check(f"settings saved: detector {args.detector}, AI review {args.review}",
                  settings["detector"] == args.detector and settings["ai_review"]["enabled"] == (args.review != "off"))

            # 3. Import: folder path, then file uploads.
            page.fill("#path-input", str(folder))
            page.press("#path-input", "Enter")
            if args.phone[1:]:
                page.set_input_files("#file-input", [str(f) for f in args.phone[1:]])
            expected_sources = 1 + len(args.phone)
            t0 = time.time()
            while len(state(page, base)["sources"]) < expected_sources and time.time() - t0 < 120:
                time.sleep(0.5)
            st = state(page, base)
            check(f"import: {expected_sources} sources (1 by folder path with {min(1, len(args.phone))} phone clip, "
                  f"{len(args.phone[1:])} uploaded)", len(st["sources"]) == expected_sources)
            long_src = next(s for s in st["sources"] if s["name"] == args.long.name)
            if args.goal_area:
                x, y, w, h = (float(v) for v in args.goal_area.split(","))
                wait_jobs(page, base, args.timeout)
                page.request.post(f"{base}/api/sources/{long_src['id']}/goals",
                                  data=json.dumps({"areas": [{"x": x, "y": y, "w": w, "h": h}]}),
                                  headers={"Content-Type": "application/json"})
                page.request.post(f"{base}/api/sources/{long_src['id']}/analyze", data="{}",
                                  headers={"Content-Type": "application/json"})

            # 4. Analysis (and review) of the long recording.
            t_analysis = time.time()
            time.sleep(2)
            st = wait_jobs(page, base, args.timeout)
            long_src = next(s for s in st["sources"] if s["id"] == long_src["id"])
            print(f"  analysis + review took {time.time() - t_analysis:.0f}s: {long_src.get('analysis')}", flush=True)
            sugg = [c for c in st["clips"] if c["source_id"] == long_src["id"] and c["origin"] == "detected"]
            check(f"suggestions: {len(sugg)} on the long recording, each with a reason and a rank",
                  sugg and all(c["reasons"] for c in sugg) and all(c.get("rank") for c in sugg))
            rv = long_src.get("review") or {}
            if args.review == "ok":
                check(f"AI review ran: {rv.get('reviewed', 0)} checked", rv.get("status") in ("done", "partial")
                      and rv.get("reviewed", 0) > 0)
            elif args.review == "fail":
                check("AI review failed safely: status unavailable, suggestions untouched",
                      rv.get("status") == "unavailable" and not any(c.get("review") for c in sugg))
                expect(page.locator(".source .status.err")).to_contain_text("AI review unavailable")
            page.screenshot(path=str(out / "2-analysed.png"))

            # 5. Open the goal suggestion, trim, keep.
            if args.goal is not None:
                goal = next((c for c in sugg if c["start"] <= args.goal <= c["end"]), None)
                check(f"goal at {args.goal:.1f}s is inside a suggestion", goal is not None)
            else:
                goal = None
            goal = goal or min(sugg, key=lambda c: c["rank"])
            page.click(f'.block[data-clip="{goal["id"]}"]')
            expect(page.locator("#editor")).to_be_visible()
            expect(page.locator("#editor .reasons .reason").first).to_be_visible()
            if args.review == "ok" and goal.get("review"):
                expect(page.locator("#editor .reason.ai")).to_be_visible()
            page.screenshot(path=str(out / "3-editor.png"))
            before = goal["start"]
            page.click('[data-act="start-5"]')
            time.sleep(0.5)
            moved = next(c for c in state(page, base)["clips"] if c["id"] == goal["id"])
            page.click("#undo-btn")
            page.wait_for_function("t => Math.abs(clipById(sel.clipId).start-t)<.005", arg=before)
            page.click('[data-act="finish"]')
            page.wait_for_function("() => Math.abs(player.currentTime-eventTime(clipById(sel.clipId)))<.05")
            check("quick lead-in, undo, jump to finish", abs(moved["start"] - max(0, before - 5)) < .05)
            handle = page.locator("#h-start").bounding_box()
            x, y = handle["x"] + handle["width"] / 2, handle["y"] + handle["height"] / 2
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x - 35, y, steps=12)
            with page.expect_response(lambda r: r.request.method == "PATCH"):
                page.mouse.up()
            moved = next(c for c in state(page, base)["clips"] if c["id"] == goal["id"])
            check("drag keeps the finish and extends the actual start", moved["start"] < before and
                  moved["anchor"] == goal["anchor"] and moved["end"] == goal["end"])
            page.click('[data-act="keep"]')

            # 6. Keep the phone clips, reorder.
            st = state(page, base)
            for c in st["clips"]:
                if c["origin"] == "short-clip" and c["status"] != "kept":
                    page.click(f'.block[data-clip="{c["id"]}"]')
                    page.click('[data-act="keep"]')
            time.sleep(0.5)
            reel = [c["id"] for c in state(page, base)["clips"] if c["status"] == "kept"]
            order_before = [c for c in state(page, base)["order"] if c in reel]
            page.click(f'.reel-item[data-clip="{order_before[0]}"] [data-move="1"]')
            time.sleep(0.5)
            order_after = [c for c in state(page, base)["order"] if c in reel]
            check(f"reel: {len(order_before)} clips kept; first moved down one place",
                  len(order_before) == 1 + len(args.phone) and order_after[:2] == order_before[1::-1])
            page.screenshot(path=str(out / "4-reel.png"))

            # 7. Export.
            if len(reel) > 1:
                page.click("#undo-btn")
                page.wait_for_function("ids => JSON.stringify(reel().map(c=>c.id)) === JSON.stringify(ids)", arg=order_before)
                check("undo restored reel order", True)
            page.select_option("#layout-select", args.layout)
            page.click(f'.block[data-clip="{goal["id"]}"]')
            if args.recap:
                page.select_option("#template-select", "recap")
                page.fill("#recap-title", "Sunday football")
                field = page.locator(f'#recap-sources [data-sid="{long_src["id"]}"]')
                for key, value in [("home", "Blue"), ("away", "Red"), ("score", "1–0")]:
                    field.locator(f'[name="{key}"]').fill(value)
                page.click("#recap-form .primary")
                expect(page.locator("#recap-dialog")).to_be_hidden()
                if args.layout == "portrait":
                    page.select_option("#frame-mode", "crop")
                    page.locator("#crop-position").fill("82")
                    page.wait_for_function("() => clipById(sel.clipId).crop === .82")
                page.click("#preview-reel-btn")
                page.wait_for_function("() => reelQueue?.intro && [...images.values()].some(i=>i.src.includes('kind=title') && i.complete)")
                page.wait_for_timeout(80)
                title_png = page.locator("#composition").evaluate("c=>c.toDataURL()")
                page.click("#preview-reel-btn")
                page.click(f'.block[data-clip="{goal["id"]}"]')
                page.click('[data-act="finish"]')
                page.wait_for_function("() => !player.seeking && player.readyState>=2 && [...images.values()].some(i=>i.src.includes('kind=board') && i.complete)")
                page.wait_for_timeout(100)
                preview_png = page.locator("#composition").evaluate("c=>c.toDataURL()")
                source_t = page.locator("#player").evaluate("v=>v.currentTime")
                page.screenshot(path=str(out / "framing.png"))
            page.click("#export-btn")
            time.sleep(1)
            st = wait_jobs(page, base, args.timeout, ("export",))
            job = next(j for j in st["jobs"] if j["kind"] == "export")
            check(f"export: {job['status']} {job.get('error') or ''}".strip(), job["status"] == "done")
            page.screenshot(path=str(out / "5-exported.png"))
            if job["status"] == "done":
                video = page.locator("#exports video").first
                video.evaluate("v => v.play()")
                page.wait_for_function("() => document.querySelector('#exports video').currentTime > .5")
                check("export plays in the browser", video.evaluate("v=>v.videoWidth>0 && !v.error"))
                if args.recap:
                    offset = 2
                    for c in [c for c in st["clips"] if c["status"] == "kept"]:
                        if c["id"] == goal["id"]:
                            break
                        offset += c["end"] - c["start"]
                    capture = cv2.VideoCapture(job["result"]["path"])
                    for label, png, at in [("title", title_png, .5), ("goal", preview_png, offset + source_t - moved["start"])]:
                        ref = cv2.imdecode(np.frombuffer(base64.b64decode(png.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
                        capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
                        ok, frame = capture.read()
                        assert ok
                        frame = cv2.resize(frame, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(out / f"{label}-preview.png"), ref)
                        cv2.imwrite(str(out / f"{label}-export.png"), frame)
                        diff = cv2.absdiff(frame, ref)
                        check(f"{label} preview matches export (mean channel error {diff.mean():.2f})", diff.mean() < 6)
                        if label == "title":
                            ink = ref.min(axis=2) > 170
                            check("opening title text visible in both renders",
                                  ink.sum() > 200 and (frame.min(axis=2)[ink] > 100).mean() > .9)
                        if label == "goal":
                            unit = min(ref.shape[:2])
                            y, x = int(unit * .05), int(unit * .06)
                            patch = diff[y:y + int(.05 * unit), x:x + int(.22 * unit)]
                            check("scoreboard region matches preview", patch.mean() < 9)
                    capture.release()
            check("no JavaScript errors", not errors)
            for e in errors:
                print("   ", e)
            browser.close()

        # Keep the QA snapshot usable after deleting the throwaway upload directory.
        input_paths = {f.name: str(f.resolve()) for f in [args.long, *args.phone]}
        for src in st["sources"]:
            src["path"] = src["info"]["path"] = input_paths[src["name"]]
        (out / "project.json").write_text(json.dumps(st, indent=1))
        if job["status"] == "done":
            reel_path = out / "reel.mp4"
            shutil.copy(job["result"]["path"], reel_path)
            qa = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_reel.py"), str(reel_path),
                                 "--workspace", str(ws / "workspace"), "--json", str(out / "check_reel.json")],
                                capture_output=True, text=True)
            print(qa.stdout + qa.stderr, end="")
            check("reel QA (scripts/check_reel.py)", qa.returncode == 0)
            offset = 2.0 if args.recap else 0.0
            for c in st["clips"]:  # state lists clips in reel order
                if c["status"] != "kept":
                    continue
                if c["id"] == goal["id"]:
                    print(f"  goal clip {c['start']:.2f}-{c['end']:.2f}s plays at "
                          f"{offset:.2f}-{offset + c['end'] - c['start']:.2f}s of the reel")
                    break
                offset += c["end"] - c["start"]
    finally:
        server.terminate()
        server.wait(10)
        shutil.rmtree(ws, ignore_errors=True)
    print(f"\n{sum(results.values())}/{len(results)} checks passed; screenshots and reel in {out}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
