"""scripts/check_reel.py passes a correct export and catches a wrong clip order."""

import json
import subprocess
import sys
import time
from pathlib import Path

from conftest import run
from watch_the_goal.app import App

ROOT = Path(__file__).resolve().parents[1]


def wait(app: App) -> None:
    while any(j.status == "running" for j in app.jobs.jobs.values()):
        time.sleep(0.1)


def check(reel: Path, state: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "check_reel.py"), str(reel), "--state", str(state)],
                          capture_output=True, text=True)


def test_check_reel_verifies_order_and_boundaries(tmp_path):
    # testsrc2 draws a moving pattern and a frame counter, so every frame is distinct.
    a, b = tmp_path / "a.mp4", tmp_path / "b.mov"
    run("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-f", "lavfi", "-i", "sine=frequency=440",
        "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(a))
    run("-f", "lavfi", "-i", "testsrc2=size=360x640:rate=25", "-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(b))
    app = App(tmp_path / "ws")
    app.import_paths([str(a), str(b)])
    wait(app)
    p = app.project
    ca, cb = (next(c for c in p.clips.values() if c["source_id"] == s["id"]) for s in p.sources)
    p.update_clip(ca["id"], start=1.0, end=4.5, status="kept")
    p.update_clip(cb["id"], start=0.5, end=3.0, status="kept")
    p.reorder([cb["id"], ca["id"]])
    job = app.export("landscape")
    wait(app)
    assert job.status == "done", job.error
    reel = Path(job.result["path"])

    state = tmp_path / "state.json"
    state.write_text(json.dumps(p.snapshot()))
    ok = check(reel, state)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert ok.stdout.count("[ok] clip") == 4  # start and end of both clips matched

    # Same reel, but claim the other order: the boundary checks must fail.
    st = json.loads(state.read_text())
    st["clips"].reverse()
    state.write_text(json.dumps(st))
    bad = check(reel, state)
    assert bad.returncode == 1
    assert "[FAIL] clip 1 start" in bad.stdout
