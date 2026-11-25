"""Build the demo reel without the UI, and check it.

    python scripts/demo_reel.py demo/footage [--out demo/reel.mp4]

Does what a person would do in the app: import the long match and the three
phone clips, mark the goals, keep the first suggested goal, export. Then it
checks the exported MP4: clip count and order, duration, and that the match
highlight contains the whole move (turnover, assist and finish).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watch_the_goal.app import App  # noqa: E402

PHONE_CLIPS = ["phone_goal_portrait.mov", "phone_goal_60fps.mp4", "phone_chance_noaudio.mp4"]


def wait(app: App) -> None:
    while any(j.status == "running" for j in app.jobs.jobs.values()):
        time.sleep(0.2)
    failed = [j for j in app.jobs.jobs.values() if j.status == "error"]
    if failed:
        raise SystemExit(f"{failed[0].kind} failed: {failed[0].error}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("footage", type=Path)
    ap.add_argument("--out", type=Path, default=Path("demo/reel.mp4"))
    args = ap.parse_args()
    labels = json.loads((args.footage / "labels.json").read_text())

    with tempfile.TemporaryDirectory() as ws:
        app = App(Path(ws))
        files = [args.footage / labels["video"], *(args.footage / n for n in PHONE_CLIPS)]
        result = app.import_paths([str(f) for f in files])
        if result["errors"]:
            raise SystemExit("; ".join(result["errors"]))
        wait(app)
        match = app.project.sources[0]
        app.project.set_goal_areas(match["id"], labels["goal_areas"])
        app.analyze(match["id"])
        wait(app)

        suggestions = [c for c in app.project.reel() + [app.project.clip(i) for i in app.project.order]
                       if c["source_id"] == match["id"] and c["origin"] == "detected"]
        print(f"suggestions on the match: {len(suggestions)}")
        for c in suggestions:
            print(f"  {c['start']:7.2f}-{c['end']:7.2f}s  {'; '.join(c['reasons'])}")
        goal = labels["goal_details"][0]["t"]
        highlight = next(c for c in suggestions if c["start"] <= goal <= c["end"])
        app.project.update_clip(highlight["id"], status="kept")
        # Match highlight first, then the phone clips in import order.
        app.project.reorder([highlight["id"]] + [c["id"] for c in app.project.reel() if c["id"] != highlight["id"]])

        reel = app.project.reel()
        job = app.export("landscape")
        wait(app)
        out = Path(ws) / "exports" / job.result["name"]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, args.out)

    expected = sum(c["end"] - c["start"] for c in reel)
    turnover = goal - 9.0
    checks = {
        "4 clips in the reel": len(reel) == 4,
        "match highlight is first": reel[0]["id"] == highlight["id"],
        "highlight includes the turnover (buildup)": highlight["start"] <= turnover,
        "highlight includes the finish": highlight["end"] >= goal + 1,
        "reel duration matches the clips": abs(job.result["duration"] - expected) < 0.15,
        "reel is 1920x1080": (job.result["width"], job.result["height"]) == (1920, 1080),
    }
    print(f"\nreel: {args.out} ({job.result['duration']:.2f}s, expected {expected:.2f}s)")
    print(f"highlight: {highlight['start']:.2f}-{highlight['end']:.2f}s around the goal at {goal}s "
          f"(turnover at {turnover}s); plays 0:00-{highlight['end'] - highlight['start']:.1f}s of the reel")
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
