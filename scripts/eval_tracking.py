"""Does ByteTrack earn its place? Compare player evidence with and without tracking.

Runs in a throwaway environment so the app never depends on a tracker:

    uv run --no-project --with supervision==0.26.1 --with numpy \\
        python scripts/eval_tracking.py VIDEO:ANCHOR[:GOAL_AREA_JSON] ... \\
        [--detector rfdetr-nano] [--fps 10] [--out bench/tracking]

For each candidate window (anchor - 6 s to anchor + 2 s, the window the app
measures) it reports, for the "near goal" and "bunched" counts the ranking uses:

* the app's current reading: raw detections sampled at 1 fps
* raw detections at --fps (what tracking needs as input)
* tracked counts at --fps: only confirmed ByteTrack tracks are counted
* track fragmentation: distinct track ids per person seen in a frame
* detector time for the window at 1 fps and at --fps

Tracking only pays off if it changes which candidates clear the ranking
thresholds (detect.PLAYERS_NEAR_GOAL_* and BUNCH_*) for a cost the app can bear.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watch_the_goal.vision import detectors, players  # noqa: E402
from watch_the_goal.vision.frames import sample_frames, window_times  # noqa: E402


def counts(boxes: np.ndarray, area: dict | None, w: int, h: int) -> tuple[int, int]:
    near = players.near_goal(boxes, area, w, h) if area else 0
    return near, players.bunch(boxes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("windows", nargs="+", help="VIDEO:ANCHOR or VIDEO:ANCHOR:'{\"x\":..,\"y\":..,\"w\":..,\"h\":..}'")
    ap.add_argument("--detector", default="rfdetr-nano")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("bench/tracking"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    det = detectors.load(args.detector)
    rows = []
    for spec in args.windows:
        video, anchor, *rest = spec.split(":", 2)
        anchor = float(anchor)
        area = json.loads(rest[0]) if rest else None
        start, end = max(0.0, anchor - players.WINDOW_BEFORE), anchor + players.WINDOW_AFTER
        dense = window_times(start, end, args.fps)
        sparse = set(window_times(start, end, players.WINDOW_FPS))
        tracker = sv.ByteTrack(frame_rate=int(args.fps))
        raw1, raw10, tracked, seen, ids = [], [], [], [], set()
        t_all = t_sparse = 0.0
        for t, frame in sample_frames(video, dense):
            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            boxes = det.detect(frame)
            dt = time.perf_counter() - t0
            t_all += dt
            raw10.append(counts(boxes, area, w, h))
            if any(abs(t - s) < 1e-6 for s in sparse):
                t_sparse += dt
                raw1.append(counts(boxes, area, w, h))
            dets = sv.Detections(xyxy=boxes[:, :4].astype(np.float32), confidence=boxes[:, 4].astype(np.float32),
                                 class_id=np.zeros(len(boxes), int)) if len(boxes) else sv.Detections.empty()
            out = tracker.update_with_detections(dets)
            ids.update(int(i) for i in out.tracker_id)
            tb = np.hstack([out.xyxy, out.confidence[:, None]]) if len(out) else np.zeros((0, 5))
            tracked.append(counts(tb, area, w, h))
            seen.append(len(boxes))
        peak = lambda xs, k: max((x[k] for x in xs), default=0)  # noqa: E731
        row = {
            "video": Path(video).name, "anchor": anchor, "goal_area": bool(area), "frames": len(raw10),
            "near_goal_peak": {"raw_1fps": peak(raw1, 0), f"raw_{args.fps:g}fps": peak(raw10, 0),
                               "tracked": peak(tracked, 0)},
            "bunch_peak": {"raw_1fps": peak(raw1, 1), f"raw_{args.fps:g}fps": peak(raw10, 1),
                           "tracked": peak(tracked, 1)},
            "track_ids": len(ids), "median_people_per_frame": float(np.median(seen)) if seen else 0.0,
            "ids_per_person": round(len(ids) / max(float(np.median(seen)), 1.0), 2),
            "detector_seconds": {"1fps": round(t_sparse, 2), f"{args.fps:g}fps": round(t_all, 2)},
        }
        rows.append(row)
        print(f"{row['video']} @ {anchor:.0f}s  near goal {row['near_goal_peak']}  bunch {row['bunch_peak']}  "
              f"ids/person {row['ids_per_person']}  detector {row['detector_seconds']}")
    (args.out / "results.json").write_text(json.dumps({"detector": args.detector, "fps": args.fps,
                                                      "supervision": sv.__version__, "windows": rows}, indent=1))
    print(f"wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
