"""Assisted labelling of people in a few frames, for scripts/bench_detectors.py.

1. Propose: every detector, tiled and at a low threshold, suggests boxes.
   Their union is numbered and drawn on zoomed crops for you to check.

       python scripts/label_people.py propose VIDEO:SECONDS [VIDEO:SECONDS ...] --out bench/people

2. Decide: write a JSON file listing, per frame, the proposal numbers that are
   NOT people and any people every detector missed (centre x, centre y,
   height, in frame pixels; the crops have a 50 px grid to read them off):

       {"frames": [{"reject": [3, 17], "add": [[812, 233, 24]]}, ...]}

3. Finalize into the labels file the benchmark reads:

       python scripts/label_people.py finalize bench/people/draft.json decisions.json --out people.labels.json

Labels store each person as [cx, cy, height] in pixels of the full frame.
Keep labels of private footage out of Git, like the footage itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watch_the_goal.vision import detectors  # noqa: E402
from watch_the_goal.vision.frames import sample_frames  # noqa: E402
from watch_the_goal.vision.overlay import label  # noqa: E402

PROPOSE_THRESHOLD = 0.15


def parse_frame(spec: str) -> tuple[str, float]:
    video, _, t = spec.rpartition(":")
    return video, float(t)


def grid(img: np.ndarray, x0: int, y0: int, step: int = 50) -> None:
    h, w = img.shape[:2]
    for x in range((x0 // step + 1) * step, x0 + w, step):
        c = (90, 90, 90) if x % 250 else (0, 160, 255)
        cv2.line(img, (x - x0, 0), (x - x0, 6 if x % 250 else 14), c, 1)
        if x % 250 == 0:
            cv2.putText(img, str(x), (x - x0 + 2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    for y in range((y0 // step + 1) * step, y0 + h, step):
        c = (90, 90, 90) if y % 250 else (0, 160, 255)
        cv2.line(img, (0, y - y0), (6 if y % 250 else 14, y - y0), c, 1)
        if y % 250 == 0:
            cv2.putText(img, str(y), (16, y - y0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)


def propose(args) -> int:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dets = [detectors.load(n, threshold=PROPOSE_THRESHOLD, tiles=(3, 2)) for n in detectors.NAMES]
    draft = {"frames": []}
    for i, spec in enumerate(args.frames):
        video, t = parse_frame(spec)
        (_, frame), = sample_frames(video, [t])
        h, w = frame.shape[:2]
        boxes = detectors.nms(np.concatenate([d.detect(frame) for d in dets]), 0.4)
        boxes = boxes[np.argsort(boxes[:, 1])]  # number top to bottom
        draft["frames"].append({"video": os.path.relpath(Path(video).resolve(), args.out.resolve()), "t": t,
                                "width": w, "height": h, "proposals": boxes.round(1).tolist()})
        # Two zoomed halves with numbered boxes and a pixel grid.
        for half, (x0, x1) in enumerate(((0, w // 2 + 60), (w // 2 - 60, w))):
            crop = frame[:, x0:x1].copy()
            for n, (bx0, by0, bx1, by1, _) in enumerate(boxes):
                if bx1 < x0 or bx0 > x1:
                    continue
                cv2.rectangle(crop, (int(bx0 - x0), int(by0)), (int(bx1 - x0), int(by1)), (0, 255, 255), 1)
                label(crop, str(n), (int(bx0 - x0), max(12, int(by0) - 2)), 0.4)
            grid(crop, x0, 0)
            crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(out / f"frame{i}_{'LR'[half]}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"frame {i}: {spec} -> {len(boxes)} proposals, crops frame{i}_L.jpg / frame{i}_R.jpg")
    (out / "draft.json").write_text(json.dumps(draft, indent=1))
    print(f"wrote {out / 'draft.json'}")
    return 0


def finalize(args) -> int:
    draft = json.loads(args.draft.read_text())
    decisions = json.loads(args.decisions.read_text())["frames"]
    out = {"frames": []}
    for f, d in zip(draft["frames"], decisions, strict=True):
        people = []
        for n, (x0, y0, x1, y1, _) in enumerate(f["proposals"]):
            if n not in set(d.get("reject", [])):
                people.append([round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1), round(y1 - y0, 1)])
        people += [list(map(float, p)) for p in d.get("add", [])]
        video = (args.draft.parent / f["video"]).resolve()
        out["frames"].append({"video": os.path.relpath(video, args.out.parent.resolve()), "t": f["t"],
                              "width": f["width"], "height": f["height"], "people": people,
                              "ignore": d.get("ignore", [])})
        print(f"{Path(f['video']).name} @ {f['t']}s: {len(people)} people "
              f"({len(d.get('reject', []))} proposals rejected, {len(d.get('add', []))} added)")
    args.out.write_text(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("frames", nargs="+", help="VIDEO:SECONDS")
    p.add_argument("--out", type=Path, default=Path("bench/people"))
    f = sub.add_parser("finalize")
    f.add_argument("draft", type=Path)
    f.add_argument("decisions", type=Path)
    f.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    return propose(args) if args.cmd == "propose" else finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
