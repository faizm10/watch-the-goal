"""Compare person detectors on labelled frames: recall (overall and for
distant players), false positives, and time per frame.

    python scripts/bench_detectors.py people.labels.json \\
        [--detectors yolox,rfdetr-nano] [--tiles 1x1,2x2] [--threshold 0.3] [--out bench/detectors]

Labels come from scripts/label_people.py: each person is [cx, cy, height].
A detection matches a person when its centre is within half the person's
height (at least 8 px) and its height is within 0.5-2x. Each person matches
at most one detection; leftover detections are false positives unless they
fall in an ``ignore`` box. "Distant" means under 60 px tall in a 1080-line
frame (scaled for other sizes): a player far across a wide shot.

Overlays (green = matched, red = false positive, blue ring = missed) are saved
to --out so you can see what each detector did.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watch_the_goal.vision import detectors  # noqa: E402
from watch_the_goal.vision.frames import sample_frames  # noqa: E402
from watch_the_goal.vision.overlay import label  # noqa: E402

DISTANT = 60.0  # px at 1080 lines


def match(dets: np.ndarray, people: np.ndarray, ignore: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (matched flag per person, false-positive flag per detection)."""
    hit = np.zeros(len(people), bool)
    fp = np.ones(len(dets), bool)
    for i in np.argsort(-dets[:, 4]) if len(dets) else []:
        x0, y0, x1, y1, _ = dets[i]
        cx, cy, h = (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0
        best, best_d = None, None
        for j, (px, py, ph) in enumerate(people):
            if hit[j]:
                continue
            tol = max(ph / 2, 8.0)
            d = max(abs(cx - px), abs(cy - py))
            if d <= tol and 0.5 <= h / max(ph, 1) <= 2.0 and (best_d is None or d < best_d):
                best, best_d = j, d
        if best is not None:
            hit[best] = True
            fp[i] = False
        elif any(ix0 <= cx <= ix1 and iy0 <= cy <= iy1 for ix0, iy0, ix1, iy1 in ignore):
            fp[i] = False
    return hit, fp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--detectors", default=",".join(detectors.NAMES))
    ap.add_argument("--tiles", default="1x1,2x2")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=Path("bench/detectors"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = json.loads(args.labels.read_text())
    frames = []
    for f in labels["frames"]:
        video = (args.labels.parent / f["video"]).resolve()
        (_, img), = sample_frames(str(video), [f["t"]])
        frames.append((f, img, np.array(f["people"], dtype=float).reshape(-1, 3)))
    total = sum(len(p) for _, _, p in frames)
    print(f"{len(frames)} frames, {total} labelled people "
          f"({sum(int((p[:, 2] * 1080 / f['height'] < DISTANT).sum()) for f, _, p in frames)} distant)")

    rows = []
    for name in args.detectors.split(","):
        for tiles in args.tiles.split(","):
            cols, rows_ = (int(v) for v in tiles.split("x"))
            det = detectors.load(name, threshold=args.threshold, tiles=(cols, rows_))
            det.detect(frames[0][1])  # warm up
            found = far_found = far_total = fps_ = 0
            times = []
            for k, (f, img, people) in enumerate(frames):
                t0 = time.perf_counter()
                boxes = det.detect(img)
                times.append(time.perf_counter() - t0)
                hit, fp = match(boxes, people, f.get("ignore", []))
                far = people[:, 2] * 1080 / f["height"] < DISTANT
                found += int(hit.sum())
                far_found += int(hit[far].sum())
                far_total += int(far.sum())
                fps_ += int(fp.sum())
                vis = img.copy()
                for b, is_fp in zip(boxes, fp, strict=True):
                    cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                                  (0, 0, 255) if is_fp else (0, 200, 0), 2)
                for (px, py, ph), ok in zip(people, hit, strict=True):
                    if not ok:
                        cv2.circle(vis, (int(px), int(py)), int(max(ph / 2, 8)), (255, 120, 0), 2)
                label(vis, f"{name} {tiles} thr {args.threshold}: {int(hit.sum())}/{len(people)} found, "
                           f"{int(fp.sum())} false", (12, 34))
                cv2.imwrite(str(args.out / f"{name}_{tiles}_frame{k}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            row = {"detector": name, "tiles": tiles, "threshold": args.threshold,
                   "recall": round(found / max(total, 1), 3), "found": found, "people": total,
                   "distant_recall": round(far_found / max(far_total, 1), 3), "distant_found": far_found,
                   "distant": far_total, "false_positives": fps_,
                   "false_per_frame": round(fps_ / len(frames), 2),
                   "ms_per_frame": round(1000 * float(np.mean(times)), 1)}
            rows.append(row)
            print(f"{name:12} {tiles:4} recall {row['recall']:.2f} ({found}/{total})  "
                  f"distant {row['distant_recall']:.2f} ({far_found}/{far_total})  "
                  f"false {fps_} ({row['false_per_frame']}/frame)  {row['ms_per_frame']:.0f} ms/frame")
    (args.out / "results.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out / 'results.json'} and overlays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
