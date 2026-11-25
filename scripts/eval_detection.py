"""Score the detector against hand-labelled goal times.

    python scripts/eval_detection.py VIDEO LABELS.json [--use-goal-areas]

LABELS.json needs at least {"goals": [seconds, ...]}. Optionally
"goal_areas": [{x, y, w, h}, ...] (fractions of the frame, as the UI stores
them) and "not_goals": {...} for notes.

A goal counts as found when a suggested clip contains it with at least
``--min-buildup`` seconds before it (so the assist would be in the clip).
Every suggestion that contains no labelled goal is a false positive.

To label your own match: scrub through it and write down the second each
goal crosses the line. That's the only thing this script needs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watch_the_goal import detect, signals  # noqa: E402
from watch_the_goal.media import probe  # noqa: E402


def score(moments: list[detect.Moment], goals: list[float], min_buildup: float) -> dict:
    found, missed, short_buildup = [], [], []
    used = set()
    for g in goals:
        hit = next((i for i, m in enumerate(moments) if m.start <= g <= m.end), None)
        if hit is None:
            missed.append(g)
        elif g - moments[hit].start < min_buildup:
            short_buildup.append(g)
            used.add(hit)
        else:
            found.append(g)
            used.add(hit)
    false_pos = [moments[i] for i in range(len(moments)) if i not in used]
    return {"found": found, "missed": missed, "found_without_buildup": short_buildup,
            "false_positives": [(round(m.start, 1), round(m.end, 1), m.reasons) for m in false_pos],
            "suggestions": len(moments)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("labels")
    ap.add_argument("--use-goal-areas", action="store_true", help="use goal_areas from the labels file")
    ap.add_argument("--min-buildup", type=float, default=6.0)
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text())
    areas = labels.get("goal_areas", []) if args.use_goal_areas else []
    info = probe(args.video)
    t0 = time.time()
    sig = signals.extract(info.path, info.width, info.height, info.duration, info.has_audio, areas)
    moments = detect.find_moments(sig, areas)
    elapsed = time.time() - t0
    result = score(moments, labels["goals"], args.min_buildup)

    print(f"video:        {Path(args.video).name} ({info.duration / 60:.1f} min, {info.width}x{info.height}, "
          f"{'audio' if info.has_audio else 'no audio'})")
    print(f"goal areas:   {'yes' if areas else 'no'}")
    print(f"analysis:     {elapsed:.1f}s ({info.duration / max(elapsed, 1e-6):.0f}x realtime)")
    print(f"goals:        {len(labels['goals'])} labelled")
    print(f"found:        {len(result['found'])} {result['found']}")
    print(f"no buildup:   {len(result['found_without_buildup'])} {result['found_without_buildup']}")
    print(f"missed:       {len(result['missed'])} {result['missed']}")
    print(f"false pos.:   {len(result['false_positives'])}")
    for start, end, reasons in result["false_positives"]:
        print(f"              {start}-{end}s: {'; '.join(reasons)}")
    print(f"suggestions:  {result['suggestions']} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
