---
name: soccer-vision
description: Label soccer footage, compare the person detectors (none, YOLOX, RF-DETR Nano), save debug overlays, and evaluate highlight suggestions with and without local AI review in Watch the Goal. Use when changing detection or ranking (watch_the_goal/detect.py, pipeline.py, vision/, review.py), adding a detector or reviewer model, or when asked whether a change finds more goals or keeps more buildup.
---

# Soccer vision

Two questions, always answered separately and on the same labelled footage:

1. **Does the person detector see the players?** Recall (overall and for distant players), false positives, ms per frame.
2. **Are the suggested highlights better?** Goals found, missed goals, false suggestions, buildup and reaction kept, seconds of processing per minute of footage, with AI review off and on.

Report synthetic and real footage separately. The synthetic match proves the mechanics, not real-world quality. Never commit footage, labels of private footage, or `bench/` output (all git-ignored).

## Setup

```bash
uv sync
uv run watch-the-goal models list                        # what is downloaded, licence, size
uv run watch-the-goal models download yolox rfdetr-nano  # pinned URLs, SHA-256 checked, ~/.cache/watch-the-goal/models
uv run python scripts/make_demo.py                       # synthetic match + labels in demo/footage/ (~2 min)
```

AI review needs [Ollama](https://ollama.com) on this machine and a vision model with open weights, for example `ollama pull qwen3.5:9b` (Apache-2.0). The app only talks to loopback endpoints.

## Look at one recording

```bash
# Baseline (motion + audio), then with a detector; overlays and candidates.json in --debug-dir
uv run watch-the-goal detect match.mp4 --debug-dir bench/debug/match-none
uv run watch-the-goal detect match.mp4 --detector rfdetr-nano --debug-dir bench/debug/match-rfdetr
# Fixed camera: mark the goal (normalised x,y,w,h) so "players near the goal" can be measured
uv run watch-the-goal detect match.mp4 --goal 0,0.42,0.25,0.24 --detector yolox --debug-dir bench/debug/match-yolox
# Add AI review of the strongest suggestions
uv run watch-the-goal detect match.mp4 --review --review-model qwen3.5:9b --review-max 6
```

Open the `rankNN_*.jpg` overlays: boxes are detected people, the inner rectangle is the goal, the outer one is the margin counted as "near". Check that the reasons printed for each suggestion match what the frame shows.

## Label highlights

One JSON per recording, next to (or pointing at) the video. Paths are relative to the JSON file:

```json
{
  "video": "../../match.mp4",
  "goal_moments": [{"t": 2810.5, "buildup_start": 2794, "reaction_end": 2814}],
  "goal_areas": [{"x": 0, "y": 0.42, "w": 0.25, "h": 0.24}],
  "use_goal_areas": false,
  "fully_labeled": false,
  "moments": [{"t": 1216, "label": "save"}, {"t": 293, "label": "other"}],
  "unclear": [923, 996]
}
```

- `t` is the moment the ball crosses the line; `buildup_start` is where the attack that led to it starts (the turnover or first pass you'd want to see); `reaction_end` is when the celebration or reaction has registered.
- `fully_labeled: true` only if every goal in the file is listed; otherwise unlabelled suggestions count as "unlabelled", not false.
- Label suggestion windows quickly from contact sheets: `ffmpeg -ss START -to END -i match.mp4 -vf "fps=1/2,scale=480:-1,tile=4x2" -frames:v 1 sheet.jpg`. Leave anything you can't call from the frames in `unclear`.

## Evaluate highlights

```bash
uv run python scripts/eval_highlights.py demo/footage/labels.json demo/footage/labels_goal_areas.json --out bench/synthetic
uv run python scripts/eval_highlights.py path/to/*.labels.json --out bench/real
# AI review on and off in one run (verdicts cached in --out/review_cache.json, so configs see identical answers)
uv run python scripts/eval_highlights.py path/to/*.labels.json --review --review-max 6 --out bench/real
```

Prints one block per recording × detector; `results.json` holds every suggestion with its truth label, and `debug/` the overlays. Signals are cached in `--out/signals/`; delete it after changing `signals.py`.

## Compare person detectors

```bash
# 1. Proposals from every detector (tiled, low threshold), numbered on zoomed crops with a pixel grid
uv run python scripts/label_people.py propose match.mp4:2808 match.mp4:1250 other.mp4:900 --out bench/people
# 2. Write bench/people/decisions.json: per frame, proposal numbers that are not people (or duplicates),
#    people every detector missed as [cx, cy, height], and "ignore" boxes for crowds you can't resolve
# 3. Finalize and benchmark
uv run python scripts/label_people.py finalize bench/people/draft.json bench/people/decisions.json --out bench/people/people.labels.json
uv run python scripts/bench_detectors.py bench/people/people.labels.json --tiles 1x1,2x2,3x2 --out bench/detectors
```

"Distant" = under 60 px tall in a 1080-line frame. Proposals come from the detectors themselves, so ground truth is biased toward what they can see: add the misses by hand and say so when you report.

## Tracking (ByteTrack)

Not a dependency of the app. Evaluate in a throwaway environment:

```bash
uv run --no-project --with supervision==0.26.1 --with numpy python scripts/eval_tracking.py \
  'match.mp4:1216:{"x":0,"y":0.42,"w":0.25,"h":0.24}' match.mp4:2808 --detector rfdetr-nano --fps 10
```

Integrate only if tracked counts change which candidates clear the ranking thresholds for a cost the app can bear (see docs/BENCHMARKS.md for the last measurement).

## Rules

- The default stays `detector: "none"` and AI review off unless a benchmark on real labelled footage shows a clear gain. Write the numbers into `docs/BENCHMARKS.md`.
- Suggestions explain themselves with reasons ("7 players near the left goal (usually 3)"), never confidence scores.
- AI review may rerank and propose trims; it never overwrites a clip the user touched, and any failure leaves the detector's suggestions as they were.
- New model files: pin URL, SHA-256, size and licence in `watch_the_goal/vision/models.py` and `THIRD_PARTY_NOTICES.md`.

## Selected-clip requests

Run `uv run pytest -q tests/test_review.py` for validation and stale trim coverage. With the local vision model enabled, request more buildup, compare both timestamps and replay after applying. Undo, then try an earlier ending. Changing a manual trim while waiting must invalidate the proposal. Record model and source when reporting accuracy.
