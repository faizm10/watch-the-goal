# Benchmarks

Measured on 2026-11-25. Two separate questions, always on the same footage:

1. **Person detection.** Do the detectors see the players?
2. **Highlights.** Are the suggested clips better, with AI review off and on?

Synthetic and real results are reported separately. The synthetic match proves
the mechanics. Only real footage says anything about quality, and the real
footage here is **not fully labelled**, so **real-world recall is unmeasured**.

## Setup

| | |
| --- | --- |
| Machine | Apple M4 (10-core CPU), 16 GB RAM, macOS 27.0.1 |
| Software | Python 3.15.2, FFmpeg 8.0.1, OpenCV 5.1.0 (`cv2.dnn`, CPU), Ollama 0.40.18 |
| Detectors | `none` (baseline: motion and audio only); `yolox` (YOLOX-S, OpenCV Zoo, 640×640); `rfdetr-nano` (RF-DETR Nano, 384×384). Person class only, threshold 0.3. Checkpoints and checksums in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) |
| AI reviewer | `qwen3.5:9b` via Ollama (Apache-2.0, digest `d63f830f28d5`), 12 frames per suggestion, 6 suggestions per recording, temperature 0 |

### Footage

| Name | What | Labels |
| --- | --- | --- |
| Synthetic match | `scripts/make_demo.py`: 6 min, 1280×720, drawn by code, 3 goals, a staged argument, a near miss. Shareable | Fully labelled: goal times, buildup start (turnover, 9 s before), reaction end |
| Veo | 57 min, 1920×1080, Veo auto-panning camera, outdoors, 11-a-side amateur match | 1 confirmed goal (46:50.5; attack starts 46:34, reaction to 46:54). 15 suggestion windows hand-labelled from contact sheets (4 "other", 7 unclear). Not fully labelled |
| Dome | 51 min, 1920×1080, phone on a tripod at the corner of an indoor multi-pitch dome; left goal 0,0.42,0.25,0.24 | No goals found by eye. 17 suggestion windows hand-labelled (1 save: keeper dive at 20:16; 14 "other": spectators, sideline gatherings, people at the lens; 2 unclear). Not fully labelled |

The real recordings show identifiable people who have not cleared them for
redistribution. They, their labels and every output below stay on the
machine that ran the benchmark (`bench/`, `demo/real/`, all git-ignored).

## 1. Person detection

126 people hand-labelled in 4 frames (Veo at 46:48 and 20:50, dome at 15:00
and 20:19), 89 of them "distant" (under 60 px tall in 1080p). Labelling was
assisted: every detector, tiled 3×2 at threshold 0.15, proposed boxes; a person
accepted or rejected each and added the people all of them missed. Dense crowds
that couldn't be resolved by eye are "ignore" regions (detections there count
neither way). This favours what the detectors can see, so recall is, if
anything, overstated.

The baseline (`none`) detects no people, so it has no row here.

| Detector | Tiles | Recall | Distant recall | False positives / frame | ms / frame |
| --- | --- | --- | --- | --- | --- |
| YOLOX-S | 1×1 (app default) | 0.31 (39/126) | 0.07 (6/89) | 0.25 | 79 |
| YOLOX-S | 2×2 | 0.51 | 0.30 | 3.5 | 381 |
| YOLOX-S | 3×2 | 0.63 | 0.49 | 3.75 | 541 |
| RF-DETR Nano | 1×1 (app default) | 0.38 (48/126) | 0.16 (14/89) | 0.25 | 140 |
| RF-DETR Nano | 2×2 | 0.62 | 0.46 | 5.25 | 504 |
| RF-DETR Nano | 3×2 | 0.74 | 0.63 | 6.5 | 797 |

In the overlays, most "false positives" when tiled are partial boxes (a head,
a pair of legs) on a person cut by a tile edge, on top of the full box. RF-DETR Nano sees more people
than YOLOX at every setting for 1.3 to 1.8 times the time. Neither sees
most distant players without tiling, and tiling costs 3.5 to 7 times as much.
The app keeps 1×1: its player evidence looks for crowding near a goal and
near the lens, where people are large.

```sh
uv run python scripts/label_people.py propose worker-fc-vs-naanstop.mp4:2808 worker-fc-vs-naanstop.mp4:1250 soccer-clip.mp4:900 soccer-clip.mp4:1219 --out bench/people
uv run python scripts/label_people.py finalize bench/people/draft.json bench/people/decisions.json --out bench/people/people.labels.json
uv run python scripts/bench_detectors.py bench/people/people.labels.json --tiles 1x1,2x2,3x2 --out bench/detectors
```

## 2. Highlights

Scored by `scripts/eval_highlights.py`. A goal is **found** if a suggestion
contains the finish; **buildup + reaction** means that clip also starts at or
before the labelled buildup start and ends at or after the reaction end.
**False** = suggestions with no goal in them (on real footage, split into
labelled kinds and unlabelled). Time is processing per minute of footage
(signals + detector; review separately). "AI trims" scores the clips as if
every proposed AI trim were applied.

### Synthetic match

| Detector | Goal boxes | Suggestions | Goals found | Buildup + reaction | Missed | False | s / min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | no | 4 | 3/3 | 3 | 0 | 1 (the argument) | 0.25 |
| YOLOX | no | 4 | 3/3 | 3 | 0 | 1 | 1.74 |
| RF-DETR Nano | no | 4 | 3/3 | 3 | 0 | 1 | 2.04 |
| none | yes | 3 | 3/3 | 3 | 0 | 0 | 0.26 |
| YOLOX | yes | 3 | 3/3 | 3 | 0 | 0 | 1.74 |
| RF-DETR Nano | yes | 3 | 3/3 | 3 | 0 | 0 | 2.00 |

The detectors find nobody in the drawn players (stick figures are not COCO
people), so they change nothing here; that is expected.

With AI review (same verdicts for every detector, since the suggestions are
identical):

| Goal boxes | Goal ranks before → after review | AI labels right | AI trims keep buildup + reaction: as offered (raw) | Review s / min |
| --- | --- | --- | --- | --- |
| no | 2, 3, 1 → 4, 2, 1 | 0 of 4 | 3 of 3 (2 of 3) | 21.5 |
| yes | 2, 3, 1 → 2, 3, 1 | 1 of 3 | 3 of 3 (2 of 3) | 15.4 |

On drawn footage the reviewer called goals "other", "save" and "near miss",
and reranking pushed one goal from #2 to #4. Its raw trims started earlier
(good) but one ended 0.6 s before the labelled reaction end; the app now only
offers the parts of a trim that lengthen the clip ("as offered").

```sh
uv run python scripts/make_demo.py
uv run python scripts/eval_highlights.py demo/footage/labels.json demo/footage/labels_goal_areas.json --review --out bench/synthetic
```

### Real footage

Without AI review. "Save" is the keeper dive at 20:16 in the dome; it counts
as a false suggestion (no goal) but is a highlight worth ranking high.

| Recording | Detector | Goal boxes | Suggestions | Goal found (rank) | Buildup + reaction | Save rank | Other / unlabelled | s / min (detector) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Veo | none | no | 31 | yes (#3) | no, 4.5 s late | | 4 / 26 | 0.82 |
| Veo | YOLOX | no | 31 | yes (#2) | no | | 4 / 26 | 1.84 (1.02) |
| Veo | RF-DETR Nano | no | 31 | yes (#3) | no | | 4 / 26 | 1.67 (0.85) |
| Dome | none | no | 3 | | | | 3 / 0 | 0.85 |
| Dome | YOLOX | no | 3 | | | | 3 / 0 | 1.10 (0.25) |
| Dome | RF-DETR Nano | no | 3 | | | | 3 / 0 | 1.14 (0.29) |
| Dome | none | yes | 24 | | | #19 | 9 / 14 | 0.82 |
| Dome | YOLOX | yes | 15 | | | **#2** | 5 / 9 | 2.40 (1.59) |
| Dome | RF-DETR Nano | yes | 15 | | | **#2** | 5 / 9 | 2.53 (1.71) |

No goal was missed (there was one to find) and none was added. With the goal
marked, both detectors dropped 9 of the dome's 24 suggestions (the lens veto,
66 to 67 candidates in the wider pool) and moved the save from #19 to #2. On
the Veo match they only add "N players bunched together" reasons. Times were
measured on an otherwise idle machine; a second run while a browser test was
running took up to 1.6 times as long.

With AI review of the 6 strongest suggestions per recording:

| Recording | Detector, goal boxes | Highlight rank before → after | AI labels right (labelled windows) | Review s / min |
| --- | --- | --- | --- | --- |
| Veo | none / YOLOX / RF-DETR | goal #3 / #2 / #3 → **#1** | 1 of 2, 1 of 2, 2 of 3 | 3.1 |
| Dome | any, no boxes | (none) | 3 of 3 | 2.0 |
| Dome | none, boxes | save #19, not reviewed | 6 of 6 | 4.2 |
| Dome | YOLOX / RF-DETR, boxes | save #2 → **#1** | 5 of 5 | 3.8 |

Over the 17 distinct labelled windows it saw, the reviewer was right on 16:
every "other" (spectators, gatherings, people at the lens, ordinary play)
and the save. It called the one goal a "near miss" ("a player in blue
dribbles past an orange defender and takes a shot that goes wide"), which
still moved it to #1, because near misses rank with saves above everything
unreviewed. Each suggestion took 28 to 40 s. "Other" is the easy majority
here, so 16 of 17 says more about telling play from non-play than about
spotting goals.

```sh
uv run watch-the-goal models download yolox rfdetr-nano
ollama pull qwen3.5:9b
uv run python scripts/eval_highlights.py demo/real/worker_fc.labels.json demo/real/soccer_clip.labels.json \
  demo/real/soccer_clip_goal.labels.json --review --out bench/real
```

### Before and after on the real goal

The one confirmed goal (Veo, 46:50.5). The attack starts at 46:34 when the
ball is played forward down the left; the finish is at 46:50.5; the reaction
lasts to about 46:54.

| Cut | Range | Keeps the attack from 46:34 | Finish | Reaction |
| --- | --- | --- | --- | --- |
| Baseline suggestion (same with either detector) | 46:38.5 to 47:02.8 | no, starts 4.5 s late | yes | yes |
| AI review's raw proposal (no longer offered as is) | 46:39.3 to 46:51.0 | no, 5.3 s late | yes, by 0.5 s | no |
| Baseline after one **⟵ 5s earlier** | 46:33.5 to 47:02.8 | yes | yes | yes |

The new surge-based lead-in does not extend this clip: motion dips at 46:26
and 46:38 when the camera reverses its pan, so the signals can't see that the
attack started earlier. Fitting the rule to this one goal would be
overfitting. The contact sheet and the three cuts are saved locally in
`bench/before_after/`.

### Celebration check (2026-11-26)

AI review's second question ("are players celebrating a goal?", see
DETECTION.md §5). Veo match, baseline detector, the 10 strongest suggestions,
labels from the calls in DETECTION.md §6 (goals 46:50 and 56:12, chance
51:35, tackles 15:16 and 42:50, plus the 4 "other" windows):

| | Labels right | Goal ranks after review | Non-goals called a celebration | Review s / min |
| --- | --- | --- | --- | --- |
| First look only | 2 of 6 | #1, #3 | – | 5.8 |
| With the celebration check | 3 of 6 | #1, #3 | 0 of 8 | 11.8 |

The first look called both goals near misses. The check found the
celebration after 46:50 ("players in orange jerseys running towards the
corner and raising their arms") and missed the one after 56:12. Recall
depends on the exact frames: over four samplings each goal was found 3 and 1
times. No sampling called a non-goal a celebration (0 of 32), and on the drawn
match it said "no" every time, so it never changed a rank there. Separately,
the first look's labels on the drawn match differed between two runs with
the same prompt and temperature 0, so treat single runs of the reviewer as
noisy.

```sh
uv run python scripts/eval_highlights.py path/to/veo.labels.json demo/footage/labels.json \
  --detectors none --review --review-max 10 --out bench/celebration
```

## 3. Tracking (ByteTrack)

`scripts/eval_tracking.py` with RF-DETR Nano at 10 fps and ByteTrack from
`supervision` 0.26.1 (throwaway environment), on the 8 s windows the app
measures: dome 20:16 (save), 23:05 and 24:02 (sideline gatherings), 6:48
(person at the lens); Veo 46:48 (goal), 42:51 and 16:36.

| | Without tracking (1 fps, app) | With ByteTrack (10 fps) |
| --- | --- | --- |
| Detector time per window | 0.9 to 1.6 s | 7.8 to 14.2 s |
| Peak players near the dome goal: save / gatherings / lens | 8 / 8, 10 / 7 | 7 / 8, 9 / 5 |
| Peak bunch, Veo goal / two non-goals | 9 / 9, 8 | 7 / 7, 6 |
| Track ids per person in 8 s | n/a | 2.3 to 3.5 |

Tracking lowers every count a little (it drops one-frame detections) but
doesn't separate the save or the goal from the non-highlights any better,
fragments identities 2 to 3.5 times over in distant crowds, and costs about 9
times the detector time. **Not integrated.**

```sh
uv run --no-project --with supervision==0.26.1 --with numpy python scripts/eval_tracking.py \
  'soccer-clip.mp4:1216:{"x":0,"y":0.42,"w":0.25,"h":0.24}' 'soccer-clip.mp4:1385:{"x":0,"y":0.42,"w":0.25,"h":0.24}' \
  'soccer-clip.mp4:1442:{"x":0,"y":0.42,"w":0.25,"h":0.24}' 'soccer-clip.mp4:408:{"x":0,"y":0.42,"w":0.25,"h":0.24}' \
  worker-fc-vs-naanstop.mp4:2808 worker-fc-vs-naanstop.mp4:2571 worker-fc-vs-naanstop.mp4:996 --out bench/tracking
```

## 4. End to end through the UI

`scripts/e2e_ui.py` (Playwright 1.65.2, Chrome) imported the 57-minute Veo
match and one phone clip by folder path, uploaded two more phone clips,
analysed, opened the goal suggestion, trimmed it 5 s earlier and back, kept
it and the three phone clips, moved the first reel item down, and exported a
70.4 s 1920×1080 reel. `scripts/check_reel.py` then confirmed it decodes
cleanly, is H.264/AAC with faststart, matches the clips' total length
(70.40 vs 70.36 s), audio and video lengths agree within 0.07 s, and 7 of 8
clip edges match their sources within 0.07 s (the eighth, a near-still phone
shot, could not be judged). The goal clip plays at 0:32.1 to 0:56.4.

The same run with RF-DETR Nano and AI review on (3 suggestions reviewed, 231 s
from import to review done, of which analysis 137 s) passed all 11 checks;
the goal suggestion showed "9 players bunched together (usually 6)" and the
AI's verdict, and its reel passed `check_reel.py`. Its screenshot
(`bench/e2e/review/3-editor.png`) is what prompted the extend-only rule: the
run predates it and offered the shrinking 46:39.3 to 46:51.0 trim. A run on
the synthetic match with the reviewer pointed at a closed port passed 11 of
11: suggestions unchanged, the source card saying "AI review unavailable,
suggestions unchanged", export fine.

```sh
uv run --group e2e python scripts/e2e_ui.py worker-fc-vs-naanstop.mp4 demo/real/phone_corner_portrait.mov \
  demo/real/phone_dome_attack.mp4 demo/real/phone_dome_noaudio.mp4 --goal 2810.5 --channel chrome --out bench/e2e/baseline
uv run --group e2e python scripts/e2e_ui.py worker-fc-vs-naanstop.mp4 demo/real/phone_corner_portrait.mov \
  demo/real/phone_dome_attack.mp4 --goal 2810.5 --detector rfdetr-nano --review ok --review-max 3 --channel chrome --out bench/e2e/review
uv run --group e2e python scripts/e2e_ui.py demo/footage/sunday_match_wide.mp4 demo/footage/phone_goal_portrait.mov \
  --goal 70 --review fail --channel chrome --out bench/e2e/synthetic-review-fail
```

## Decision

Defaults stay **`detector: none`** and **AI review off**.

- **Detectors.** No goal was found or lost by turning one on. With a fixed
  camera and the goal marked, both cut false suggestions by a third and lifted
  the one save from #19 to #2, at about 1.7 s per minute of footage plus a 36
  or 108 MB download. That is one save in one recording: promising for fixed
  cameras, not enough to make every user download a model. RF-DETR Nano sees
  more people; YOLOX is faster. Either is a reasonable opt-in.
- **AI review.** Its ranking helped on real footage (goal and save to #1) and
  hurt on drawn footage (a goal to #4). Its labels are good at "this isn't a
  highlight" and unreliable on goals; the celebration check now recognises
  some goals and has never invented one. It needs an 8 GB model and about a
  minute per suggestion. Opt-in, shown as a second opinion, never overriding the
  user, and now only allowed to lengthen clips.
- **ByteTrack.** Not integrated (section 3).
- **Buildup.** The one real goal still starts 4.5 s late with every
  configuration. One click fixes it; the signals alone could not.
