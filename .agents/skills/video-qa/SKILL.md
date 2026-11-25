---
name: video-qa
description: Verify an exported Watch the Goal reel - duration, clip order, clip boundaries, playback and audio sync - with scripts/check_reel.py, ffprobe and a contact sheet. Use after any export, after changing watch_the_goal/export.py, or when someone says the reel is out of order, cuts off the buildup, plays wrong or drifts out of sync.
---

# Video QA (Watch the Goal reels)

An exit code of 0 from FFmpeg is not proof. Check the file against the project it came from, then look at it.

## 1. Automated checks

```bash
uv run python scripts/check_reel.py path/to/reel.mp4 --workspace ~/WatchTheGoal --json bench/qa.json
```

`--workspace` is the app workspace the reel was exported from (its `project.json` gives the clips in reel order). The script checks:

- **plays**: FFmpeg decodes every frame without an error
- **format**: H.264 `yuv420p` + AAC, `moov` before `mdat` (faststart, so browsers and phones start playing at once)
- **duration**: reel length equals the sum of the clips plus the optional two-second opening title (± 0.15 s)
- **audio sync**: audio and video streams are the same length (± 0.1 s), so nothing drifts at the joins
- **order and boundaries**: for each clip, frames 0.25 s inside its start and end are matched against the source around the expected time; the best match must be within 0.15 s. Still shots where every frame looks alike are reported as "not judged" instead of passing.

It also notes black, frozen and silent stretches. Silence is expected for phone clips recorded without sound; anywhere else, look.

## 2. Look at it

```bash
# One frame every 2 s, 8 per row
ffmpeg -v error -y -i reel.mp4 -vf "fps=1/2,scale=320:-1,tile=8x6" -frames:v 1 bench/qa-sheet.jpg
# Frames either side of each join (times from check_reel's output, e.g. a join at 21.5 s)
ffmpeg -v error -y -ss 21.3 -i reel.mp4 -frames:v 1 bench/join-before.jpg
ffmpeg -v error -y -ss 21.6 -i reel.mp4 -frames:v 1 bench/join-after.jpg
```

For a goal clip, confirm by eye that it starts before the pass that set it up (the assist) and runs past the finish into the reaction. That's the point of the app.

## 3. Play it

```bash
open reel.mp4                       # macOS: QuickTime
ffplay -autoexit reel.mp4           # anywhere with FFmpeg's player
```

Or in the app: **Preview reel** plays the kept clips in order from the sources; the exported MP4 plays from the Exports list.

## 4. Manual ffprobe

```bash
ffprobe -v error -show_entries format=duration:stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,duration -of default=nw=1 reel.mp4
ffmpeg -v error -i reel.mp4 -f null -                       # any output here is a decode error
ffmpeg -i reel.mp4 -af silencedetect=noise=-50dB:d=1.5 -vn -f null - 2>&1 | grep silence_
ffmpeg -i reel.mp4 -vf blackdetect=d=0.5:pix_th=0.10 -an -f null - 2>&1 | grep blackdetect
```

## Report

Say: path, duration, size, resolution, fps, codecs, clip count and order, each clip's reel time range, and anything not judged or not checked (subjective pacing, whether everyone in the footage agreed to sharing). If a check fails, fix and re-run before handing the reel over.

## Titles, framing and audio

Use the two `--recap --layout` runs in webapp-testing. The checker accounts for the title duration, compares source frames inside the chosen crop, checks rotation, 30 fps and stereo 48 kHz AAC, and compares a short decoded audio sample from each clip with its source. Repetitive or quiet audio may leave sync unjudged. A still title, silent phone clip or stationary camera can trigger a review note. Inspect flagged timestamps in the MP4 before deciding whether to change a cut. For a saved project with accessible sources use:

```sh
uv run python scripts/check_reel.py path/to/reel.mp4 --workspace path/to/workspace --json bench/qa.json
ffplay -autoexit path/to/reel.mp4
```
