# Watch the Goal

Turn your Sunday pickup footage into one highlight reel, buildup included.

Drop in a whole match filmed from a tripod, a few phone clips, or both. Watch
the Goal suggests the moments worth keeping and explains why ("crowd noise +14
dB for 2.5s", "motion near the marked left goal"). You keep or skip each one
and nudge its start back to catch the assist. Then export one MP4 for the
group chat.

It runs on your machine. No account, no upload, no API key, no GPU. Two
optional extras also stay local: a small player detector (downloaded on
request) and an AI review of the strongest suggestions by a vision model you
run with [Ollama](https://ollama.com).

![Reviewing a suggested goal](docs/screenshot.png)

## Quickstart

**Prerequisites:** Python 3.10+, [uv](https://docs.astral.sh/uv/), and FFmpeg
(which includes `ffprobe`).

```sh
# macOS
brew install ffmpeg uv
# Ubuntu / Debian
sudo apt install ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
winget install Gyan.FFmpeg astral-sh.uv
```

Then, from a clone of this repo:

```sh
uv run watch-the-goal
```

That opens `http://127.0.0.1:8765` in your browser. Projects, previews and
exported reels are kept in `~/WatchTheGoal` (change it with `--workspace`).

Without uv: `pip install .` then `watch-the-goal`.

### Try it with the demo footage

```sh
uv run python scripts/make_demo.py      # renders a synthetic match + 3 phone clips (~2 min)
uv run watch-the-goal demo/footage      # starts with all four imported
```

The demo footage is drawn by code, so it's safe to share. See
[demo/README.md](demo/README.md).

## Using it

1. **Import.** Click **Add videos**, drop files anywhere on the page, or paste a
   file or folder path. Pasted paths are read in place. Dropped files are
   copied into the workspace. Your originals are never modified. MP4, MOV, M4V,
   MKV and WebM work, including iPhone HEVC and rotated portrait video.
2. **Long recordings** (over 2 minutes) are analysed straight away and get
   suggested moments (amber on the timeline). For a fixed, wide camera, click
   **Mark goals** and draw a box around each goal mouth. Analysis then also
   looks for movement in front of the goals, and noise that comes with no
   goal-area action is ignored.
3. **Short clips** go straight into the reel. If a clip has a clear moment, a
   tighter trim is suggested; apply it with one click.
4. **Review.** Click a suggestion. The zoomed strip shows the clip with handles
   you can drag, a clickable **◆ Finish** timestamp, separate In/Out boundaries and a zoomed audio waveform.
   **⟵ 5s earlier** extends the start back to recover a turnover or an assist.

   | Key | Action |
   | --- | --- |
   | `J` / `L` | previous / next clip |
   | `K` / `X` | keep / skip (then move to the next suggestion) |
   | `Shift+←` / `Shift+→` | start 5 s earlier / end 5 s later |
   | `[` / `]` | set start / end at the playhead |
   | `N` | new clip around the playhead (for anything detection missed) |
   | `1`–`4` | your call on the selected clip: goal, save, near miss, not a highlight |
   | `G` | goal here: call a goal at the playhead (adds a clip if none covers it) |
   | `D` | live detection on / off |
   | `Space` | play / pause |
   | `Ctrl/⌘ Z` | undo a trim or reel reorder |

5. **See what it sees, and correct it.** **◉ Live detection** (`D`) overlays
   the player while you watch. It shows each signal at the playhead against
   the bar it has to clear ("crowd noise +4.2 dB · counts at +7.7 dB",
   "left goal 63% of the bar"). It says whether anything was heard or seen
   there, and which suggestion covers it. With a player detector downloaded
   (see Analysis), every frame you watch is also run through it on this
   machine (about 60–80 ms a frame on an M-series Mac). Players show as
   boxes, turning green near a marked goal. Ticks along each timeline track
   are every raw event the detector registered, suggested or not.

   Then make **calls**: press `1`–`4` on a suggestion (goal, save, near miss,
   not a highlight), or `G` on a goal it missed. Each call explains in numbers
   what the detector measured there and why it did or didn't suggest it
   ("crowd noise rose +9.6 dB; here it needs +10.8 dB"). A goal or save call
   keeps the clip, and "not a highlight" skips it. **Re-tune from my calls**
   on the recording's card then moves that recording's bars just far enough
   to agree with you. Candidates that resemble a call move up or down and
   say so ("like your goal at 46:50"). Calls download as `labels.json` for
   `scripts/eval_highlights.py`. Details: [docs/DETECTION.md](docs/DETECTION.md#6-your-calls).
6. **Arrange.** The Reel panel lists kept clips in play order. Drag them, or
   use ▲▼. The meter tracks the 30 to 60 second sweet spot. **Preview reel**
   plays everything back to back.
7. **Analysis (optional).** The **Analysis** button picks a player detector
   (*None*, *YOLOX* or *RF-DETR Nano*; each is a one-click download of 36 or
   108 MB) and turns on AI review. With a detector, suggestions also say things
   like "7 players near the left goal (usually 3)", and people blocking the
   lens no longer count as goal-mouth action. AI review sends a few
   timestamped frames from each of the strongest suggestions to a local vision
   model, shows its verdict ("AI: near miss · shot goes wide"), moves likely
   goals, saves and near misses up, and offers a trim when it sees more
   buildup or aftermath worth keeping. It never changes a clip you've touched, and if
   the model is missing, slow or talks nonsense, your suggestions stay as they
   were and the card says why.
8. **Make a match recap.** Choose **Match recap** in the Reel panel. Enter a
   title for a two-second opening card, or leave it blank to open on the
   action. Teams and scores are optional and belong to each recording.
   Choose **Portrait 9:16**, select a clip, and switch between **Fit full shot**
   and **Crop to fill**. Drag **Position** to follow the action in a wide
   shot. Framing is saved per clip. **Preview reel** plays the title and
   scoreboards over the kept clips in order.
9. **Ask for a trim.** With local AI review enabled in **Analysis**, select
   a clip and choose **Include more of the buildup** or **Cut sooner after
   the goal**. Compare the proposed range to the current one before applying.
   **Replay range** checks the cut. **Undo** restores the previous trim or
   order. The last 60 trim and reorder actions are available until the server restarts.
10. **Export.** Pick a shape (Auto, 16:9, 9:16 or 1:1) and click **Export MP4**.
   Every clip is rotated upright, scaled onto the frame over a blurred fill,
   converted to 30 fps with 48 kHz stereo AAC (clips with no audio get
   silence), and joined into one H.264 MP4 with fast start. Progress shows per
   clip. If something fails, the error says which clip failed and what to do.

### Command line

```sh
watch-the-goal [VIDEOS_OR_FOLDERS...] [--workspace DIR] [--port 8765] [--no-browser]
watch-the-goal detect match.mp4 [--goal x,y,w,h ...]   # print suggestions as JSON
    [--detector none|yolox|rfdetr-nano] [--debug-dir DIR]   # player evidence, overlays
    [--review [--review-model qwen3.5:9b] [--review-max 6]] # local AI review via Ollama
watch-the-goal models [list | download yolox rfdetr-nano]   # pinned, checksum-verified
```

## How detection works

The baseline detector streams each video through FFmpeg once and computes
loudness (every 0.25 s) and frame-to-frame motion (5 per second, for the whole
frame and inside each marked goal box). It looks for:

- sustained crowd noise well above the local level,
- motion in a marked goal area,
- play stopping after a burst of action.

Agreeing signals form one moment. Its clip starts 12 s early to keep the
buildup, or earlier (up to 20 s) when play was already surging at that
point, and runs 5 to 10 s past it. Overlapping suggestions from the same
recording are merged. Each suggestion is ranked; **strongest first** in the
timeline walks them in that order.

With a player detector, the strongest candidates are re-checked: people are
found in one frame per second around each moment, counted near the marked goal
and in the tightest bunch, and compared with what is usual for that recording.
That adds reasons, extends the clip to cover the reaction, and drops goal-mouth
"action" that was really someone standing in front of the lens. Details and
improvement ideas: [docs/DETECTION.md](docs/DETECTION.md).

## Detection quality

Measured on the synthetic demo match and on two real recordings (one hour of
Veo footage, outdoors; 51 minutes from a phone on a tripod in an indoor dome).
The full tables, labels, hardware and commands are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). In short:

- **Synthetic match** (3 goals): every configuration finds all 3 with buildup
  and reaction. It proves the mechanics, not real-world quality.
- **Real footage:** the one confirmed goal is found by every configuration
  (rank 2 to 3 of 31) but the suggested clip starts 4.5 s after the attack
  began. One **⟵ 5s earlier** click recovers it. Real-world recall is
  **unmeasured**: neither match is fully labelled.
- **Player detectors** see about a third (YOLOX) to two fifths (RF-DETR Nano)
  of the people in a 1080p frame, and very few distant players. Tiling helps
  recall but costs 3.5 to 7 times as much and adds false positives. On the
  dome recording with its goal marked they cut suggestions from 24 to 15 by
  vetoing people at the lens, and lift the one save from #19 to #2, for about
  1.7 s of extra processing per minute of footage.
- **AI review** (Qwen3.5 9B on an M4, about 30 s per suggestion) labelled 16
  of 17 hand-labelled real windows right and moved the goal and the save to
  #1, but called the goal a near miss, and its own trim would have cut both
  the buildup and the reaction. Trims are therefore only offered when they
  make a clip longer. On the drawn demo match it got most labels wrong.

The defaults stay **no detector, AI review off**: one goal and one save in two
recordings is not enough evidence to make everyone download a model. For a
fixed camera with the goals marked, turning on a detector is worth trying. Expect to scrub and add
clips with `N` as often as you accept suggestions. PRs with labelled matches
are very welcome.

## Known limitations

- By default suggestions come from loudness and motion only. The optional
  player detector finds people, not the ball or the net, and misses most
  distant players. Indoor domes, nearby games and people walking past the
  lens cause false positives. A quiet goal with no reaction is easy to miss.
- AI review is slow (about 30 s per suggestion on an Apple M4) and often
  wrong on wide, distant footage. Treat its labels as a second opinion.
- Goal boxes assume a fixed camera. If the tripod gets bumped, re-mark them.
- Multiple files are independent recordings. No automatic angle matching,
  syncing, or digital camera tracking yet. You order clips yourself.
- Preview plays the original file when the browser supports it. Other codecs
  (e.g. HEVC) get a 480p H.264 preview built in the background. Export always
  uses the original.
- Dropping a file into the browser copies it into the workspace. For huge
  match files, paste the path instead so it's read in place.
- Export re-encodes every clip with libx264 on the CPU (measured about 10x
  realtime for 1080p30 on an Apple-silicon Mac; slower machines will notice).
- One project per workspace. Use `--workspace` to keep several.

## Development

```sh
uv sync
uv run pytest -q
```

Tests build their own tiny fixture videos with FFmpeg: multi-file import,
timestamp boundaries (frames encode their own timestamp as brightness, so
cuts are checked to the frame), merging overlapping suggestions, and a real MP4
export mixing a rotated portrait HEVC MOV, a 60 fps clip and a clip with no
audio. CI runs them on Ubuntu and macOS, renders the demo, prints the
detection report, exports the demo reel (`scripts/demo_reel.py`), and drives
the real UI in a browser with Playwright (`scripts/e2e_ui.py`): import by path
and upload, suggestions, trimming, reordering, AI review failing safely, and
export, with the MP4 checked by `scripts/check_reel.py`.

Agent skills for this repo (for Codex in `.agents/skills/`, for Claude Code in
`.claude/skills/`, kept identical by a test): `soccer-vision`,
`ffmpeg-essentials`, `video-qa` and `webapp-testing`.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE). Third-party packages, model checkpoints and adapted skill
material are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
