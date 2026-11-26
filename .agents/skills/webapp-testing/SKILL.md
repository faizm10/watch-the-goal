---
name: webapp-testing
description: Test the Watch the Goal web UI in a real browser with Playwright - multi-file import, suggestions and their reasons, trimming, keeping, reordering, AI review and its fallback, and export - using scripts/e2e_ui.py or a short custom script. Use after changing watch_the_goal/static/, server.py or app.py, or when asked to check that the app works for a person, not just in unit tests.
---

# Web app testing (Watch the Goal)

Unit tests (`uv run pytest -q`) cover the API and the pipeline. This skill drives the actual page the way a person would.

## Setup

```bash
uv sync --group e2e
uv run --group e2e playwright install chromium   # or pass --channel chrome to use an installed Chrome
```

## The end-to-end script

```bash
# Long recording + phone clips, baseline (no models), goal at 2810.5 s must land inside a kept suggestion
uv run --group e2e python scripts/e2e_ui.py match.mp4 phone1.mov phone2.mp4 phone3.mp4 \
  --goal 2810.5 --review off --out bench/e2e/baseline
# With a player detector and AI review through a local Ollama
uv run --group e2e python scripts/e2e_ui.py match.mp4 phone1.mov --detector rfdetr-nano --review ok --out bench/e2e/review
# Reviewer unreachable: suggestions must be unchanged and the card must say so
uv run --group e2e python scripts/e2e_ui.py match.mp4 phone1.mov --review fail --out bench/e2e/review-fail
# Fixed camera: mark the goal first (normalised x,y,w,h)
uv run --group e2e python scripts/e2e_ui.py dome.mp4 phone1.mov --goal-area 0,0.42,0.25,0.24 --out bench/e2e/dome
```

It starts the app on a free port with a throwaway workspace, sets the Analysis dialog, imports one file by folder path and the rest through the file picker, waits for analysis, opens the goal suggestion, jumps to the finish, undoes a quick trim, drags the start earlier, keeps it and every phone clip, moves the first reel item down and undoes the reorder, exports, and runs `scripts/check_reel.py` on the MP4 (see the `video-qa` skill). Screenshots `1-settings.png` … `5-exported.png`, the final project state and `reel.mp4` land in `--out`. Exit code 1 on any failed check or JavaScript error.

Synthetic footage for a quick run: `uv run python scripts/make_demo.py`, then use `demo/footage/sunday_match_wide.mp4` with `--goal 70` and the `phone_*` clips.

## Writing a new check

Start the server yourself and script against it:

```bash
uv run watch-the-goal --workspace /tmp/wtg-ui --port 8799 --no-browser &
```

```python
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport={"width": 1440, "height": 900})
    page.goto("http://127.0.0.1:8799")
    page.wait_for_load_state("networkidle")       # the page renders from /api/state
    page.screenshot(path="bench/inspect.png", full_page=True)
    state = page.request.get("http://127.0.0.1:8799/api/state").json()   # ground truth for assertions
```

Look at a screenshot before choosing selectors. Assert on `/api/state`, not on text that may be reworded.

| Thing | Selector |
|---|---|
| Import by path | `#path-input` then Enter (a file or a folder) |
| Upload files | `#file-input` (`set_input_files`, multiple) |
| Source card, its buttons | `.source[data-src=ID]`, `[data-act=analyze]`, `[data-act=review]`, `[data-act=goals]` |
| Clip on the timeline | `.block[data-clip=ID]` |
| Editor buttons | `[data-act=keep]`, `[data-act=skip]`, `[data-act=start-5]`, `[data-act=start+1]`, `[data-act=end+5]`, `[data-act=apply-trim]` |
| Reasons, AI verdict | `#editor .reason`, `#editor .reason.ai`, `#editor .reason.ai.fail` |
| Reel item, reorder | `#reel-list .reel-item[data-clip=ID]`, `[data-move="-1"]`, `[data-move="1"]` |
| Reel item, edit / remove | `[data-reel="edit"]`, `[data-reel="remove"]` (back to a suggestion) |
| Add a clip at the playhead | `#reel-add-btn` |
| Suggestions not in the reel | `#suggest-list .suggest-item[data-clip=ID]`, its `[data-add]`; `#add-all-btn` |
| Analysis settings | `#settings-btn`, `#detector-select`, `#review-enabled`, `#review-endpoint`, `#review-check`, `#settings-save` |
| Export | `#layout-select`, `#export-btn`, `#export-status` |

Keyboard shortcuts work too: `J`/`L` previous/next suggestion, `K` keep, `X` skip, `[`/`]` set start/end at the playhead.

## Rules

- Never point the app at a workspace you care about; use a temp folder.
- Real footage stays local: keep `--out` under `bench/` (git-ignored).
- Wait on `/api/state` jobs, not fixed sleeps, for analysis, review and export.

## Recap and trim regression

Generate the synthetic fixtures, then run both shapes in separate output folders.

```sh
uv run python scripts/make_demo.py --quick
for shape in landscape portrait; do
  uv run --group e2e python scripts/e2e_ui.py demo/footage/sunday_match_wide.mp4 demo/footage/phone* --goal 70 --recap --layout "$shape" --out "bench/recap-$shape"
done
```

The browser jumps to the detected finish, extends the start with both a button and a handle, undoes a trim and reorder, sets a scoreboard on only the match recording, and plays the exported file. The portrait run uses an off-centre crop on the goal and full-shot fit on the phones. Inspect `framing.png`, `goal-preview.png` and `goal-export.png` together. Compare the title pair, listen at each join and review notes in `check_reel.json`.
