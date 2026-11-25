# Third-party notices

Watch the Goal is MIT-licensed (see [LICENSE](LICENSE)). It uses, downloads on
request, or adapts the following. Versions are the ones tested; `uv.lock` pins
the Python packages.

## Python packages

| Package | Version tested | License | Used for |
| --- | --- | --- | --- |
| [numpy](https://numpy.org) | 2.5.5 | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | signals, detection |
| [opencv-python-headless](https://github.com/opencv/opencv-python) | 5.1.0.95 | Apache-2.0 (OpenCV); the wheel bundles FFmpeg (LGPL-2.1) and other libraries under their own licenses, listed in the wheel | frame sampling, ONNX inference (`cv2.dnn`), debug overlays |
| [pytest](https://pytest.org) | 9.2.0 | MIT | tests (`dev` group) |
| [playwright](https://playwright.dev/python/) | 1.65.2 | Apache-2.0 | browser tests, `scripts/e2e_ui.py` (`e2e` group, optional) |
| [greenlet](https://github.com/python-greenlet/greenlet) | 3.5.6 | MIT AND PSF-2.0 | dependency of playwright |
| [supervision](https://github.com/roboflow/supervision) | 0.26.1 | MIT | ByteTrack, **evaluation only** (`scripts/eval_tracking.py` runs it in a throwaway `uv run --no-project --with` environment); not a dependency of the app |

FFmpeg/ffprobe are required but installed separately by the user and not
distributed with this project. Ollama is optional, installed separately, and
MIT-licensed.

## Model checkpoints (downloaded on request, not redistributed)

`watch-the-goal models download NAME` fetches these from pinned revisions and
checks the SHA-256 before use. Nothing is downloaded unless asked; the default
analysis uses no model.

| Name | File | Source (pinned revision) | SHA-256 | Size | License |
| --- | --- | --- | --- | --- | --- |
| `yolox` | `object_detection_yolox_2022nov.onnx` (YOLOX-S, COCO, 640×640) | OpenCV Zoo, [huggingface.co/opencv/object_detection_yolox](https://huggingface.co/opencv/object_detection_yolox) @ `78c368f74ce73ee28fc7a1be418a598c71b58b52`; model by Megvii ([YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)) | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` | 35,858,002 B | Apache-2.0 |
| `rfdetr-nano` | `rfdetr_nano.onnx` (RF-DETR Nano, COCO, 384×384) | Roboflow [RF-DETR](https://github.com/roboflow/rf-detr); ONNX export [huggingface.co/onnx-community/rfdetr_nano-ONNX](https://huggingface.co/onnx-community/rfdetr_nano-ONNX) @ `eae21cee0687a91bcf9fa071605c48d7705d2d91` (`onnx/model.onnx`) | `9cbac6b11ce34a03034e4d5a24cfac5f18632fd6761d1311dd640232088d7fee` | 108,074,865 B | Apache-2.0 |

Only the COCO "person" class is used.

## AI review model (pulled by the user, not redistributed)

| Model | How to get it | Tested digest | License |
| --- | --- | --- | --- |
| Qwen3.5 9B (vision), Qwen team, Alibaba Cloud | `ollama pull qwen3.5:9b` (Ollama 0.40.18) | `d63f830f28d5` | Apache-2.0 (open weights) |

Any other vision model Ollama serves can be configured in the Analysis dialog;
check its license before use.

## Agent skills

The skills in `.agents/skills/` and `.claude/skills/` (identical copies) were
written for this repository. Two take structure and recipes from these
projects; no files were copied verbatim.

- **webapp-testing**: structure (reconnaissance then action, wait for
  `networkidle` before inspecting, script against a running server) adapted
  from [anthropics/skills `skills/webapp-testing`](https://github.com/anthropics/skills/tree/main/skills/webapp-testing)
  @ `b9e19e6f44773509fbdd7001d77ff41a49a486c1`, Apache-2.0
  (Copyright Anthropic, PBC). The upstream `with_server.py` helper is not
  included; `scripts/e2e_ui.py` starts the app itself.
- **ffmpeg-essentials** and **video-qa**: probe, trim and troubleshooting
  recipes and the QA detector commands (blackdetect, silencedetect, contact
  sheet) adapted from
  [iamaanahmad/agents-video-editing-skills](https://github.com/iamaanahmad/agents-video-editing-skills)
  @ `981466bde5d5044d08cb5af391770bfd16f9c4e4`, MIT. Its notice, and the
  notice it carries for its own source, follow.

```
MIT License

Copyright (c) 2026 agents-video-editing-skills contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

agents-video-editing-skills states that it used
[claude-code-video-toolkit](https://github.com/digitalsamba/claude-code-video-toolkit)
as a reference, MIT, Copyright (c) 2024 Digital Samba, under the same
permission notice as above.

## Fixtures and footage

Test fixtures and the demo match are generated by code (`tests/conftest.py`,
`scripts/make_demo.py`). No real footage, real-footage labels, model files or
benchmark output are in the repository.

## Recap artwork

- Pillow renders the same transparent PNGs for the browser and FFmpeg. The
  lockfile selects 12.3.0 for Python 3.10 and 13.0.0 for newer interpreters.
  License: MIT-CMU (HPND PIL license). [Project and license](https://github.com/python-pillow/Pillow/blob/main/LICENSE).
- Barlow SemiBold, Copyright 2017 The Barlow Project Authors. SIL Open Font
  License 1.1. The unmodified TTF and its full license are in
  `watch_the_goal/assets/`. [Font source](https://github.com/google/fonts/tree/89f5431ff0db41bd2fe3f7ba21a723a01622428b/ofl/barlow).
  Recap layouts, demo captions and scoreboard artwork are original to this repo.

## Interaction and rendering references

The editor uses native pointer and keyboard controls, a small canvas envelope
and FFmpeg. Reference projects were studied without copying code or skills.

- [OpenFilm timeline](https://github.com/openfilm/openfilm) and [LosslessCut](https://github.com/mifi/lossless-cut)
  informed independent event markers, draggable ranges and reversible edits.
- [LottieFiles motion design](https://github.com/LottieFiles/motion-design-skill)
  informed the short selection and reorder transitions and reduced-motion path.
- [WaveSurfer.js](https://github.com/katspaugh/wavesurfer.js) recommends precomputed
  peaks for large files. Here the existing canvas draws 512 local FFmpeg peaks
  for the selected window, reusing them while dragging.
- [HyperFrames](https://github.com/heygen-com/hyperframes) renders HTML through
  Chrome, Node 22+ and FFmpeg. A static title and scoreboard use the existing
  segment compositor and shared PNGs. No HyperFrames runtime was added.
- [FFmpeg skill](https://github.com/kajisho5/ffmpeg-skill) and
  [EditAtlas video-check](https://github.com/EditAtlas/skills/tree/main/skills/video-check)
  informed source probing, audio comparisons and reviewing flagged frames.
