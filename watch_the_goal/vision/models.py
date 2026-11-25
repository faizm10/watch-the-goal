"""Pinned model checkpoints for the optional person detectors.

Nothing is downloaded unless the user asks (``watch-the-goal models download``
or the button in the analysis settings). Each file is pinned to a commit and
checked against its SHA-256, and its licence is recorded here and in
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ModelFile:
    detector: str
    filename: str
    url: str
    sha256: str
    size: int
    license: str
    source: str
    description: str


MODELS: dict[str, ModelFile] = {
    "yolox": ModelFile(
        detector="yolox",
        filename="object_detection_yolox_2022nov.onnx",
        url="https://huggingface.co/opencv/object_detection_yolox/resolve/"
            "78c368f74ce73ee28fc7a1be418a598c71b58b52/object_detection_yolox_2022nov.onnx",
        sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        size=35858002,
        license="Apache-2.0",
        source="OpenCV Zoo (github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox)",
        description="YOLOX-S, COCO, 640x640 input, from OpenCV Zoo",
    ),
    "rfdetr-nano": ModelFile(
        detector="rfdetr-nano",
        filename="rfdetr_nano.onnx",
        url="https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/"
            "eae21cee0687a91bcf9fa071605c48d7705d2d91/onnx/model.onnx",
        sha256="9cbac6b11ce34a03034e4d5a24cfac5f18632fd6761d1311dd640232088d7fee",
        size=108074865,
        license="Apache-2.0",
        source="Roboflow RF-DETR Nano (github.com/roboflow/rf-detr), ONNX export by "
               "huggingface.co/onnx-community/rfdetr_nano-ONNX",
        description="RF-DETR Nano, COCO, 384x384 input",
    ),
}


class ModelUnavailable(Exception):
    """The checkpoint for a detector isn't on this machine (or doesn't verify)."""


def model_dir() -> Path:
    override = os.environ.get("WTG_MODEL_DIR")
    return Path(override).expanduser() if override else Path.home() / ".cache" / "watch-the-goal" / "models"


def model_path(name: str) -> Path:
    if name not in MODELS:
        raise ModelUnavailable(f"Unknown detector {name!r}. Choose one of: {', '.join(MODELS)}.")
    return model_dir() / MODELS[name].filename


def require(name: str) -> Path:
    path = model_path(name)
    spec = MODELS[name]
    if not path.exists():
        raise ModelUnavailable(
            f"The {name} model isn't downloaded ({spec.size / 1e6:.0f} MB). "
            f"Run `watch-the-goal models download {name}` or use the Download button.")
    if path.stat().st_size != spec.size:
        raise ModelUnavailable(f"{path} is incomplete. Download it again.")
    return path


def status() -> list[dict]:
    out = []
    for name, spec in MODELS.items():
        path = model_path(name)
        ready = path.exists() and path.stat().st_size == spec.size
        out.append({**asdict(spec), "name": name, "path": str(path), "downloaded": ready})
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, progress: Callable[[float], None] | None = None) -> Path:
    """Fetch one checkpoint, verify it, and move it into place."""
    path = model_path(name)
    spec = MODELS[name]
    if path.exists() and path.stat().st_size == spec.size and sha256_of(path) == spec.sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(spec.url, timeout=60) as resp, open(tmp, "wb") as fh:
            done = 0
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(min(1.0, done / spec.size))
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailable(f"Couldn't download {name}: {exc}") from None
    digest = sha256_of(tmp)
    if digest != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailable(f"Downloaded {name} doesn't match its pinned checksum ({digest[:12]}…).")
    tmp.replace(path)
    return path
