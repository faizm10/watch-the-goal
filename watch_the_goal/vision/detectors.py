"""Frame-level person detectors behind one interface.

Every detector takes a BGR frame and returns an ``(N, 5)`` float array of
``x0, y0, x1, y1, score`` in that frame's pixels, people only. Both models
run through OpenCV's DNN module, so no extra runtime is needed.

Distant players are often under 20 px tall in a 1080p wide shot. ``tiles``
splits the frame into overlapping parts and detects in each, which trades
time for recall on small people.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from . import models

NAMES = ("yolox", "rfdetr-nano")


class PersonDetector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> np.ndarray: ...


def nms(boxes: np.ndarray, iou: float = 0.5) -> np.ndarray:
    if len(boxes) == 0:
        return boxes.reshape(0, 5)
    xywh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), boxes[:, 4].tolist(), 0.0, iou)
    return boxes[np.asarray(keep, dtype=int).reshape(-1)]


def tile_grid(width: int, height: int, cols: int, rows: int, overlap: float = 0.15) -> list[tuple[int, int, int, int]]:
    """Overlapping (x0, y0, x1, y1) tiles covering the frame."""
    tw = width / (cols - (cols - 1) * overlap)
    th = height / (rows - (rows - 1) * overlap)
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * tw * (1 - overlap)))
            y0 = int(round(r * th * (1 - overlap)))
            out.append((x0, y0, min(width, int(round(x0 + tw))), min(height, int(round(y0 + th)))))
    return out


class _Base:
    name = "base"

    def __init__(self, threshold: float, tiles: tuple[int, int] = (1, 1)):
        self.threshold = threshold
        self.tiles = tiles

    def _infer(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> np.ndarray:
        cols, rows = self.tiles
        if cols * rows <= 1:
            return self._infer(frame)
        h, w = frame.shape[:2]
        found = [self._infer(frame)]  # whole frame too, for people split by a tile edge
        for x0, y0, x1, y1 in tile_grid(w, h, cols, rows):
            boxes = self._infer(frame[y0:y1, x0:x1])
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                found.append(boxes)
        return nms(np.concatenate(found), 0.5)


class YoloxPersons(_Base):
    """YOLOX-S from OpenCV Zoo. Letterboxed RGB input, no normalisation."""

    name = "yolox"
    size = 640
    strides = (8, 16, 32)

    def __init__(self, path: str, threshold: float = 0.3, tiles: tuple[int, int] = (1, 1)):
        super().__init__(threshold, tiles)
        self.net = cv2.dnn.readNet(str(path))
        grids, strides = [], []
        for s in self.strides:
            n = self.size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.expanded = np.concatenate(strides).astype(np.float32)

    def _infer(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ratio = min(self.size / h, self.size / w)
        canvas = np.full((self.size, self.size, 3), 114.0, dtype=np.float32)
        nw, nh = int(w * ratio), int(h * ratio)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.net.setInput(canvas.transpose(2, 0, 1)[None])
        out = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]
        return self.decode(out, ratio)

    def decode(self, out: np.ndarray, ratio: float) -> np.ndarray:
        scores = out[:, 4] * out[:, 5]  # objectness x person (COCO class 0)
        keep = scores >= self.threshold
        if not keep.any():
            return np.zeros((0, 5), dtype=np.float32)
        d = out[keep]
        xy = (d[:, :2] + self.grids[keep]) * self.expanded[keep]
        wh = np.exp(d[:, 2:4]) * self.expanded[keep]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / ratio
        return nms(np.concatenate([boxes, scores[keep, None]], 1).astype(np.float32), 0.5)


class RfDetrNanoPersons(_Base):
    """RF-DETR Nano (ONNX). Plain 384x384 resize, ImageNet normalisation,
    300 queries with sigmoid class scores; COCO id 1 is a person."""

    name = "rfdetr-nano"
    size = 384
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    person = 1

    def __init__(self, path: str, threshold: float = 0.3, tiles: tuple[int, int] = (1, 1)):
        super().__init__(threshold, tiles)
        self.net = cv2.dnn.readNet(str(path))

    def _infer(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = ((rgb - self.mean) / self.std).transpose(2, 0, 1)[None]
        self.net.setInput(blob)
        boxes, logits = self.net.forward(["pred_boxes", "logits"])
        return self.decode(boxes[0], logits[0], w, h)

    def decode(self, boxes: np.ndarray, logits: np.ndarray, w: int, h: int) -> np.ndarray:
        scores = 1.0 / (1.0 + np.exp(-logits[:, self.person]))
        keep = scores >= self.threshold
        if not keep.any():
            return np.zeros((0, 5), dtype=np.float32)
        cx, cy, bw, bh = boxes[keep].T
        out = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h,
                        scores[keep]], 1)
        return nms(out.astype(np.float32), 0.7)


def load(name: str, threshold: float | None = None, tiles: tuple[int, int] = (1, 1)) -> PersonDetector:
    """Build a detector. Raises :class:`models.ModelUnavailable` if its checkpoint is missing."""
    path = models.require(name)
    kwargs = {"tiles": tiles}
    if threshold is not None:
        kwargs["threshold"] = threshold
    if name == "yolox":
        return YoloxPersons(str(path), **kwargs)
    if name == "rfdetr-nano":
        return RfDetrNanoPersons(str(path), **kwargs)
    raise models.ModelUnavailable(f"Unknown detector {name!r}")
