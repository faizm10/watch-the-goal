"""Person detectors, frame sampling and player evidence (no model downloads needed)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from watch_the_goal.vision import detectors, frames, models, players

from conftest import luma_second


def test_sample_frames_lands_on_the_requested_second(media):
    got = dict(frames.sample_frames(str(media["timecode"]), [7.5, 0.5, 3.5, 3.7, 9.5]))
    assert sorted(got) == [0.5, 3.5, 3.7, 7.5, 9.5]
    for t, frame in got.items():
        # OpenCV converts to full-range BGR; undo that to read the luma code.
        y = frame.mean() * 219 / 255 + 16
        assert luma_second(y) == int(t)


def test_sample_frames_applies_phone_rotation(media):
    (_, frame), = frames.sample_frames(str(media["portrait"]), [1.0])
    h, w = frame.shape[:2]
    assert h > w  # displayed upright, not as stored
    assert frame[: h // 10].mean() > 180  # the white marker bar is at the top


def test_window_times_cover_both_ends():
    assert frames.window_times(10, 14, 1.0) == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert frames.window_times(5, 5, 1.0) == [5.0]


def test_tiles_cover_the_frame_with_overlap():
    tiles = detectors.tile_grid(1920, 1080, 2, 2)
    assert len(tiles) == 4
    assert min(t[0] for t in tiles) == 0 and max(t[2] for t in tiles) == 1920
    assert min(t[1] for t in tiles) == 0 and max(t[3] for t in tiles) == 1080
    assert tiles[0][2] > tiles[1][0]  # neighbours overlap


def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 20, 0.9], [1, 1, 11, 21, 0.6], [50, 50, 60, 70, 0.5]], dtype=np.float32)
    kept = detectors.nms(boxes)
    assert len(kept) == 2 and kept[:, 4].max() == pytest.approx(0.9)


def test_yolox_decode_maps_a_grid_cell_back_to_pixels():
    det = detectors.YoloxPersons.__new__(detectors.YoloxPersons)
    detectors._Base.__init__(det, threshold=0.3)
    det.size, det.strides = 640, (8, 16, 32)
    grids, strides = [], []
    for s in det.strides:
        n = det.size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    det.grids = np.concatenate(grids).astype(np.float32)
    det.expanded = np.concatenate(strides).astype(np.float32)
    out = np.zeros((len(det.grids), 85), dtype=np.float32)
    cell = 80 * 10 + 20  # stride 8, column 20, row 10
    out[cell, :4] = [0.5, 0.5, np.log(2.0), np.log(4.0)]  # 16 x 32 px box centred in the cell
    out[cell, 4], out[cell, 5] = 0.9, 0.9  # objectness, person
    out[cell + 1, 4], out[cell + 1, 6] = 0.9, 0.9  # a bicycle, ignored
    boxes = det.decode(out, ratio=0.5)  # frame was twice the input size
    assert boxes.shape == (1, 5)
    x0, y0, x1, y1, score = boxes[0]
    assert (x0, y0, x1, y1) == pytest.approx(((20.5 * 8 - 8) * 2, (10.5 * 8 - 16) * 2,
                                              (20.5 * 8 + 8) * 2, (10.5 * 8 + 16) * 2))
    assert score == pytest.approx(0.81)


def test_rfdetr_decode_keeps_people_only():
    det = detectors.RfDetrNanoPersons.__new__(detectors.RfDetrNanoPersons)
    detectors._Base.__init__(det, threshold=0.5)
    boxes = np.array([[0.5, 0.5, 0.1, 0.2], [0.2, 0.2, 0.1, 0.1]], dtype=np.float32)
    logits = np.full((2, 91), -10.0, dtype=np.float32)
    logits[0, 1] = 3.0  # person
    logits[1, 2] = 3.0  # bicycle
    out = det.decode(boxes, logits, 1000, 500)
    assert out.shape == (1, 5)
    assert out[0, :4] == pytest.approx([450, 200, 550, 300])


def test_missing_model_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setenv("WTG_MODEL_DIR", str(tmp_path))
    with pytest.raises(models.ModelUnavailable, match="models download yolox"):
        detectors.load("yolox")
    assert all(not m["downloaded"] for m in models.status())


def test_download_verifies_the_checksum(tmp_path, monkeypatch):
    payload = b"not really a model"
    src = tmp_path / "remote.onnx"
    src.write_bytes(payload)
    monkeypatch.setenv("WTG_MODEL_DIR", str(tmp_path / "models"))
    good = models.ModelFile("yolox", "y.onnx", src.as_uri(), hashlib.sha256(payload).hexdigest(),
                            len(payload), "Apache-2.0", "test", "test")
    monkeypatch.setitem(models.MODELS, "yolox", good)
    assert models.download("yolox").read_bytes() == payload
    assert models.require("yolox").exists()

    bad = models.ModelFile("yolox", "z.onnx", src.as_uri(), "0" * 64, len(payload), "x", "x", "x")
    monkeypatch.setitem(models.MODELS, "yolox", bad)
    with pytest.raises(models.ModelUnavailable, match="checksum"):
        models.download("yolox")
    assert not (tmp_path / "models" / "z.onnx").exists()


# --------------------------------------------------------------------------
# Player evidence

GOAL = {"x": 0.0, "y": 0.4, "w": 0.1, "h": 0.2}  # left goal mouth in a 1000 x 500 frame


def box(cx, foot_y, h=40, score=0.9):
    return [cx - h / 4, foot_y - h, cx + h / 4, foot_y, score]


def test_near_goal_counts_feet_in_the_widened_goal_box():
    boxes = np.array([box(50, 250), box(140, 260), box(600, 250), box(20, 150)], dtype=np.float32)
    # Box spans x 0-100, y 200-300; widened by half: x -50-150, y 150-350.
    assert players.near_goal(boxes, GOAL, 1000, 500) == 3
    assert players.near_goal(boxes, GOAL, 1000, 500, margin=0.0) == 1


def test_bunch_finds_the_tightest_group():
    group = [box(500 + 15 * i, 300) for i in range(5)]
    alone = [box(50, 100), box(950, 450)]
    assert players.bunch(np.array(group + alone, dtype=np.float32)) == 5
    assert players.bunch(np.zeros((0, 5))) == 0


def test_lens_blocked_needs_a_big_person_over_the_goal():
    near_camera = np.array([[0, 100, 120, 480, 0.9]], dtype=np.float32)  # 380 px of a 500 px frame
    far_player = np.array([box(50, 260)], dtype=np.float32)
    assert players.lens_blocked(near_camera, GOAL, 1000, 500)
    assert not players.lens_blocked(far_player, GOAL, 1000, 500)
    assert not players.lens_blocked(near_camera, {"x": 0.8, "y": 0.4, "w": 0.1, "h": 0.2}, 1000, 500)


class FakeDetector:
    """Returns three players in the goal box after ``crowd_after`` seconds into the window."""

    name = "fake"

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        h, w = frame.shape[:2]
        if frame.mean() > 100:  # bright frames: crowd the goal
            return np.array([box(0.05 * w, 0.5 * h, h * 0.1)] * 1 + [box(0.06 * w, 0.52 * h, h * 0.1),
                            box(0.04 * w, 0.55 * h, h * 0.1)], dtype=np.float32)
        return np.array([box(0.6 * w, 0.5 * h, h * 0.1)], dtype=np.float32)


def test_measure_reports_peaks_against_the_recording_typical(media):
    det = FakeDetector()
    # The "long" fixture brightens by 20 luma per second: frames after ~5 s are "bright".
    evidence, stats = players.measure(str(media["long"]), 130.0, [3.0, 100.0], det, [GOAL], ["left goal"])
    assert stats["frames"] == det.calls > 0
    early, late = evidence
    assert early.frames == 6  # window clipped at the start of the file: 0-5 s
    assert late.frames == 9  # 94-102 s at one frame a second
    assert late.near_goal_peak == 3 and late.near_goal_label == "left goal"
    assert late.near_goal_t == 94.0  # first frame at the peak
    assert early.near_goal_peak <= late.near_goal_peak
