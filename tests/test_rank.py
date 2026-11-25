"""Ranking with player evidence, lead-in, and the shared pipeline."""

from __future__ import annotations

import numpy as np

from watch_the_goal import detect, pipeline
from watch_the_goal.detect import Moment
from watch_the_goal.signals import Signals
from watch_the_goal.vision import models
from watch_the_goal.vision.players import PlayerEvidence

from test_detect import burst
from test_detect import make_signals as base_signals


def make_signals(duration, cheers):
    """Quiet footage with a cheer of ``db`` above the floor at each (t, seconds, db)."""
    sig = base_signals(duration)
    sig.goal_motion = []
    for t, length, db in cheers:
        burst(sig.audio_db, sig.audio_hop, t, t + length, -40 + db)
    return sig


def moment(anchor, kinds, reasons, strength):
    return Moment(anchor, anchor - 12, anchor + 5, list(reasons), list(kinds), strength)


def test_players_near_goal_add_a_reason_and_lift_the_rank():
    quiet = moment(100, ["audio"], ["crowd noise +8 dB above the surrounding level for 1.0s"], 2.5)
    loud = moment(300, ["audio", "lull"], ["crowd noise +14 dB above the surrounding level for 3.0s",
                                           "play stopped after a burst of action"], 4.0)
    ev_quiet = PlayerEvidence(near_goal_peak=5, near_goal_t=101.0, near_goal_label="left goal", near_goal_typical=1)
    ev_loud = PlayerEvidence(near_goal_peak=1, near_goal_label="left goal", near_goal_typical=1)
    out = detect.apply_player_evidence([quiet, loud], [ev_quiet, ev_loud])
    assert "5 players near the left goal (usually 1)" in out[0].reasons
    assert "players" in out[0].kinds and "players" not in out[1].kinds
    ranked = detect.select(out, 2)
    assert [m.anchor for m in ranked] == [100, 300]
    assert ranked[0].rank == 1  # the goalmouth scramble now leads


def test_finish_found_by_players_stays_inside_the_clip():
    m = moment(100, ["audio"], ["crowd noise"], 3)
    ev = PlayerEvidence(near_goal_peak=4, near_goal_t=106.0, near_goal_label="right goal", near_goal_typical=0)
    (out,) = detect.apply_player_evidence([m], [ev], post_roll=5.0, duration=200)
    assert out.end >= 106.0 + 2.5


def test_someone_at_the_lens_cancels_goal_box_motion():
    goal_only = moment(50, ["goal", "lull"], ["motion near the marked left goal",
                                              "play stopped after a burst of action"], 4)
    with_noise = moment(200, ["audio", "goal"], ["motion near the marked left goal",
                                                 "crowd noise +12 dB above the surrounding level for 2.0s"], 5)
    blocked = PlayerEvidence(lens_blocked=True)
    out = detect.apply_player_evidence([goal_only, with_noise], [blocked, blocked])
    assert [m.anchor for m in out] == [200]  # nothing left to support the first one
    assert out[0].kinds == ["audio"]
    assert not any(r.startswith("motion near") for r in out[0].reasons)


def test_bunched_players_count_when_no_goal_is_marked():
    m = moment(80, ["audio"], ["crowd noise"], 2)
    ev = PlayerEvidence(bunch_peak=9, bunch_typical=4)
    (out,) = detect.apply_player_evidence([m], [ev])
    assert out.reasons[-1] == "9 players bunched together (usually 4)"
    ev = PlayerEvidence(bunch_peak=6, bunch_typical=4)  # not unusual enough
    (out,) = detect.apply_player_evidence([moment(80, ["audio"], ["crowd noise"], 2)], [ev])
    assert "players" not in out.kinds


def test_unmeasured_moments_pass_through_unchanged():
    m = moment(80, ["audio"], ["crowd noise"], 2)
    (out,) = detect.apply_player_evidence([m], [None])
    assert out is m and out.reasons == ["crowd noise"]


def test_merged_moments_keep_one_reason_per_kind_of_player_evidence():
    a = Moment(100, 88, 105, ["7 players near the left goal (usually 2)"], ["players"], 3, rank=2)
    b = Moment(104, 92, 109, ["5 players near the left goal (usually 2)"], ["players"], 2, rank=1)
    (m,) = detect.merge_overlapping([a, b])
    assert m.reasons == ["7 players near the left goal (usually 2)"]
    assert m.rank == 1


def test_lead_in_pulls_back_when_play_is_in_full_flow():
    n = 200  # 100 s at 0.5 s per sample
    motion = np.full(n, 1.0)
    motion[int(40 / detect.STEP):int(70 / detect.STEP)] = 3.0  # a surge from 40 s to 70 s
    # Moment at 60 s: the 12 s cut (48 s) lands mid-surge, so start where it began, within 20 s.
    assert detect.lead_in(motion, 60.0, 12.0) == 40.0
    # Moment at 80 s: the cut (68 s) is mid-surge, but the surge began 40 s earlier; stop at 20 s.
    assert detect.lead_in(motion, 80.0, 12.0) == 60.0
    # Quiet play at the cut: the normal pre-roll.
    assert detect.lead_in(motion, 30.0, 12.0) == 18.0
    assert detect.lead_in(None, 30.0, 12.0) == 18.0


def test_find_moments_still_ranks_and_caps():
    sig = make_signals(300, cheers=[(60, 3, 15), (150, 3, 9), (240, 3, 12)])
    ms = detect.find_moments(sig, max_moments=2)
    assert len(ms) == 2 and sorted(m.rank for m in ms) == [1, 2]
    assert all("crowd noise" in m.reasons[0] for m in ms)


def test_pipeline_without_a_model_falls_back_to_the_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("WTG_MODEL_DIR", str(tmp_path))
    sig = make_signals(300, cheers=[(60, 3, 15), (240, 3, 12)])
    info = {"duration": sig.duration, "width": 320, "height": 180, "has_audio": True}
    base = pipeline.analyze("unused.mp4", info, options=pipeline.Options(), sig=sig)
    off = pipeline.analyze("unused.mp4", info, options=pipeline.Options(detector="yolox"), sig=sig)
    assert [m.to_dict() for m in off.moments] == [m.to_dict() for m in base.moments]
    assert off.notes and "isn't downloaded" in off.notes[0]


def test_pipeline_uses_a_detector_and_saves_debug_files(tmp_path, monkeypatch, media):
    from test_vision import GOAL, FakeDetector

    monkeypatch.setattr("watch_the_goal.vision.detectors.load", lambda name, **kw: FakeDetector())
    n = int(130 / 0.25)
    audio = np.full(n, -40.0, dtype=np.float32)
    audio[int(100 / 0.25):int(103 / 0.25)] = -20.0  # a cheer at 100 s
    goal = np.full(650, 0.01, dtype=np.float32)
    goal[500:510] = 0.3  # action in the goal box at 100 s (marked goals need it)
    rng = np.random.default_rng(0)
    goal += np.abs(rng.normal(0, 0.002, 650)).astype(np.float32)
    sig = Signals(duration=130.0, audio_db=audio, motion=np.full(650, 0.01, dtype=np.float32), goal_motion=[goal])
    info = {"duration": 130.0, "width": 160, "height": 90, "has_audio": True}
    res = pipeline.analyze(str(media["long"]), info, [GOAL],
                           pipeline.Options(detector="yolox", debug_dir=tmp_path / "debug"), sig=sig)
    assert res.stats["player_frames"] > 0 and "players_seconds" in res.stats
    (m,) = res.moments
    assert any(r.startswith("motion near the marked") for r in m.reasons)
    assert (tmp_path / "debug" / "candidates.json").exists()
    assert models.MODELS  # registry stays importable without cv2 models


def test_signals_without_detector_are_untouched_by_ranking_fields():
    sig = make_signals(300, cheers=[(60, 3, 15)])
    (m,) = detect.find_moments(sig)
    d = m.to_dict()
    assert d["rank"] == 1 and set(d) == {"anchor", "start", "end", "reasons", "kinds", "rank"}
