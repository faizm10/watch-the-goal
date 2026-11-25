import numpy as np
import pytest

from conftest import run
from watch_the_goal import detect, signals
from watch_the_goal.media import probe
from watch_the_goal.signals import Signals

GOALS = [{"x": 0.0, "y": 0.4, "w": 0.12, "h": 0.2}, {"x": 0.88, "y": 0.4, "w": 0.12, "h": 0.2}]


def make_signals(duration=300.0, seed=0):
    rng = np.random.default_rng(seed)
    audio = -40 + rng.normal(0, 1.0, int(duration / signals.AUDIO_HOP)).astype(np.float32)
    motion = 0.02 + np.abs(rng.normal(0, 0.002, int(duration * signals.MOTION_FPS))).astype(np.float32)
    goal = [0.01 + np.abs(rng.normal(0, 0.002, len(motion))).astype(np.float32) for _ in GOALS]
    return Signals(duration=duration, audio_db=audio, motion=motion, goal_motion=goal)


def burst(series, hop, t0, t1, value):
    series[int(t0 / hop):int(t1 / hop)] = value


def test_goal_with_cheer_keeps_buildup_and_explains_itself():
    sig = make_signals()
    burst(sig.goal_motion[1], sig.motion_hop, 100, 102, 0.2)  # ball into the right goal at 100s
    burst(sig.audio_db, sig.audio_hop, 100.4, 104, -12)  # cheer
    moments = detect.find_moments(sig, GOALS)
    assert len(moments) == 1
    mo = moments[0]
    assert mo.anchor == pytest.approx(100, abs=1)
    assert mo.start == pytest.approx(mo.anchor - detect.PRE_ROLL, abs=0.01)  # buildup kept
    assert mo.end >= 104  # celebration kept
    assert any("right goal" in r for r in mo.reasons)
    assert any("crowd noise" in r for r in mo.reasons)


def test_shout_far_from_goals_is_ignored_once_goals_are_marked():
    sig = make_signals()
    burst(sig.audio_db, sig.audio_hop, 150, 153, -12)
    assert detect.find_moments(sig, GOALS) == []
    # Without goal areas the same shout is a reasonable candidate to review.
    sig.goal_motion = []
    assert len(detect.find_moments(sig, [])) == 1


def test_ball_strike_is_too_short_to_count():
    sig = make_signals()
    burst(sig.audio_db, sig.audio_hop, 80, 80.25, -5)
    sig.goal_motion = []
    assert detect.find_moments(sig, []) == []


def test_quiet_footage_without_audio_yields_nothing_or_lulls_only():
    sig = make_signals()
    sig.audio_db = None
    sig.goal_motion = []
    assert detect.find_moments(sig, []) == []
    # Busy play then a clear stoppage is the only thing left to go on.
    burst(sig.motion, sig.motion_hop, 190, 200, 0.08)
    burst(sig.motion, sig.motion_hop, 203, 230, 0.004)
    moments = detect.find_moments(sig, [])
    assert len(moments) == 1 and moments[0].kinds == ["lull"]


def test_candidates_are_capped_for_noisy_footage():
    sig = make_signals(duration=360)
    for t in range(20, 340, 15):
        burst(sig.audio_db, sig.audio_hop, t, t + 2, -10)
    sig.goal_motion = []
    assert len(detect.find_moments(sig, [], max_moments=5)) <= 5


def test_short_clip_trim_needs_audio_or_goal_evidence():
    sig = make_signals(duration=30)
    sig.goal_motion = []
    assert detect.suggest_trim(sig) is None
    burst(sig.audio_db, sig.audio_hop, 20, 23, -10)
    trim = detect.suggest_trim(sig)
    assert trim is not None and trim.start < 20 < trim.end


def test_extract_finds_a_real_noise_burst(tmp_path):
    """End to end through ffmpeg: quiet noise with a loud 3 s burst at 40 s."""
    path = tmp_path / "burst.mp4"
    run("-f", "lavfi", "-i", "color=c=green:s=160x90:r=10:d=60",
        "-f", "lavfi", "-i", "anoisesrc=a=0.01:d=60:r=16000",
        "-af", "volume='if(between(t,40,43),30,1)':eval=frame",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path))
    info = probe(path)
    sig = signals.extract(info.path, info.width, info.height, info.duration, info.has_audio)
    moments = detect.find_moments(sig, [])
    assert len(moments) == 1
    assert moments[0].anchor == pytest.approx(39.5, abs=1)
    assert len(sig.motion) == pytest.approx(60 * signals.MOTION_FPS, abs=2)
