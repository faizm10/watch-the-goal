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


def panning_signals(duration=600.0, seed=1):
    """A camera that follows play: it sweeps about a frame width back and forth."""
    sig = make_signals(duration, seed)
    t = np.arange(len(sig.motion)) * sig.motion_hop
    pos = 0.5 * np.sin(2 * np.pi * t / 120)
    return sig, t, pos


def test_auto_panning_camera_held_at_one_end_then_play_stops_is_a_moment():
    sig, t, pos = panning_signals()
    held = (t >= 300) & (t < 308)
    pos[held] = 0.6  # play is at the left end...
    pos[t >= 308] = 0.6 + 0.5 * np.sin(2 * np.pi * (t[t >= 308] - 308) / 120 + np.pi / 2) - 0.5
    burst(sig.motion, sig.motion_hop, 290, 308, 0.04)  # ...and busy
    burst(sig.motion, sig.motion_hop, 308, 325, 0.005)  # then the ball goes dead
    sig.pan = np.diff(pos, prepend=pos[0]).astype(np.float32)
    n = int(np.ceil(sig.duration / detect.STEP))
    held = next(e for e in detect.attack_events(sig, n) if abs(e.t - 308) <= 2)
    assert "left end" in held.reason
    # The camera reaches either end every minute, but only this time did play stop after.
    moments = [m for m in detect.find_moments(sig, []) if "attack" in m.kinds]
    assert len(moments) == 1
    mo = moments[0]
    assert mo.start <= 300 <= 308 <= mo.end
    assert any("camera followed play to the left end" in r for r in mo.reasons)


def test_attack_joins_the_cheer_it_led_to():
    sig, t, pos = panning_signals()
    pos[(t >= 290) & (t < 312)] = 0.6
    burst(sig.motion, sig.motion_hop, 285, 312, 0.04)
    burst(sig.audio_db, sig.audio_hop, 300, 303, -12)  # the finish; the celebration keeps the camera there
    sig.pan = np.diff(pos, prepend=pos[0]).astype(np.float32)
    mo = next(m for m in detect.find_moments(sig, []) if "audio" in m.kinds)
    assert "attack" in mo.kinds
    assert mo.anchor == pytest.approx(299.5, abs=1)


def test_fixed_camera_never_reports_an_attack():
    sig = make_signals(600.0, 1)
    burst(sig.motion, sig.motion_hop, 290, 308, 0.04)
    burst(sig.motion, sig.motion_hop, 308, 325, 0.005)
    n = int(np.ceil(sig.duration / detect.STEP))
    assert detect.camera_offset(sig, n) is None  # no pan measured
    sig.pan = np.zeros(len(sig.motion), np.float32)
    assert detect.camera_offset(sig, n) is None  # on a tripod
    assert detect.attack_events(sig, n) == []


def test_signals_cached_before_pan_was_measured_are_stale(tmp_path):
    sig, _, pos = panning_signals(120.0)
    sig.pan = np.diff(pos, prepend=pos[0]).astype(np.float32)
    f = tmp_path / "s.npz"
    signals.save(sig, f, key="[]")
    assert np.allclose(signals.load(f, key="[]").pan, sig.pan)
    d = dict(np.load(f))
    del d["pan"]
    np.savez(f, **d)
    assert signals.load(f, key="[]") is None


def test_extract_measures_which_way_the_camera_turns(tmp_path):
    """A crop window sliding right over a wide scene is a camera turning right."""
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(3)  # a still, textured scene, the same on every ffmpeg build
    scene = tmp_path / "scene.png"
    Image.fromarray(rng.integers(0, 256, (180, 960), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(2)).save(scene)
    path = tmp_path / "pan.mp4"
    run("-loop", "1", "-framerate", "10", "-t", "8", "-i", str(scene),
        "-vf", "crop=320:180:x='min(t*40,600)':y=0,format=yuv420p", "-c:v", "libx264", str(path))
    info = probe(path)
    sig = signals.extract(info.path, info.width, info.height, info.duration, info.has_audio)
    assert len(sig.pan) == len(sig.motion)
    turned = float(np.sum(sig.pan))  # in frame widths
    assert turned == pytest.approx(-8 * 40 / 320, rel=0.25)
