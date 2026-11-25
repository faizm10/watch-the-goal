"""Calls, explanations and re-tuning (watch_the_goal/coach.py), on synthetic signals."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from test_detect import GOALS, burst, make_signals

from watch_the_goal import coach, detect, signals
from watch_the_goal.app import App
from watch_the_goal.project import Project, ProjectError
from watch_the_goal.server import make_server


def cheer(sig, t, db):
    """A crowd reaction ``db`` above the -40 dB floor, from t to t + 3 s."""
    burst(sig.audio_db, sig.audio_hop, t, t + 3, -40 + db)


def anchors(sig, tuning=None, areas=None):
    return [m.anchor for m in detect.candidate_moments(sig, areas or [], tuning=tuning)]


def test_readout_measures_against_the_bar():
    sig = make_signals()
    cheer(sig, 100, 12)
    ev = coach.Evidence(sig)
    r = ev.readout(100)
    assert r.audio_db == pytest.approx(12, abs=1.5)
    assert r.audio_ratio >= 1.5  # twice the default +6 dB
    assert ev.readout(200).audio_ratio < 1


def test_a_missed_quiet_goal_lowers_the_bar_just_enough():
    sig = make_signals()
    cheer(sig, 100, 12)  # loud: suggested
    cheer(sig, 200, 4.5)  # quiet: under the +6 dB bar, missed
    assert not any(abs(a - 200) < 3 for a in anchors(sig))

    result = coach.retune(coach.Evidence(sig), [{"t": 200, "label": "goal"}])
    assert result.tuning.audio_min_db < detect.AUDIO_MIN_DB
    assert result.tuning.audio_min_db > 3.0  # not further than needed
    assert "lowered" in result.changes[0] and "3:20" in result.changes[0]
    assert any(abs(a - 200) < 3 for a in anchors(sig, result.tuning))


def test_not_a_highlight_raises_the_bar_only_up_to_a_confirmed_goal():
    sig = make_signals()
    cheer(sig, 100, 14)  # the goal
    cheer(sig, 200, 8)  # an argument
    cheer(sig, 250, 16)  # louder than the goal: can't be dropped
    calls = [{"t": 200, "label": "other"}]

    alone = coach.retune(coach.Evidence(sig), calls)
    assert alone.tuning == detect.DEFAULT_TUNING  # no confirmed highlight: nothing to anchor a raise
    assert "Confirm at least one highlight" in alone.notes[0]

    calls += [{"t": 100, "label": "goal"}, {"t": 250, "label": "other"}]
    result = coach.retune(coach.Evidence(sig), calls)
    found = anchors(sig, result.tuning)
    assert any(abs(a - 100) < 3 for a in found)
    assert not any(abs(a - 200) < 3 for a in found)
    assert "raised" in result.changes[0] and "3:20" in result.changes[0]


def test_retune_starts_from_the_defaults_every_time():
    sig = make_signals()
    cheer(sig, 200, 4.5)
    calls = [{"t": 200, "label": "goal"}]
    first = coach.retune(coach.Evidence(sig), calls)
    again = coach.retune(coach.Evidence(sig), calls)
    assert first.tuning == again.tuning


def test_a_goal_with_nothing_to_hear_is_explained_not_learned():
    sig = make_signals()
    result = coach.retune(coach.Evidence(sig), [{"t": 150, "label": "goal"}])
    assert result.tuning == detect.DEFAULT_TUNING
    assert "too little crowd noise" in result.notes[0]


def test_goal_box_movement_carries_its_own_highlights():
    sig = make_signals()
    burst(sig.goal_motion[1], sig.motion_hop, 100, 102, 0.2)  # clear goal-mouth action, no cheer
    cheer(sig, 100, 4)
    result = coach.retune(coach.Evidence(sig, GOALS), [{"t": 100, "label": "goal"}])
    # The goal box already sees it, so the noise bar isn't lowered for it.
    assert result.tuning.audio_min_db == detect.AUDIO_MIN_DB


def test_resemblance_moves_lookalikes_and_names_the_call():
    sig = make_signals(duration=600)
    for t in (100, 300):
        cheer(sig, t, 8)  # two alike
    for t in (200, 450):
        cheer(sig, t, 20)  # two alike, much louder
    moments = detect.candidate_moments(sig)
    by_t = lambda t: next(m for m in moments if abs(m.anchor - t) < 3)  # noqa: E731
    before = {t: by_t(t).strength for t in (300, 450)}
    moved = coach.resemble(coach.Evidence(sig), moments, [{"t": 100, "label": "other"}, {"t": 200, "label": "goal"}])
    assert moved == 2
    assert by_t(300).strength < before[300] and "1:40, which you called not a highlight" in by_t(300).reasons[-1]
    assert by_t(450).strength > before[450] and by_t(450).reasons[-1] == "like your goal at 3:20"
    assert moments[0].anchor == by_t(450).anchor or moments[0].anchor == by_t(200).anchor


def test_explain_says_why_in_words():
    sig = make_signals()
    cheer(sig, 200, 4.5)
    ev = coach.Evidence(sig)
    ex = ev.explain(200, detect.DEFAULT_TUNING, [], detect.candidate_moments(sig), 4, label="goal")
    assert ex["verdict"] == "missed"
    assert any("Crowd noise rose" in line and "needs" in line for line in ex["lines"])
    assert ex["lines"][-1] == "Nothing here cleared the bars, so it was missed."
    agree = ev.explain(200, detect.DEFAULT_TUNING, [], [], 4, label="other")
    assert agree["lines"][-1].endswith("which agrees with your call.")


def test_timeline_has_every_signal_next_to_its_bar():
    sig = make_signals()
    cheer(sig, 100, 12)
    tl = coach.Evidence(sig, GOALS).timeline()
    n = int(np.ceil(sig.duration / detect.STEP))
    assert len(tl["audio_db"]) == len(tl["audio_ratio"]) == len(tl["flow"]) == n
    assert [g["label"] for g in tl["goals"]] == ["left goal", "right goal"]
    assert max(tl["audio_ratio"]) >= 1 and tl["audio_needs_db"] >= detect.AUDIO_MIN_DB
    assert any(e["kind"] == "audio" and abs(e["t"] - 100) < 2 for e in tl["events"])


def test_signals_cache_round_trip_and_staleness(tmp_path):
    sig = make_signals()
    f = tmp_path / "s.npz"
    signals.save(sig, f, key="[]")
    back = signals.load(f, key="[]")
    assert back.duration == sig.duration and np.array_equal(back.audio_db, sig.audio_db)
    assert len(back.goal_motion) == len(sig.goal_motion)
    assert signals.load(f, key='[{"x": 0}]') is None  # goal areas changed
    assert signals.load(tmp_path / "missing.npz") is None


def _project_with_source(tmp_path, media):
    from watch_the_goal.media import probe

    p = Project(tmp_path / "ws")
    return p, p.add_source(probe(media["long"]), "long.mp4")


def test_calls_replace_nearby_and_survive_their_clip(tmp_path, media):
    p, src = _project_with_source(tmp_path, media)
    first = p.add_call(src["id"], 50, "goal")
    second = p.add_call(src["id"], 51.5, "save")  # same moment, changed mind
    assert first["id"] == second["id"] and len(src["calls"]) == 1 and src["calls"][0]["label"] == "save"
    clip = p.add_manual_clip(src["id"], 60, 75)
    call = p.add_call(src["id"], 70, "goal", clip["id"])
    p.delete_clip(clip["id"])
    assert call["clip_id"] is None and len(src["calls"]) == 2
    with pytest.raises(ProjectError):
        p.add_call(src["id"], 50, "maybe")
    p.remove_call(src["id"], first["id"])
    assert [c["id"] for c in src["calls"]] == [call["id"]]


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("WTG_MODEL_DIR", str(tmp_path / "no-models"))
    app = App(tmp_path / "ws")
    srv = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", app
    srv.shutdown()


def call(url, method="GET", body=None, raw=None, ctype="application/json"):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, method=method, data=data, headers={"Content-Type": ctype})
    return urllib.request.urlopen(req)


def test_goal_here_adds_a_kept_clip_explains_and_exports_labels(server, media):
    url, app = server
    src = app.import_paths([str(media["long"])])["added"][0]
    sid = src["id"]
    res = json.load(call(f"{url}/api/sources/{sid}/calls", "POST", {"label": "goal", "t": 40}))
    clip = app.project.clip(res["call"]["clip_id"])
    assert clip["status"] == "kept" and clip["origin"] == "manual" and clip["start"] < 40 < clip["end"]
    assert res["explain"]["lines"]

    labels = json.load(call(f"{url}/api/sources/{sid}/labels.json"))
    assert labels["goal_moments"][0]["t"] == 40 and labels["goal_moments"][0]["buildup_start"] == clip["start"]
    assert labels["fully_labeled"] is False

    call(f"{url}/api/sources/{sid}/calls/{res['call']['id']}", "DELETE")
    assert app.project.source(sid)["calls"] == []


def test_live_without_a_model_says_how_to_get_one(server, media):
    import cv2

    url, app = server
    sid = app.import_paths([str(media["long"])])["added"][0]["id"]
    ok, jpg = cv2.imencode(".jpg", np.zeros((90, 160, 3), np.uint8))
    out = json.load(call(f"{url}/api/sources/{sid}/live?t=1.5", "POST", raw=jpg.tobytes(), ctype="image/jpeg"))
    assert out["detector"] is None and "Download" in out["message"]
    with pytest.raises(urllib.error.HTTPError) as err:
        call(f"{url}/api/sources/{sid}/live?t=1", "POST", raw=b"not a jpeg", ctype="image/jpeg")
    assert err.value.code == 400


def test_raising_a_bar_cannot_sweep_away_suggestions_nobody_reviewed():
    sig = make_signals(duration=900)
    cheer(sig, 100, 16)  # the goal
    cheer(sig, 200, 9)  # the one false alarm the user called
    for t in (300, 400, 500, 600):
        cheer(sig, t, 7.5)  # unreviewed and a bit quieter: maybe goals nobody has looked at yet
    calls = [{"t": 100, "label": "goal"}, {"t": 200, "label": "other"}]
    found = detect.candidate_moments(sig)
    held = coach.retune(coach.Evidence(sig), calls, found)
    assert held.tuning == detect.DEFAULT_TUNING
    assert "would also drop 4 suggestions you haven't reviewed" in held.notes[0]

    calls += [{"t": t, "label": "other"} for t in (300, 400, 500)]  # reviewed most of them
    raised = coach.retune(coach.Evidence(sig), calls, found)
    assert raised.tuning.audio_min_db > detect.AUDIO_MIN_DB
    assert "drops 1 suggestion you hadn't reviewed" in raised.changes[0]


def test_marking_goals_resets_learned_tuning_but_keeps_calls(server, media):
    url, app = server
    sid = app.import_paths([str(media["long"])])["added"][0]["id"]
    app.project.add_call(sid, 40, "goal")
    app.project.update_source(sid, tuning={"audio_min_db": 4.0}, coaching={"changes": ["x"]})
    call(f"{url}/api/sources/{sid}/goals", "POST", {"areas": [{"x": 0, "y": .4, "w": .2, "h": .2}]})
    src = app.project.source(sid)
    assert src["tuning"] is None and src["coaching"] is None and len(src["calls"]) == 1


def test_tagging_a_tackle_or_chance_at_the_playhead_keeps_a_clip(server, media):
    url, app = server
    sid = app.import_paths([str(media["long"])])["added"][0]["id"]
    tackle = json.load(call(f"{url}/api/sources/{sid}/calls", "POST", {"label": "tackle", "t": 60}))["call"]
    clip = app.project.clip(tackle["clip_id"])
    assert clip["status"] == "kept" and (clip["start"], clip["end"]) == (54, 64)  # short: a tackle needs little buildup
    assert clip["reasons"] == ["your call: tackle"]
    chance = json.load(call(f"{url}/api/sources/{sid}/calls", "POST", {"label": "chance", "t": 100}))["call"]
    clip = app.project.clip(chance["clip_id"])
    assert clip["start"] == 100 - app.project.settings["pre_roll"]  # a chance keeps the usual buildup
    assert {c["label"] for c in app.project.source(sid)["calls"]} == {"tackle", "chance"}
    assert coach.KIND["save"] == coach.KIND["near_miss"] == "shot"
