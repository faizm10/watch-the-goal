"""Trimming a recording to the match: kick-off, full time and halftime."""

import json
import time

import pytest
from test_coach import call, cheer, server  # noqa: F401
from test_detect import make_signals

from watch_the_goal import detect
from watch_the_goal.media import probe
from watch_the_goal.project import Project, ProjectError


def test_play_spans():
    assert detect.play_spans(None, 600) == [(0.0, 600)]
    assert detect.play_spans({"start": 60, "end": 500, "break": [250, 300]}, 600) == [(60, 250), (300, 500)]
    assert detect.play_spans({"start": 60, "end": None, "break": None}, 600) == [(60, 600)]
    assert detect.play_seconds([(60, 250), (300, 500)]) == 390


def test_only_moments_during_play_are_suggested_and_clips_stay_in_their_half():
    sig = make_signals(duration=900)
    for t in (30, 200, 320, 470, 850):  # warm-up, first half, halftime, second half, after full time
        cheer(sig, t, 12)
    spans = detect.play_spans({"start": 60, "end": 800, "break": [300, 460]}, 900)
    anchors = sorted(round(m.anchor) for m in detect.candidate_moments(sig, spans=spans))
    assert [a for a in anchors if a < 300] and all(60 <= a <= 300 or 460 <= a <= 800 for a in anchors)
    assert any(abs(a - 470) < 5 for a in anchors) and not any(abs(a - 320) < 5 for a in anchors)
    second_half = next(m for m in detect.candidate_moments(sig, spans=spans) if abs(m.anchor - 470) < 5)
    assert second_half.start >= 460  # the buildup doesn't reach back into halftime


def test_set_play_validates_and_clears(tmp_path, media):
    p = Project(tmp_path / "ws")
    src = p.add_source(probe(media["long"]), "long.mp4")  # 130 s
    sid = src["id"]
    assert p.set_play(sid, {"start": 5, "end": 120, "break": [50, 60]})["play"] == {
        "start": 5, "end": 120, "break": [50, 60]}
    with pytest.raises(ProjectError):
        p.set_play(sid, {"start": 100, "end": 120})  # under a minute
    with pytest.raises(ProjectError):
        p.set_play(sid, {"start": 5, "end": 120, "break": [130, 140]})
    assert p.set_play(sid, {"start": 0, "end": 999})["play"] is None  # the whole recording: nothing trimmed
    assert p.set_play(sid, None)["play"] is None


def test_route_reanalyses_and_explains_outside_the_match(server, media):  # noqa: F811
    url, app = server
    sid = app.import_paths([str(media["long"])])["added"][0]["id"]
    out = json.load(call(f"{url}/api/sources/{sid}/play", "POST", {"play": {"start": 20, "end": 110}}))
    assert out["play"] == {"start": 20, "end": 110, "break": None} and out["job"]
    for _ in range(200):
        if app.project.source(sid)["analysis"].get("status") == "done":
            break
        time.sleep(.05)
    ex = json.load(call(f"{url}/api/sources/{sid}/explain?t=5"))
    assert ex["verdict"] == "outside" and "outside the match" in ex["lines"][-1]
    assert all(c["anchor"] is None or 20 <= c["anchor"] <= 110
               for c in app.project.clips.values() if c["origin"] == "detected")
