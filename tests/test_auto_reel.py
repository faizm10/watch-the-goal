"""Auto-filling the reel from suggestions (Project.auto_reel and POST /api/reel/auto)."""

import json
import urllib.error

import pytest
from test_coach import call, server  # noqa: F401
from test_merge import m

from watch_the_goal.media import probe
from watch_the_goal.project import Project


def ranked(*spans):
    """Moments ranked in the order given: ranked((60, 75), (10, 28)) makes 60-75 the strongest."""
    out = []
    for rank, (start, end) in enumerate(spans, 1):
        moment = m(start, end, [f"#{rank}"])
        moment.rank = rank
        out.append(moment)
    return out


@pytest.fixture
def project(tmp_path, media):
    p = Project(tmp_path / "ws")
    src = p.add_source(probe(media["long"]), "long.mp4")  # 130 s
    return p, src["id"]


def statuses(p):
    return {c["reasons"][0]: c["status"] for c in (p.clip(i) for i in p.order)}


def test_keeps_the_strongest_until_the_target(project):
    p, sid = project
    p.apply_detection(sid, ranked((60, 75), (10, 28), (100, 115), (35, 50)))  # 15, 18, 15, 15 s
    res = p.auto_reel(30)
    assert statuses(p) == {"#1": "kept", "#2": "kept", "#3": "candidate", "#4": "candidate"}
    assert res["seconds"] == 33 and res["left"] == 2
    # The reel stays in match order, not rank order.
    assert [p.clip(c)["start"] for c in p.order if p.clip(c)["status"] == "kept"] == [10, 60]


def test_never_overrides_the_user_and_skips_ai_rejections(project):
    p, sid = project
    p.apply_detection(sid, ranked((10, 25), (35, 50), (60, 75), (100, 115)))
    ids = {p.clip(c)["reasons"][0]: c for c in p.order}
    p.update_clip(ids["#1"], status="skipped")
    p.clips[ids["#2"]]["review"] = {"status": "done", "label": "other"}
    p.update_clip(ids["#4"], status="kept")  # already 15 s in the reel
    p.auto_reel(None)
    assert statuses(p) == {"#1": "skipped", "#2": "candidate", "#3": "kept", "#4": "kept"}


def test_long_clips_that_overshoot_are_passed_over(project):
    p, sid = project
    p.apply_detection(sid, ranked((10, 25), (30, 90), (100, 115)))  # 15, 60, 15 s
    p.auto_reel(30)
    assert statuses(p) == {"#1": "kept", "#2": "candidate", "#3": "kept"}


def test_one_undo_puts_it_all_back_but_spares_later_decisions(project):
    p, sid = project
    p.apply_detection(sid, ranked((10, 25), (35, 50), (60, 75)))
    kept = p.auto_reel(None)["kept"]
    assert len(kept) == 3
    p.update_clip(kept[0], status="skipped")  # the user changed their mind about one
    assert p.undo()["kind"] == "status"
    assert sorted(statuses(p).values()) == ["candidate", "candidate", "skipped"]


def test_route_validates_the_target(server, media):  # noqa: F811
    url, app = server
    sid = app.import_paths([str(media["long"])])["added"][0]["id"]
    app.project.apply_detection(sid, ranked((10, 25), (35, 50)))
    out = json.load(call(f"{url}/api/reel/auto", "POST", {"target": 20}))
    assert len(out["kept"]) == 1 and out["left"] == 1
    for bad in (-5, 0.5, True, "60"):
        with pytest.raises(urllib.error.HTTPError) as err:
            call(f"{url}/api/reel/auto", "POST", {"target": bad})
        assert err.value.code == 400
