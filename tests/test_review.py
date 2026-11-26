"""Optional AI review: validation, fallbacks, and keeping the user in charge.

No model is needed: a fake client stands in for Ollama, and a tiny HTTP
server exercises the real client's error handling.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from watch_the_goal import review
from watch_the_goal.detect import Moment
from watch_the_goal.media import probe
from watch_the_goal.project import Project, ProjectError

from test_server import call, server  # noqa: F401  (fixture)

# --------------------------------------------------------------------------
# Validation


def test_valid_answer_is_accepted_and_tidied():
    v = review.validate('{"label": "Likely goal", "start": 99.8, "end": 120.4, "reason": "  ball in   the net at 110s "}',
                        100, 120)
    assert v == review.Verdict("likely_goal", 100.0, 120.0, "ball in the net at 110s")


@pytest.mark.parametrize("answer, problem", [
    ("not json", "wasn't JSON"),
    ("[1, 2]", "JSON object"),
    ({"label": "goal", "start": 101, "end": 110, "reason": "x"}, "unknown label"),
    ({"label": "save", "start": 90, "end": 110, "reason": "x"}, "outside the window"),
    ({"label": "save", "start": 101, "end": 125, "reason": "x"}, "outside the window"),
    ({"label": "save", "start": 110, "end": 105, "reason": "x"}, "outside the window"),
    ({"label": "save", "start": 101, "end": 102, "reason": "x"}, "shorter than"),
    ({"label": "save", "start": True, "end": 110, "reason": "x"}, "expected a number"),
    ({"label": "save", "start": 101, "end": float("nan"), "reason": "x"}, "finite"),
    ({"label": "save", "start": 101, "end": 110, "reason": "  "}, "no reason"),
    ({"label": "save", "start": 101}, "missing end, reason"),
])
def test_bad_answers_are_rejected(answer, problem):
    with pytest.raises(review.InvalidResponse, match=problem):
        review.validate(answer, 100, 120)


def test_long_reasons_are_cut_short():
    v = review.validate({"label": "other", "start": 100, "end": 110, "reason": "word " * 200}, 100, 120)
    assert len(v.reason) <= review.MAX_REASON


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:11434/", "http://[::1]:11434"])
def test_local_endpoints_are_allowed(url):
    assert review.require_local(url).startswith("http")


@pytest.mark.parametrize("url", ["http://192.168.1.20:11434", "https://api.example.com", "ftp://127.0.0.1", "nonsense"])
def test_footage_never_goes_to_another_machine(url):
    with pytest.raises(review.Unavailable):
        review.require_local(url)


def test_window_shows_buildup_and_reaction_within_limits():
    assert review.window_for(88, 105, 100, 3000) == (80.0, 108.0)
    assert review.window_for(2, 15, 10, 3000) == (0.0, 18.0)
    w0, w1 = review.window_for(0, 300, 150, 3000)
    assert w1 - w0 <= review.MAX_WINDOW and w0 <= 150 <= w1
    assert review.frame_times(10, 20, 3) == [10.0, 15.0, 20.0]


def test_config_fills_defaults_and_clamps():
    cfg = review.config({"max_candidates": 500, "frames": 1, "timeout": 0, "bogus": 1})
    assert cfg["max_candidates"] == 20 and cfg["frames"] == 4 and cfg["timeout"] == 5
    assert "bogus" not in cfg and cfg["enabled"] is False


# --------------------------------------------------------------------------
# Applying reviews to a project


class FakeClient:
    """Answers per window start; a callable answer can raise or have side effects."""

    def __init__(self, answers, check_error=None, celebrations=None):
        self.answers = answers
        self.check_error = check_error
        self.celebrations = celebrations or {}  # moment -> "yes", "no", "unsure" or a callable
        self.prompts = []
        self.checks = []

    def check(self):
        if self.check_error:
            raise self.check_error
        return {}

    def review(self, prompt, images, schema=review.SCHEMA):
        if schema is review.CELEBRATION_SCHEMA:
            t = float(prompt.split("possible shot at ")[1].split(" s")[0])
            self.checks.append((t, len(images)))
            answer = self.celebrations.get(round(t), "no")
            if callable(answer):
                return answer(t)
            return json.dumps({"celebrating": answer, "reason": f"seen at {t + 3:.1f}s"})
        self.prompts.append((prompt, len(images)))
        w0 = float(prompt.split("between ")[1].split(" s")[0])
        answer = self.answers[round(w0)]
        return answer(w0) if callable(answer) else json.dumps(answer)


@pytest.fixture
def project(tmp_path, media):
    p = Project(tmp_path / "ws")
    src = p.add_source(probe(media["long"]), "long.mp4")
    moments = [Moment(30, 22, 34, ["crowd noise"], ["audio"], 3, rank=2),
               Moment(70, 62, 74, ["crowd noise"], ["audio"], 5, rank=1),
               Moment(110, 102, 114, ["crowd noise"], ["audio"], 2, rank=3)]
    p.apply_detection(src["id"], moments)
    return p, src


def clips_by_anchor(p):
    return {c["anchor"]: c for c in p.clips.values()}


def test_review_labels_reranks_and_proposes_trims(project):
    p, src = project
    client = FakeClient({
        50: {"label": "other", "start": 52, "end": 72, "reason": "players jog back"},
        10: {"label": "likely_goal", "start": 12, "end": 36, "reason": "net bulges at 31s, players celebrate"},
        90: {"label": "near_miss", "start": 92, "end": 113, "reason": "shot over the bar at 108s"},
    })
    summary = review.review_source(p, src["id"], {"frames": 4}, client=client)
    assert summary["status"] == "done" and summary["reviewed"] == 3
    assert summary["labels"] == {"other": 1, "likely_goal": 1, "near_miss": 1}
    c = clips_by_anchor(p)
    assert [c[a]["rank"] for a in (30, 110, 70)] == [1, 2, 3]  # goal, near miss, then "other"
    assert c[30]["detector_rank"] == 2
    assert c[30]["review"]["label"] == "likely_goal" and c[30]["review"]["window"] == [10.0, 38.0]
    # The AI's 12-36 s is widened only as far as a reel clip may run.
    assert (c[30]["suggested_trim"]["start"], c[30]["suggested_trim"]["end"]) == (20, 35)
    assert c[30]["suggested_trim"]["source"] == "ai"
    # Proposals only: the clips themselves are unchanged.
    assert (c[30]["start"], c[30]["end"], c[30]["status"]) == (22, 34, "candidate")
    assert all(n == 4 for _, n in client.prompts)
    assert p.source(src["id"])["review"]["status"] == "done"


def test_review_skips_clips_the_user_has_acted_on(project):
    p, src = project
    c = clips_by_anchor(p)
    p.update_clip(c[70]["id"], status="kept")
    p.update_clip(c[110]["id"], start=99)
    client = FakeClient({10: {"label": "save", "start": 12, "end": 30, "reason": "keeper dives"}})
    summary = review.review_source(p, src["id"], {}, client=client)
    assert summary["reviewed"] == 1 and len(client.prompts) == 1
    c = clips_by_anchor(p)
    assert c[70]["review"] is None and c[110]["review"] is None
    assert (c[110]["start"], c[110]["end"]) == (99, 114)


def test_an_edit_during_review_wins(project):
    p, src = project
    target = clips_by_anchor(p)[70]

    def user_edits_meanwhile(w0):
        p.update_clip(target["id"], start=55)
        return json.dumps({"label": "likely_goal", "start": 52, "end": 78, "reason": "goal"})

    review.review_source(p, src["id"], {"max_candidates": 1}, client=FakeClient({50: user_edits_meanwhile}))
    clip = p.clip(target["id"])
    assert clip["start"] == 55 and clip["suggested_trim"] is None
    assert clip["review"]["label"] == "likely_goal"  # the verdict is still shown


def test_unavailable_reviewer_leaves_the_baseline_alone(project):
    p, src = project
    before = json.dumps(p.snapshot()["clips"])
    summary = review.review_source(p, src["id"], {}, client=FakeClient({}, review.Unavailable("Ollama isn't reachable")))
    assert summary["status"] == "unavailable" and "reachable" in summary["message"]
    assert json.dumps(p.snapshot()["clips"]) == before


def test_timeouts_and_bad_answers_keep_each_suggestion(project):
    p, src = project

    def slow(w0):
        raise review.Timeout("the model took longer than 5s")

    client = FakeClient({50: slow, 10: "definitely a goal!",
                         90: {"label": "save", "start": 0, "end": 300, "reason": "x"}})
    summary = review.review_source(p, src["id"], {}, client=client)
    assert summary["reviewed"] == 0 and summary["failed"] == 3
    c = clips_by_anchor(p)
    assert c[70]["review"]["status"] == "timeout"
    assert c[30]["review"]["status"] == "invalid" and c[110]["review"]["status"] == "invalid"
    assert all(x["suggested_trim"] is None for x in c.values())
    assert [c[a]["rank"] for a in (70, 30, 110)] == [1, 2, 3]  # detector order kept
    assert (c[70]["start"], c[70]["end"]) == (62, 74)


def test_server_dying_mid_review_stops_cleanly(project):
    p, src = project

    def gone(w0):
        raise review.Unavailable("connection refused")

    summary = review.review_source(p, src["id"], {}, client=FakeClient({50: gone}))
    assert summary["status"] == "partial" and summary["reviewed"] == 0


# --------------------------------------------------------------------------
# The real HTTP client against a stand-in Ollama


class FakeOllama(BaseHTTPRequestHandler):
    mode = "ok"
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.seen.append((self.path, body))
        if self.path == "/api/show":
            caps = ["completion"] if FakeOllama.mode == "blind" else ["completion", "vision"]
            return self._send({"capabilities": caps, "license": "Apache License\nVersion 2.0"})
        if FakeOllama.mode == "slow":
            time.sleep(3)
        if FakeOllama.mode == "garbage":
            return self._send({"message": {"content": "I think it's a goal"}})
        return self._send({"message": {"content": json.dumps(
            {"label": "near_miss", "start": 95, "end": 110, "reason": "shot wide at 104s"})}})

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def ollama():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeOllama.seen = []
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_client_sends_frames_and_schema_and_validates(ollama, media):
    FakeOllama.mode = "ok"
    client = review.OllamaClient(ollama, "qwen3.5:9b", 10)
    assert client.check()["capabilities"] == ["completion", "vision"]
    v = review.review_window(client, str(media["long"]), 90, 118, 5)
    assert v.label == "near_miss" and (v.start, v.end) == (95, 110)
    path, body = FakeOllama.seen[-1]
    assert path == "/api/chat" and body["format"] == review.SCHEMA and body["think"] is False
    msg = body["messages"][0]
    assert len(msg["images"]) == 5 and "90.0, 97.0, 104.0, 111.0, 118.0" in msg["content"]


@pytest.mark.parametrize("mode, error", [("garbage", review.InvalidResponse), ("blind", review.Unavailable)])
def test_client_rejects_bad_models_and_answers(ollama, media, mode, error):
    FakeOllama.mode = mode
    client = review.OllamaClient(ollama, "m", 10)
    with pytest.raises(error):
        client.check()
        review.review_window(client, str(media["long"]), 90, 118, 4)


def test_client_times_out(ollama, media):
    FakeOllama.mode = "slow"
    with pytest.raises(review.Timeout):
        review.review_window(review.OllamaClient(ollama, "m", 1), str(media["long"]), 90, 118, 4)


def test_client_reports_a_missing_server():
    client = review.OllamaClient("http://127.0.0.1:9", "m", 2)
    with pytest.raises(review.Unavailable, match="ollama serve"):
        client.check()


# --------------------------------------------------------------------------
# Server settings and old projects


def test_settings_reject_remote_reviewers_and_unknown_detectors(server):  # noqa: F811
    with pytest.raises(urllib.error.HTTPError) as err:
        call(server + "/api/settings", "POST", {"ai_review": {"endpoint": "http://10.0.0.5:11434"}})
    assert "localhost" in json.load(err.value)["error"]
    with pytest.raises(urllib.error.HTTPError):
        call(server + "/api/settings", "POST", {"detector": "magic"})
    s = json.load(call(server + "/api/settings", "POST", {"detector": "yolox", "ai_review": {"enabled": True}}))
    assert s["settings"]["detector"] == "yolox" and s["settings"]["ai_review"]["enabled"] is True
    models = json.load(call(server + "/api/models"))["models"]
    assert {m["name"] for m in models} == {"yolox", "rfdetr-nano"}
    assert all(m["license"] == "Apache-2.0" for m in models)


def test_review_endpoint_reports_unavailable_without_failing(server):  # noqa: F811
    call(server + "/api/settings", "POST", {"ai_review": {"endpoint": "http://127.0.0.1:9"}})
    res = json.load(call(server + "/api/review/check"))
    assert res["ok"] is False and "isn't reachable" in res["message"]


def test_projects_saved_before_this_version_still_load(tmp_path, media):
    ws = tmp_path / "old"
    ws.mkdir()
    info = probe(media["long"]).to_dict()
    old = {
        "sources": [{"id": "src_1", "name": "long.mp4", "path": info["path"], "copied": False, "kind": "long",
                     "info": info, "goal_areas": [], "analysis": {"status": "done", "found": 1}, "preview": "original"}],
        "clips": [{"id": "clip_1", "source_id": "src_1", "start": 10, "end": 25, "status": "candidate",
                   "origin": "detected", "reasons": ["crowd noise"], "anchor": 22, "edited": False,
                   "suggested_trim": None},
                  {"id": "clip_2", "source_id": "src_1", "start": 40, "end": 50, "status": "kept",
                   "origin": "manual", "reasons": ["added by you"], "anchor": None, "edited": True,
                   "suggested_trim": None}],
        "order": ["clip_1", "clip_2"],
        "exports": [],
        "settings": {"layout": "portrait", "pre_roll": 12.0, "post_roll": 5.0},
    }
    (ws / "project.json").write_text(json.dumps(old))
    p = Project(ws)
    assert p.settings["layout"] == "portrait" and p.settings["detector"] == "none"
    assert p.settings["ai_review"]["enabled"] is False
    assert [c["id"] for c in p.reel()] == ["clip_2"]
    # Review works on old clips (no rank yet) and leaves the manual clip alone.
    client = FakeClient({2: {"label": "save", "start": 5, "end": 28, "reason": "keeper tips it wide"}})
    summary = review.review_source(p, "src_1", {}, client=client)
    assert summary["reviewed"] == 1
    assert p.clip("clip_1")["rank"] == 1 and p.clip("clip_2").get("review") is None
    # A fresh analysis replaces the untouched suggestion but keeps the manual clip.
    p.apply_detection("src_1", [Moment(80, 68, 85, ["crowd noise"], ["audio"], 3, rank=1)])
    assert sorted(c["id"] for c in p.clips.values() if c["origin"] == "manual") == ["clip_2"]
    assert any(c.get("rank") == 1 and c["anchor"] == 80 for c in p.clips.values())


def test_celebration_makes_a_likely_goal(project):
    p, src = project
    client = FakeClient({
        10: {"label": "near_miss", "start": 24, "end": 34, "reason": "shot wide at 29s"},
        50: {"label": "near_miss", "start": 62, "end": 74, "reason": "shot wide at 70s"},
        90: {"label": "other", "start": 102, "end": 114, "reason": "throw-in"},
    }, celebrations={30: "yes", 70: "unsure", 110: "no"})
    review.review_source(p, src["id"], {"frames": 4}, client=client)
    c = clips_by_anchor(p)
    assert c[30]["review"]["label"] == "likely_goal" and c[30]["review"]["celebration"] == "yes"
    assert c[30]["review"]["reason"] == "players celebrate: seen at 33.0s"
    assert (c[70]["review"]["label"], c[70]["review"]["celebration"]) == ("near_miss", "unsure")
    assert (c[110]["review"]["label"], c[110]["review"]["celebration"]) == ("other", "no")
    assert [c[a]["rank"] for a in (30, 70, 110)] == [1, 2, 3]
    # Seven bigger frames from just before to 12 s after each moment, inside the 130 s recording.
    assert sorted(client.checks) == [(30, 7), (70, 7), (110, 7)]


def test_a_failed_celebration_check_keeps_the_first_look(project):
    p, src = project

    def slow(t):
        raise review.Timeout("too slow")

    client = FakeClient({
        10: {"label": "near_miss", "start": 24, "end": 34, "reason": "shot wide at 29s"},
        50: {"label": "save", "start": 62, "end": 74, "reason": "keeper dives at 70s"},
        90: {"label": "other", "start": 102, "end": 114, "reason": "throw-in"},
    }, celebrations={30: slow, 70: lambda t: "not json", 110: lambda t: json.dumps({"celebrating": "maybe"})})
    summary = review.review_source(p, src["id"], {"frames": 4}, client=client)
    assert summary["reviewed"] == 3 and summary["failed"] == 0
    c = clips_by_anchor(p)
    assert [c[a]["review"]["label"] for a in (30, 70, 110)] == ["near_miss", "save", "other"]
    assert not any("celebration" in c[a]["review"] for a in (30, 70, 110))
    assert review.celebration_times(120, 130) == [118, 120, 122, 124, 126, 129]
    assert review.celebration_times(1, 130)[:2] == [1, 3]  # nothing before the recording starts


def test_ai_trims_only_extend_the_clip(project):
    """A verdict that would cut into the buildup or the reaction only widens."""
    p, src = project
    client = FakeClient({
        10: {"label": "likely_goal", "start": 24, "end": 30, "reason": "shot at 29s"},     # inside 22-34
        50: {"label": "near_miss", "start": 52, "end": 70, "reason": "shot wide at 68s"},  # earlier start, cuts end
        90: {"label": "other", "start": 100, "end": 112, "reason": "throw-in"},
    })
    review.review_source(p, src["id"], {"frames": 4}, client=client)
    c = clips_by_anchor(p)
    assert c[30]["suggested_trim"] is None  # nothing to add
    assert c[70]["review"]["start"] == 52  # the verdict itself is kept as given
    assert (c[70]["suggested_trim"]["start"], c[70]["suggested_trim"]["end"]) == (59, c[70]["end"])
    assert review.widen({"start": 22, "end": 34, "anchor": 30}, {"start": 20, "end": 30}) == (20, 34)
    # Never past MAX_CLIP, and never cutting a clip that was already longer.
    assert review.widen({"start": 22, "end": 34, "anchor": 30}, {"start": 5, "end": 50}) == (20, 35)
    assert review.widen({"start": 10, "end": 30, "anchor": 25}, {"start": 0, "end": 40}) == (10, 30)


@pytest.mark.parametrize("request_name,answer,expected", [
    ("buildup", (53, 73), (53, 74)),
    ("sooner", (64, 72), (62, 72)),
])
def test_context_proposal_preserves_other_edge_and_requires_apply(project, monkeypatch, request_name, answer, expected):
    p, _ = project
    p.settings["ai_review"]["enabled"] = True
    clip = clips_by_anchor(p)[70]
    client = FakeClient({47: {"label": "likely_goal", "start": answer[0], "end": answer[1], "reason": "pass"},
                         62: {"label": "likely_goal", "start": answer[0], "end": answer[1], "reason": "finish"}})
    monkeypatch.setattr(review, "OllamaClient", lambda *args: client)
    result = review.propose_edit(p, clip["id"], request_name)
    assert (clip["start"], clip["end"]) == (62, 74)
    assert (result["start"], result["end"]) == expected
    p.update_clip(clip["id"], **result)
    p.undo()
    assert (clip["start"], clip["end"]) == (62, 74)
    p.update_clip(clip["id"], start=61)
    with pytest.raises(ProjectError, match="changed"):
        p.update_clip(clip["id"], **result)
    assert clip["start"] == 61


def test_context_proposal_rejects_end_before_the_event(project, monkeypatch):
    p, _ = project
    p.settings["ai_review"]["enabled"] = True
    client = FakeClient({62: {"label": "likely_goal", "start": 64, "end": 68, "reason": "too soon"}})
    monkeypatch.setattr(review, "OllamaClient", lambda *args: client)
    clip = clips_by_anchor(p)[70]
    with pytest.raises(review.InvalidResponse, match="no usable change"):
        review.propose_edit(p, clip["id"], "sooner")
    assert (clip["start"], clip["end"]) == (62, 74)
