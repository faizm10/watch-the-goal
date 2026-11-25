import json
import threading
import types

import pytest

from watch_the_goal import writer
from watch_the_goal.app import App
from watch_the_goal.export import ExportClip, export_reel
from watch_the_goal.media import probe
from watch_the_goal.recap import FADE_SECONDS, caption_window
from watch_the_goal.server import make_server


def reel_project(tmp_path, media):
    app = App(tmp_path / "ws")
    app.import_paths([str(media["sixty"]), str(media["silent"])])
    p = app.project
    first, second = p.reel()
    src = p.source(first["source_id"])
    src["play"] = {"start": 0.5, "end": 2.8, "break": [1.0, 1.5]}
    first["anchor"] = 2.2  # 1.7 s into the match, minus a 0.5 s break: the first minute
    p.add_call(first["source_id"], 2.2, "goal", first["id"])
    second["reasons"] = ["the camera followed play to the left end for 6s"]
    p.source(first["source_id"])["scoreboard"] = {"home": "Worker FC", "away": "Naanstop", "score": "3–2"}
    return app, first, second


def test_match_minute_counts_from_kick_off_and_skips_halftime():
    src = {"play": {"start": 100.0, "end": 6000.0, "break": [2800.0, 3700.0]}}
    assert writer.match_minute(src, 100.0) == 1
    assert writer.match_minute(src, 100.0 + 40 * 60 + 5) == 41
    assert writer.match_minute(src, 3000.0) == writer.match_minute(src, 2800.0) == 46  # during the break
    assert writer.match_minute(src, 3700.0 + 60) == 47  # 45 minutes played before the break, then one more
    assert writer.match_minute({}, 500.0) is None  # untrimmed: no match clock


def test_drafts_from_tags_and_detection_without_a_key(tmp_path, media):
    app, first, second = reel_project(tmp_path, media)
    f = writer.facts(app.project)
    out = writer.draft(f)
    assert out["provider"] == "detection"
    assert out["title"] == "Worker FC v Naanstop"
    assert out["captions"] == {first["id"]: "GOAL · 1'", second["id"]: "Pressure"}
    assert out["end"] == "Full time · Worker FC 3–2 Naanstop"


def test_only_text_facts_reach_a_provider(tmp_path, media, monkeypatch):
    app, first, second = reel_project(tmp_path, media)
    seen = {}

    def fake(sdk, key, model, prompt):
        seen.update(key=key, model=model, prompt=prompt)
        return json.dumps({"title": "A" * 90, "end": "Full time", "captions": [
            {"clip": 2, "text": "Wave after wave"}, {"clip": 9, "text": "ignored"}, "junk"]})

    monkeypatch.setitem(writer.CALLS, "claude", fake)
    monkeypatch.setattr(writer, "_sdk", lambda provider: types.SimpleNamespace())
    out = app.draft_text("claude", key="sk-test")
    assert seen["key"] == "sk-test" and seen["model"] == "claude-opus-5-5"
    assert "Clip 1: minute 1" in seen["prompt"] and "tagged goal" in seen["prompt"]
    assert "camera followed play" in seen["prompt"]
    for src in app.project.sources:  # no file names or paths
        assert src["name"] not in seen["prompt"] and src["path"] not in seen["prompt"]
    assert out["captions"] == {first["id"]: "", second["id"]: "Wave after wave"}
    assert len(out["title"]) == writer.TITLE_LIMIT and out["title"].endswith("…")
    # The pasted key is remembered for the session, and never shown back.
    app.draft_text("claude")
    assert seen["key"] == "sk-test"
    assert next(w for w in app.writers() if w["id"] == "claude")["ready"] is True
    assert "sk-test" not in json.dumps(app.writers())
    assert "sk-test" not in app.project.file.read_text()


def test_bad_answers_and_missing_keys_are_explained(tmp_path, media, monkeypatch):
    app, *_ = reel_project(tmp_path, media)
    f = writer.facts(app.project)
    with pytest.raises(writer.WriterError, match="JSON"):
        writer.validate("not json", f)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(writer.WriterError, match="OPENAI_API_KEY"):
        writer.draft(f, "openai")
    with pytest.raises(writer.WriterError, match="Keep some clips"):
        writer.draft({**f, "clips": []})


def test_caption_and_fades_render_into_the_export(media, tmp_path):
    info = probe(media["sixty"])
    clip = ExportClip(str(media["sixty"]), 0, info.duration, info.has_audio, info.width, info.height,
                      caption="GOAL · 47'", fade=FADE_SECONDS)
    out = tmp_path / "reel.mp4"
    result = export_reel([clip], out, "landscape", title="Worker FC v Naanstop", end="Full time · 3–2",
                         fade=FADE_SECONDS)
    assert result["duration"] == pytest.approx(2 + info.duration + 3, abs=0.15)
    from conftest import frame_at

    a, b = caption_window(info.duration)
    # The left of the caption's dark plate, before the text starts.
    plate = lambda f: f[int(f.shape[0] * .84):int(f.shape[0] * .89), int(f.shape[1] * .03):int(f.shape[1] * .042)]  # noqa: E731
    assert plate(frame_at(out, 2 + (FADE_SECONDS + a) / 2)).mean() > 110  # faded in, caption not up yet
    assert plate(frame_at(out, 2 + (a + b) / 2)).mean() < 50  # up
    assert 50 < plate(frame_at(out, 2 + info.duration - 0.1)).mean() < 100  # plate gone, clip fading out
    assert frame_at(out, 2.0).mean() < 20  # a dip to black between the title and the clip


def test_recap_saves_captions_and_end_card_over_http(tmp_path, media):
    app, first, second = reel_project(tmp_path, media)
    srv = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    from test_server import call
    try:
        draft = json.load(call(base + "/api/reel/draft", "POST", {"provider": "detection"}))
        assert draft["captions"][first["id"]] == "GOAL · 1'"
        json.load(call(base + "/api/recap", "POST", {"template": "recap", "title": draft["title"],
                                                     "end": draft["end"], "captions": draft["captions"]}))
        state = json.load(call(base + "/api/state"))
        assert state["settings"]["end_card"] == "Full time · Worker FC 3–2 Naanstop"
        assert {c["id"]: c.get("caption") for c in state["clips"] if c["status"] == "kept"} == draft["captions"]
        assert [w["id"] for w in state["writers"]] == ["detection", "claude", "openai", "gemini"]
        png = call(base + "/api/artwork?kind=caption&layout=portrait&text=GOAL").read()
        assert png[:4] == b"\x89PNG"
    finally:
        srv.shutdown()
