import json
import threading
import urllib.request

import pytest

from watch_the_goal.app import App
from watch_the_goal.server import make_server


@pytest.fixture
def server(tmp_path):
    srv = make_server(App(tmp_path / "ws"), "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(url, method="GET", body=None, headers=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    return urllib.request.urlopen(req)


def test_ui_and_state(server):
    assert b"Watch the Goal" in call(server + "/").read()
    state = json.load(call(server + "/api/state"))
    assert state["sources"] == [] and state["clips"] == []


def test_import_range_requests_and_clip_edits(server, media):
    res = json.load(call(server + "/api/import", "POST", {"paths": [str(media["sixty"])]}))
    src = res["added"][0]
    size = media["sixty"].stat().st_size

    part = call(server + f"/media/{src['id']}", headers={"Range": "bytes=100-199"})
    assert part.status == 206
    assert part.headers["Content-Range"] == f"bytes 100-199/{size}"
    assert part.read() == media["sixty"].read_bytes()[100:200]

    clip = json.load(call(server + "/api/state"))["clips"][0]
    edited = json.load(call(server + f"/api/clips/{clip['id']}", "PATCH", {"start": -3, "end": 2}))["clip"]
    assert (edited["start"], edited["end"]) == (0, 2)

    jpg = call(server + f"/api/sources/{src['id']}/frame?t=1").read()
    assert jpg[:2] == b"\xff\xd8"


def test_errors_come_back_as_json(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        call(server + "/api/clips/clip_nope", "PATCH", {"status": "kept"})
    assert err.value.code == 400
    assert "Unknown clip" in json.load(err.value)["error"]
