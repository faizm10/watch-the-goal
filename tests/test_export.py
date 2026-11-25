import hashlib

import pytest

from conftest import frame_at
from watch_the_goal.app import App
from watch_the_goal.export import ExportClip, ExportError, choose_layout, export_reel
from watch_the_goal.media import probe


def clip_for(path, start=None, end=None):
    info = probe(path)
    return ExportClip(str(path), start or 0.0, end if end is not None else info.duration,
                      info.has_audio, info.width, info.height, label=path.name)


def streams(path):
    import json, subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


def test_mixed_phone_formats_export_to_one_mp4(media, tmp_path):
    """Portrait HEVC/rotated MOV + 60 fps landscape + no-audio clip, in a chosen order."""
    before = {k: hashlib.sha256(media[k].read_bytes()).hexdigest() for k in ("portrait", "sixty", "silent")}
    clips = [clip_for(media["sixty"]), clip_for(media["silent"]), clip_for(media["portrait"])]
    out = tmp_path / "reel.mp4"
    result = export_reel(clips, out, layout="portrait")

    v, a = (next(s for s in streams(out) if s["codec_type"] == t) for t in ("video", "audio"))
    assert (v["width"], v["height"]) == (1080, 1920)
    assert v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p"
    assert v["r_frame_rate"] == "30/1"
    assert a["codec_name"] == "aac" and int(a["sample_rate"]) == 48000 and a["channels"] == 2
    expected = sum(c.duration for c in clips)
    assert result["duration"] == pytest.approx(expected, abs=0.1)
    assert float(a["duration"]) == pytest.approx(expected, abs=0.1)  # silent clip got a silent track

    # Clip order by brightness code: sixty (base 120), silent (200), portrait (40).
    centre = lambda f: f[f.shape[0] // 3: 2 * f.shape[0] // 3, f.shape[1] // 3: 2 * f.shape[1] // 3].mean()
    t_sixty, t_silent, t_portrait = 1.0, 3.0 + 1.0, 6.0 + 2.5
    assert abs(centre(frame_at(out, t_sixty)) - 140) < 15  # 120 + 20 * 1 s
    assert abs(centre(frame_at(out, t_silent)) - 220) < 15
    # The portrait clip fills the portrait frame upright: white marker at the top.
    f = frame_at(out, t_portrait)
    h = f.shape[0]
    assert f[int(h * 0.03), f.shape[1] // 2] > 200
    assert abs(f[int(h * 0.6), f.shape[1] // 2] - 80) < 15  # 40 + 20 * 2 s

    after = {k: hashlib.sha256(media[k].read_bytes()).hexdigest() for k in before}
    assert after == before  # sources untouched


def test_auto_layout_follows_most_footage(media):
    portrait = clip_for(media["portrait"])
    landscape = clip_for(media["sixty"], 0, 1)
    assert choose_layout([portrait, landscape]) == (1080, 1920)
    assert choose_layout([portrait, clip_for(media["timecode"])]) == (1920, 1080)


def test_export_errors_are_actionable(media, tmp_path):
    with pytest.raises(ExportError, match="Keep at least one clip"):
        export_reel([], tmp_path / "x.mp4")
    gone = clip_for(media["sixty"])
    gone.path = str(tmp_path / "moved.mp4")
    with pytest.raises(ExportError, match="missing.*re-import"):
        export_reel([gone], tmp_path / "x.mp4")


def test_app_export_job_uses_reel_order(media, tmp_path):
    app = App(tmp_path / "ws")
    app.import_paths([str(media["silent"]), str(media["sixty"])])
    ids = [c["id"] for c in app.project.reel()]
    app.project.reorder(list(reversed(ids)))
    job = app.export("landscape")
    import time
    for _ in range(300):
        if job.status != "running":
            break
        time.sleep(0.1)
    assert job.status == "done", job.error
    out = tmp_path / "ws" / "exports" / job.result["name"]
    first = frame_at(out, 0.5)
    assert abs(first[first.shape[0] // 2, first.shape[1] // 2] - 120) < 15  # sixty first now
    assert app.project.exports[0]["name"] == job.result["name"]
