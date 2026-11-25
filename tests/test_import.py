import hashlib
import os
import shutil
from pathlib import Path

import pytest

from watch_the_goal.app import App
from watch_the_goal.media import MediaError, probe


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_probe_reads_phone_formats(media):
    portrait = probe(media["portrait"])
    assert (portrait.width, portrait.height) == (240, 426)  # rotation applied
    assert portrait.rotation == 90
    assert portrait.orientation == "portrait"
    assert portrait.has_audio

    sixty = probe(media["sixty"])
    assert round(sixty.fps) == 60
    assert sixty.duration == pytest.approx(3, abs=0.1)

    silent = probe(media["silent"])
    assert not silent.has_audio and silent.audio_codec is None


def test_import_multiple_files_and_folders(media, tmp_path):
    folder = tmp_path / "phone"
    folder.mkdir()
    # Within a folder, recording order wins over file names.
    shutil.copy(media["silent"], folder / "b.mp4")
    shutil.copy(media["sixty"], folder / "a.mp4")
    os.utime(folder / "b.mp4", (1_000_000, 1_000_000))
    os.utime(folder / "a.mp4", (2_000_000, 2_000_000))
    (folder / "notes.txt").write_text("not a video")

    app = App(tmp_path / "ws")
    result = app.import_paths([str(media["long"]), str(media["portrait"]), str(folder)])
    assert result["errors"] == []
    names = [s["name"] for s in app.project.sources]
    assert names == ["long.mp4", "portrait.mov", "b.mp4", "a.mp4"]

    kinds = {s["name"]: s["kind"] for s in app.project.sources}
    assert kinds["long.mp4"] == "long"
    assert all(kinds[n] == "short" for n in names[1:])

    # Short clips are in the reel immediately, whole.
    reel = app.project.reel()
    assert len(reel) == 3
    for clip in reel:
        src = app.project.source(clip["source_id"])
        assert clip["start"] == 0 and clip["end"] == pytest.approx(src["info"]["duration"])

    # Importing the same file again does nothing.
    app.import_paths([str(media["portrait"])])
    assert len(app.project.sources) == 4


def test_import_reports_bad_files(media, tmp_path):
    bogus = tmp_path / "broken.mp4"
    bogus.write_bytes(b"definitely not a video")
    app = App(tmp_path / "ws")
    result = app.import_paths([str(bogus), str(tmp_path / "missing.mov")])
    assert result["added"] == []
    assert len(result["errors"]) == 2
    assert "broken.mp4" in result["errors"][0]
    assert "not found" in result["errors"][1].lower()


def test_upload_copies_and_leaves_original_alone(media, tmp_path):
    app = App(tmp_path / "ws")
    before = digest(media["sixty"])
    with open(media["sixty"], "rb") as fh:
        src = app.import_upload("../../evil name?.mp4", fh, media["sixty"].stat().st_size)
    assert src["copied"]
    assert (tmp_path / "ws" / "media").resolve() in Path(src["path"]).parents
    assert digest(media["sixty"]) == before
    with pytest.raises(MediaError), open(media["sixty"], "rb") as fh:
        app.import_upload("notes.txt", fh, 10)


def test_project_survives_restart(media, tmp_path):
    app = App(tmp_path / "ws")
    app.import_paths([str(media["sixty"])])
    clip = app.project.reel()[0]
    app.project.update_clip(clip["id"], start=0.5, end=2.0)
    again = App(tmp_path / "ws")
    reloaded = again.project.clip(clip["id"])
    assert (reloaded["start"], reloaded["end"]) == (0.5, 2.0)
