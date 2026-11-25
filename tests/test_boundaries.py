import pytest

from conftest import frame_at, luma_second
from watch_the_goal.detect import clamp_window
from watch_the_goal.export import ExportClip, export_reel
from watch_the_goal.media import probe
from watch_the_goal.project import Project, ProjectError


@pytest.mark.parametrize("start,end,duration,expected", [
    (5, 10, 60, (5, 10)),
    (-4, 6, 60, (0, 6)),  # extending back past the first frame stops at 0
    (55, 70, 60, (55, 60)),  # extending past the end stops at the end
    (59.8, 59.9, 60, (59.0, 60)),  # too short near the end: keep 1s, inside the file
    (10, 10.2, 60, (10, 11)),  # too short: widened to 1s
    (0, 5, 0.6, (0, 0.6)),  # file shorter than the minimum
])
def test_clamp_window(start, end, duration, expected):
    assert clamp_window(start, end, duration, 1.0) == pytest.approx(expected)


def test_clip_edits_are_clamped(media, tmp_path):
    project = Project(tmp_path / "ws")
    src = project.add_source(probe(media["timecode"]), "timecode.mp4")  # 10 s
    clip = project.add_manual_clip(src["id"], -5, 4)
    assert (clip["start"], clip["end"]) == (0, 4)
    clip = project.update_clip(clip["id"], start=3, end=99)
    assert (clip["start"], clip["end"]) == (3, 10)
    clip = project.update_clip(clip["id"], start=8, end=6)  # swapped handles
    assert (clip["start"], clip["end"]) == (6, 8)
    with pytest.raises(ProjectError):
        project.update_clip(clip["id"], status="maybe")


@pytest.mark.parametrize("start,end", [(3.0, 5.0), (0.0, 1.5), (7.5, 10.0)])
def test_export_cuts_exactly_at_boundaries(media, tmp_path, start, end):
    out = tmp_path / "reel.mp4"
    info = probe(media["timecode"])
    clip = ExportClip(str(media["timecode"]), start, end, True, info.width, info.height)
    result = export_reel([clip], out, layout="landscape")

    # Duration within one output frame (1/30 s) plus AAC padding slack.
    assert result["duration"] == pytest.approx(end - start, abs=0.05)
    # First frame comes from the start second, last frame from just before the end.
    # The clip is letterboxed onto 16:9, so sample the centre of the picture.
    first = frame_at(out, 0.0)
    last = frame_at(out, result["duration"] - 0.04)
    h, w = first.shape
    centre = (slice(h // 3, 2 * h // 3), slice(w // 3, 2 * w // 3))
    assert luma_second(first[centre].mean()) == int(start)
    assert luma_second(last[centre].mean()) == int(end - 0.04)
