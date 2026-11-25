from watch_the_goal.detect import Moment, merge_overlapping
from watch_the_goal.media import probe
from watch_the_goal.project import Project


def m(start, end, reasons, anchor=None):
    return Moment(anchor if anchor is not None else start + 12, start, end, list(reasons), ["audio"], 1.0)


def test_overlapping_moments_merge_into_one():
    merged = merge_overlapping([m(10, 28, ["crowd noise"]), m(20, 40, ["motion near the marked left goal"])])
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (10, 40)
    assert merged[0].reasons == ["crowd noise", "motion near the marked left goal"]
    assert merged[0].anchor == 22


def test_nearly_touching_moments_merge_but_separate_ones_do_not():
    merged = merge_overlapping([m(0, 10, ["a"]), m(10.5, 20, ["b"]), m(40, 50, ["c"])], gap=1.0)
    assert [(x.start, x.end) for x in merged] == [(0, 20), (40, 50)]


def test_merge_does_not_duplicate_reasons_and_is_order_independent():
    a = merge_overlapping([m(5, 15, ["x"]), m(0, 8, ["x", "y"])])
    assert len(a) == 1 and a[0].reasons == ["x", "y"] and a[0].start == 0
    b = merge_overlapping([m(0, 10, ["crowd noise +15 dB above the surrounding level for 4.2s"]),
                           m(8, 20, ["crowd noise +10 dB above the surrounding level for 0.8s", "play stopped"])])
    assert b[0].reasons == ["crowd noise +15 dB above the surrounding level for 4.2s", "play stopped"]


def test_reanalysis_replaces_suggestions_but_keeps_user_decisions(media, tmp_path):
    project = Project(tmp_path / "ws")
    src = project.add_source(probe(media["long"]), "long.mp4")  # 130 s, a "long recording"
    project.apply_detection(src["id"], [m(10, 28, ["a"]), m(20, 35, ["b"]), m(60, 78, ["c"])])
    clips = [c for c in project.order]
    assert len(clips) == 2  # the first two overlapped and merged
    first, second = (project.clip(c) for c in clips)
    assert first["reasons"] == ["a", "b"] and first["status"] == "candidate"

    project.update_clip(first["id"], status="kept")
    # A fresh analysis finds the kept moment again plus a new one.
    project.apply_detection(src["id"], [m(12, 30, ["a"]), m(100, 118, ["d"])])
    clips = [project.clip(c) for c in project.order]
    assert [(c["start"], c["status"]) for c in clips] == [(10, "kept"), (100, "candidate")]
