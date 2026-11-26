from watch_the_goal.detect import Moment, fit_window, merge_overlapping
from watch_the_goal.media import probe
from watch_the_goal.project import Project


def m(start, end, reasons, anchor=None):
    return Moment(anchor if anchor is not None else start + 8, start, end, list(reasons), ["audio"], 1.0)


def test_overlapping_moments_merge_into_one():
    merged = merge_overlapping([m(10, 22, ["crowd noise"]), m(16, 24, ["motion near the marked left goal"])])
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (10, 24)
    assert merged[0].reasons == ["crowd noise", "motion near the marked left goal"]
    assert merged[0].anchor == 18


def test_merged_moments_stay_short_clips():
    # The later moment runs well past 15 s: it continues as its own clip.
    merged = merge_overlapping([m(10, 22, ["a"]), m(18, 30, ["b"])])
    assert [(x.start, x.end, x.reasons) for x in merged] == [(10, 22, ["a"]), (22, 30, ["b"])]
    # Only just past: one clip, cut at 15 s.
    merged = merge_overlapping([m(10, 22, ["a"]), m(18, 26, ["b"])])
    assert [(x.start, x.end) for x in merged] == [(10, 25)]


def test_fit_window_keeps_clips_between_5_and_15_seconds():
    assert fit_window(90, 112, 100, 600) == (90, 105)  # 10 s buildup kept, reaction cut
    assert fit_window(80, 105, 100, 600) == (90, 105)  # buildup past 10 s goes first
    assert fit_window(95, 108, 100, 600) == (95, 108)  # short enough already
    assert fit_window(99, 101, 100, 600) == (99, 104)  # at least 5 s
    assert fit_window(590, 598, 597, 600) == (590, 598)


def test_nearly_touching_moments_merge_but_separate_ones_do_not():
    merged = merge_overlapping([m(0, 7, ["a"]), m(7.5, 14, ["b"]), m(40, 50, ["c"])], gap=1.0)
    assert [(x.start, x.end) for x in merged] == [(0, 14), (40, 50)]


def test_merge_does_not_duplicate_reasons_and_is_order_independent():
    a = merge_overlapping([m(5, 15, ["x"]), m(0, 8, ["x", "y"])])
    assert len(a) == 1 and a[0].reasons == ["x", "y"] and a[0].start == 0
    b = merge_overlapping([m(0, 10, ["crowd noise +15 dB above the surrounding level for 4.2s"]),
                           m(8, 14, ["crowd noise +10 dB above the surrounding level for 0.8s", "play stopped"])])
    assert b[0].reasons == ["crowd noise +15 dB above the surrounding level for 4.2s", "play stopped"]


def test_reanalysis_replaces_suggestions_but_keeps_user_decisions(media, tmp_path):
    project = Project(tmp_path / "ws")
    src = project.add_source(probe(media["long"]), "long.mp4")  # 130 s, a "long recording"
    project.apply_detection(src["id"], [m(10, 22, ["a"]), m(18, 24, ["b"]), m(60, 72, ["c"])])
    clips = [c for c in project.order]
    assert len(clips) == 2  # the first two overlapped and merged
    first, second = (project.clip(c) for c in clips)
    assert first["reasons"] == ["a", "b"] and first["status"] == "candidate"

    project.update_clip(first["id"], status="kept")
    # A fresh analysis finds the kept moment again plus a new one.
    project.apply_detection(src["id"], [m(12, 24, ["a"]), m(100, 112, ["d"])])
    clips = [project.clip(c) for c in project.order]
    assert [(c["start"], c["status"]) for c in clips] == [(10, "kept"), (100, "candidate")]


def test_older_projects_get_short_clips_and_rolls(media, tmp_path):
    project = Project(tmp_path / "ws")
    src = project.add_source(probe(media["long"]), "long.mp4")
    project.apply_detection(src["id"], [m(60, 72, ["a"])])
    kept = project.clip(project.order[0])
    project.update_clip(kept["id"], status="kept")
    trimmed = project.add_manual_clip(src["id"], 90, 115)
    # As saved before clips were capped: a 25 s suggestion and the old 12 s / 5 s rolls.
    kept.update(start=46, end=74, anchor=68)
    project.settings.update(pre_roll=12.0, post_roll=5.0)
    project.save()

    project = Project(tmp_path / "ws")
    assert (project.clip(kept["id"])["start"], project.clip(kept["id"])["end"]) == (58, 73)
    assert project.clip(kept["id"])["status"] == "kept"
    assert (project.clip(trimmed["id"])["start"], project.clip(trimmed["id"])["end"]) == (90, 115)  # the user's own
    assert (project.settings["pre_roll"], project.settings["post_roll"]) == (10.0, 4.0)
