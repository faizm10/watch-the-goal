"""Optional computer vision on top of the baseline audio/motion detector.

* :mod:`.frames`     sample frames with OpenCV (rotation applied)
* :mod:`.models`     pinned, opt-in model checkpoints and where they live
* :mod:`.detectors`  frame-level person detectors behind one interface
* :mod:`.players`    turn detections into per-moment player evidence
* :mod:`.overlay`    debug images of what a detector saw

Person detection only produces boxes per frame. Deciding whether a moment is a
highlight stays in :mod:`watch_the_goal.detect`.
"""
