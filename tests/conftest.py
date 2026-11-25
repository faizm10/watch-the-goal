"""Small synthetic fixtures built with ffmpeg's test sources (a few KB each).

Frames encode time as brightness: luma = BASE + 20 * whole seconds, so a
decoded frame tells us which second of the source it came from.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    pytest.skip("ffmpeg/ffprobe not installed", allow_module_level=True)


def run(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def make_video(
    path: Path,
    duration: float,
    size: str = "320x240",
    fps: int = 30,
    base: int = 16,
    audio: bool = True,
    rotate: int = 0,
    codec: str = "libx264",
    marker: bool = False,
    tone: int = 440,
) -> Path:
    """Write a test video. ``marker`` draws a white bar across the top of the
    *displayed* frame, so tests can check rotation was applied."""
    w, h = (int(v) for v in size.split("x"))
    video = (
        f"color=c=black:s={w}x{h}:r={fps}:d={duration},"
        f"geq=lum='min(235,{base}+20*floor(T))':cb=128:cr=128"
    )
    if marker:
        video += ",drawbox=x=0:y=0:w=iw:h=ih/8:color=white:t=fill"
    if rotate:
        # Store the frames turned sideways; the rotation tag turns them back.
        video += ",transpose=2" if rotate == 90 else ",transpose=1"
    args = ["-f", "lavfi", "-i", video]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=44100:duration={duration}"]
    args += ["-c:v", codec, "-pix_fmt", "yuv420p", "-g", str(fps)]
    if codec == "libx265":
        args += ["-tag:v", "hvc1", "-x265-params", "log-level=error"]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    if rotate:
        tmp = path.with_name(path.stem + "_raw" + path.suffix)
        run(*args, str(tmp))
        # -display_rotation is counter-clockwise; phones store "rotate 90" (clockwise).
        run("-display_rotation", str(-rotate), "-i", str(tmp), "-c", "copy", str(path))
        tmp.unlink()
    else:
        run(*args, str(path))
    return path


def frame_at(path: Path, t: float) -> np.ndarray:
    """Decode the luma plane of one frame at ``t`` seconds (limited range, as encoded)."""
    from watch_the_goal.media import probe

    info = probe(path)
    w, h = info.width, info.height
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.uint8)[: w * h].reshape(h, w)


def luma_second(value: float, base: int = 16) -> int:
    """Invert the brightness code back to a whole second."""
    return int(round((value - base) / 20))


@pytest.fixture(scope="session")
def media(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("media")
    has_x265 = "libx265" in subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return {
        "timecode": make_video(d / "timecode.mp4", 10, base=16),
        # Mixed phone formats: portrait MOV with rotation metadata (HEVC when
        # available, like an iPhone), 60 fps landscape, and one with no audio.
        "portrait": make_video(d / "portrait.mov", 4, size="240x426", base=40, rotate=90,
                               codec="libx265" if has_x265 else "libx264", marker=True),
        "sixty": make_video(d / "sixty.mp4", 3, size="480x270", fps=60, base=120, tone=660),
        "silent": make_video(d / "silent.mp4", 3, size="320x240", base=200, audio=False),
        "long": make_video(d / "long.mp4", 130, size="160x90", fps=5, base=16),
    }
