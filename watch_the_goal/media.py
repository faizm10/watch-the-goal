"""Thin wrappers around ffprobe/ffmpeg for reading source footage.

Source files are only ever opened for reading.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp"}
# Codecs every mainstream browser can preview directly from an MP4/MOV.
BROWSER_SAFE_CODECS = {"h264"}


class MediaError(Exception):
    """A problem with a media file, phrased so it can be shown to the user."""


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise MediaError(
            f"{' and '.join(missing)} not found on PATH. Install FFmpeg "
            "(macOS: `brew install ffmpeg`, Ubuntu: `sudo apt install ffmpeg`) and try again."
        )


@dataclass
class MediaInfo:
    path: str
    duration: float
    width: int  # display width, after applying rotation
    height: int  # display height, after applying rotation
    rotation: int  # degrees clockwise the player must rotate the stored frames
    fps: float
    video_codec: str
    has_audio: bool
    audio_codec: str | None
    size_bytes: int
    recorded_at: float  # camera's creation_time if present, else file modified time (epoch s)

    @property
    def orientation(self) -> str:
        if self.width == self.height:
            return "square"
        return "portrait" if self.height > self.width else "landscape"

    @property
    def browser_safe(self) -> bool:
        return self.video_codec in BROWSER_SAFE_CODECS

    def to_dict(self) -> dict:
        data = asdict(self)
        data["orientation"] = self.orientation
        data["browser_safe"] = self.browser_safe
        return data


def _parse_rate(value: str | None) -> float:
    if not value or value in ("0/0", "0"):
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream: dict) -> int:
    """Return the clockwise display rotation in degrees (0, 90, 180, 270)."""
    rotation = 0.0
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            # Display matrix rotation is counter-clockwise; flip to clockwise.
            rotation = -float(side["rotation"])
            break
    else:
        tag = (stream.get("tags") or {}).get("rotate")
        if tag:
            rotation = float(tag)
    return int(round(rotation)) % 360


def probe(path: str | Path) -> MediaInfo:
    path = Path(path)
    if not path.exists():
        raise MediaError(f"File not found: {path}")
    if path.is_dir():
        raise MediaError(f"{path} is a folder, not a video file.")
    require_ffmpeg()
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        raise MediaError(f"Could not read {path.name}: {detail}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next(
        (s for s in streams if s.get("codec_type") == "video"
         and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    if video is None:
        raise MediaError(f"{path.name} has no video stream.")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    for candidate in (data.get("format", {}).get("duration"), video.get("duration")):
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue
    if duration <= 0:
        raise MediaError(f"{path.name} has no readable duration; it may be truncated or still recording.")

    rotation = _rotation(video)
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    if rotation in (90, 270):
        width, height = height, width
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    recorded_at = path.stat().st_mtime
    created = (data.get("format", {}).get("tags") or {}).get("creation_time")
    if created:
        try:
            recorded_at = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass

    return MediaInfo(
        path=str(path.resolve()),
        duration=round(duration, 3),
        width=width,
        height=height,
        rotation=rotation,
        fps=round(fps, 3),
        video_codec=video.get("codec_name", "unknown"),
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        size_bytes=path.stat().st_size,
        recorded_at=recorded_at,
    )


def extract_frame(path: str, t: float, width: int = 640) -> bytes:
    """Return one JPEG frame at time ``t`` (rotation applied)."""
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", path,
            "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2pipe",
            "-vcodec", "mjpeg", "-q:v", "4", "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise MediaError(f"Could not read a frame at {t:.1f}s.")
    return proc.stdout


def make_preview_proxy(path: str, out: Path, height: int = 480) -> None:
    """Transcode a browser-safe H.264 preview. Used when the source codec
    (e.g. HEVC from iPhones) may not play in every browser."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", path,
            "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "28", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart", str(tmp),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise MediaError(f"Could not build a preview: {proc.stderr.strip()[-300:]}")
    tmp.replace(out)
