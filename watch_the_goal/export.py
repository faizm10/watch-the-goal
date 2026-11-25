"""Render selected clips into one shareable MP4.

Each clip is cut and normalised on its own (rotation, size, frame rate, audio
format) so footage from different phones lines up, then the normalised pieces
are joined without a second re-encode.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .recap import CAPTION_EASE, END_SECONDS, TITLE_SECONDS, artwork, caption_window
from .media import MediaError, probe

OUTPUT_FPS = 30
SIZES = {"landscape": (1920, 1080), "portrait": (1080, 1920), "square": (1080, 1080)}


@dataclass
class ExportClip:
    path: str
    start: float
    end: float
    has_audio: bool
    width: int
    height: int
    label: str = ""

    crop: float | None = None  # horizontal position, 0..1; None fits the full shot
    board: str = ""
    caption: str = ""  # a lower third that slides in after the clip starts
    fade: float = 0.0  # seconds to fade in from and out to black

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class ExportError(Exception):
    pass


def choose_layout(clips: list[ExportClip], layout: str = "auto") -> tuple[int, int]:
    if layout in SIZES:
        return SIZES[layout]
    portrait = sum(c.duration for c in clips if c.height > c.width)
    landscape = sum(c.duration for c in clips if c.width >= c.height)
    return SIZES["portrait" if portrait > landscape else "landscape"]


def segment_command(clip: ExportClip, out: Path, size: tuple[int, int], board: Path | None = None,
                    caption: Path | None = None) -> list[str]:
    w, h = size
    dur = clip.duration
    # ffmpeg applies the source's rotation metadata while decoding, so every
    # clip reaches the filters upright. Clips whose shape differs from the
    # reel are fitted over a blurred fill instead of black bars.
    video = (
        f"[0:v]split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={w // 4}:{h // 4}:force_original_aspect_ratio=increase,"
        f"crop={w // 4}:{h // 4},boxblur=8:2,eq=brightness=-0.06,scale={w}:{h}[bg];"
        f"[fgsrc]scale={w}:{h}:force_original_aspect_ratio=decrease:force_divisible_by=2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={OUTPUT_FPS},format=yuv420p[v]"
    )
    if h > w and clip.crop is not None:
        video = (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase:force_divisible_by=2,"
                 f"crop={w}:{h}:(iw-ow)*{clip.crop:.4f}:(ih-oh)/2,setsar=1,fps={OUTPUT_FPS},format=yuv420p[v]")
    edge = min(max(clip.fade, 0.05), dur / 3)
    audio_in = "0:a:0" if clip.has_audio else "1:a"
    audio = (
        f"[{audio_in}]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,"
        f"atrim=0:{dur:.3f},afade=t=in:d={edge:.3f},afade=t=out:st={dur - max(edge, 0.12):.3f}:d={max(edge, 0.12):.3f}[a]"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-nostdin",
        "-ss", f"{clip.start:.3f}", "-t", f"{dur:.3f}", "-i", clip.path,
    ]
    if not clip.has_audio:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    index = 1 if clip.has_audio else 2
    video = video.replace("[v]", "[l0]")
    layer = 0
    if board:
        cmd += ["-i", str(board)]
        video += f";[l{layer}][{index}:v]overlay=0:0[l{layer + 1}]"
        index, layer = index + 1, layer + 1
    if caption:
        a, b = caption_window(dur)
        e = CAPTION_EASE
        cmd += ["-loop", "1", "-framerate", str(OUTPUT_FPS), "-t", f"{dur:.3f}", "-i", str(caption)]
        slide = round(size[0] * 0.04)
        video += (f";[{index}:v]format=rgba,fade=t=in:st={a:.3f}:d={e}:alpha=1,"
                  f"fade=t=out:st={b:.3f}:d={e}:alpha=1[cap];"
                  f"[l{layer}][cap]overlay=x='-{slide}*max(0,1-max(0,t-{a:.3f})/{e})':y=0:eof_action=pass[l{layer + 1}]")
        index, layer = index + 1, layer + 1
    if clip.fade:
        video += f";[l{layer}]fade=t=in:d={edge:.3f},fade=t=out:st={dur - edge:.3f}:d={edge:.3f}[v]"
    else:
        video += f";[l{layer}]null[v]"
    cmd += [
        "-filter_complex", f"{video};{audio}",
        "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-profile:v", "high",
        "-c:a", "aac", "-b:a", "160k", "-video_track_timescale", "30000",
        "-progress", "pipe:1", "-nostats",
        str(out),
    ]
    return cmd


def friendly_error(stderr: str, clip: ExportClip | None = None) -> str:
    text = stderr.strip()
    lower = text.lower()
    where = f" while rendering {clip.label or Path(clip.path).name}" if clip else ""
    if "no space left" in lower:
        return f"The disk is full{where}. Free some space and export again."
    if "permission denied" in lower:
        return f"Permission denied{where}. Check that the export folder is writable."
    if "no such file" in lower or "does not exist" in lower:
        return f"A source file is missing{where}. It may have been moved or deleted; re-import it."
    if "invalid data found" in lower or "moov atom not found" in lower:
        return f"The source video is damaged or incomplete{where}. Try re-copying it from the phone."
    last = text.splitlines()[-1] if text else "ffmpeg failed without a message"
    return f"Export failed{where}: {last}"


def _run_with_progress(cmd: list[str], duration: float, on_time: Callable[[float], None]) -> tuple[int, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and value.isdigit():
            on_time(min(duration, int(value) / 1e6))
    _, err = proc.communicate()
    return proc.returncode, err


def export_reel(
    clips: list[ExportClip],
    out_path: Path,
    layout: str = "auto",
    work_dir: Path | None = None,
    progress: Callable[[float, str], None] | None = None,
    title: str = "",
    end: str = "",
    fade: float = 0.0,
) -> dict:
    if not clips:
        raise ExportError("Nothing to export yet. Keep at least one clip.")
    if shutil.which("ffmpeg") is None:
        raise ExportError("ffmpeg is not installed or not on PATH.")
    for clip in clips:
        if not os.path.exists(clip.path):
            raise ExportError(
                f"Source file is missing: {clip.path}. It may have been moved; re-import it."
            )
        if clip.duration <= 0.05:
            raise ExportError(f"{clip.label or 'A clip'} is empty. Widen its start and end.")

    size = choose_layout(clips, layout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".segments-", dir=work_dir or out_path.parent))
    total = sum(c.duration for c in clips)
    total += (TITLE_SECONDS if title else 0) + (END_SECONDS if end else 0)
    done = 0.0
    report = progress or (lambda fraction, message: None)
    try:
        segments = []

        def card(kind: str, text: str, seconds: float) -> Path:
            png = work / f"{kind}.png"
            png.write_bytes(artwork(kind, size, text))
            seg = work / f"{kind}.mp4"
            # Loop the still at the output frame rate, with the same stream layout as every segment.
            cmd = segment_command(ExportClip(str(png), 0, seconds, False, *size, fade=fade), seg, size)
            cmd[cmd.index("-ss"):cmd.index("-ss")] = ["-loop", "1", "-framerate", str(OUTPUT_FPS)]
            code, err = _run_with_progress(cmd, seconds, lambda _: None)
            if code:
                raise ExportError(friendly_error(err))
            return seg

        if title:
            report(0.0, "Rendering the opening title")
            segments.append(card("title", title, TITLE_SECONDS))
            done += TITLE_SECONDS
        for i, clip in enumerate(clips):
            seg = work / f"{i:03d}.mp4"
            label = clip.label or f"clip {i + 1}"
            report(done / total * 0.97, f"Rendering {label} ({i + 1}/{len(clips)})")

            def on_time(t: float, base=done, label=label, i=i) -> None:
                report((base + t) / total * 0.97, f"Rendering {label} ({i + 1}/{len(clips)})")

            board = caption = None
            if clip.board:
                board = work / "scoreboard.png"
                board.write_bytes(artwork("board", size, clip.board))
            if clip.caption:
                caption = work / "caption.png"
                caption.write_bytes(artwork("caption", size, clip.caption))
            code, err = _run_with_progress(segment_command(clip, seg, size, board, caption), clip.duration, on_time)
            if code != 0:
                raise ExportError(friendly_error(err, clip))
            segments.append(seg)
            done += clip.duration
        if end:
            report(done / total * 0.97, "Rendering the end card")
            segments.append(card("end", end, END_SECONDS))
            done += END_SECONDS

        report(0.98, "Joining clips")
        listing = work / "list.txt"
        listing.write_text("".join(f"file '{s.name}'\n" for s in segments))
        tmp_out = out_path.with_name(out_path.stem + ".part.mp4")
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-nostdin", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(tmp_out)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tmp_out.unlink(missing_ok=True)
            raise ExportError(friendly_error(proc.stderr))
        tmp_out.replace(out_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    try:
        info = probe(out_path)
    except MediaError as exc:
        raise ExportError(f"The reel was written but cannot be read back: {exc}") from exc
    report(1.0, "Done")
    return {
        "path": str(out_path),
        "duration": info.duration,
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "clips": len(clips),
        "expected_duration": round(total, 3),
    }
