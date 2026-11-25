"""Check an exported reel against the project it came from.

    python scripts/check_reel.py REEL.mp4 --workspace WORKSPACE [--json out.json]
    python scripts/check_reel.py REEL.mp4 --state STATE.json   # a saved /api/state, as e2e_ui.py writes

Checks, each printed as ok/FAIL:

* plays: FFmpeg decodes every frame without an error
* format: H.264 yuv420p video, AAC audio, moov atom up front (faststart)
* duration: the reel lasts as long as its clips added up (within 0.15 s)
* audio sync: the audio track is as long as the video (within 0.1 s)
* order and boundaries: frames just inside the start and end of each clip
  match the source at the clip's start and end time (within 0.15 s), found
  by comparing the reel frame with source frames around that time; a reel
  frame that looks like no nearby source frame fails (wrong clip or order)

Also reports black and silent stretches, which are expected in silent phone
clips but worth a look anywhere else. Exit code 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watch_the_goal.recap import TITLE_SECONDS
from watch_the_goal.project import Project  # noqa: E402
from watch_the_goal.vision.frames import sample_frames  # noqa: E402

INSET = 0.25  # seconds inside each clip edge
SEARCH = 0.6  # seconds either side of the expected source time
STEP = 1 / 30
# Mean grey-level difference (0-255, 96x54 thumbnails) above which a reel frame
# doesn't show that source at all. Re-encoding and scaling stay well under it.
MISMATCH = 15.0


def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def faststart(path: Path) -> bool:
    """True when the moov atom comes before mdat."""
    with open(path, "rb") as f:
        while header := f.read(8):
            if len(header) < 8:
                return False
            size, kind = int.from_bytes(header[:4], "big"), header[4:8]
            if kind in (b"moov", b"mdat"):
                return kind == b"moov"
            if size == 1:
                size = int.from_bytes(f.read(8), "big") - 8
            f.seek(size - 8, 1)
    return False


def small(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(img, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)


def foreground(reel_frame: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    """The part of a reel frame showing the source (the export centres it over a blurred fill)."""
    h, w = reel_frame.shape[:2]
    scale = min(w / src_w, h / src_h)
    fw, fh = int(src_w * scale), int(src_h * scale)
    x0, y0 = (w - fw) // 2, (h - fh) // 2
    return reel_frame[y0:y0 + fh, x0:x0 + fw]


def match_time(reel: Path, t_reel: float, source: str, expected: float,
               duration: float, crop: float | None = None, board: bool = False) -> tuple[float | None, bool, float]:
    """Source time whose frame best matches the reel at t_reel, whether the match
    is clear, and how different the best frame still is."""
    (_, rf), = sample_frames(str(reel), [t_reel])
    times = [t for t in np.arange(expected - SEARCH, expected + SEARCH + 1e-6, STEP) if 0 <= t < duration]
    diffs, ts = [], []
    for t, sf in sample_frames(source, times):
        if crop is not None and rf.shape[0] > rf.shape[1]:
            rh, rw = rf.shape[:2]
            sh, sw = sf.shape[:2]
            scale = max(rw / sw, rh / sh)
            cw, ch = round(rw / scale), round(rh / scale)
            x, y = round((sw - cw) * crop), (sh - ch) // 2
            sf = sf[y:y + ch, x:x + cw]
            fg = small(rf)
        else:
            fg = small(foreground(rf, sf.shape[1], sf.shape[0]))
        delta = np.abs(fg - small(sf))
        diffs.append(float((delta[9:] if board else delta).mean()))
        ts.append(t)
    if not diffs:
        return None, False, float("inf")
    i = int(np.argmin(diffs))
    clear = diffs[i] < 0.8 * float(np.median(diffs))  # a flat curve means a still shot: can't tell
    return float(ts[i]), bool(clear), diffs[i]


def audio_sample(path, start, seconds):
    proc = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.4f}", "-i", str(path),
                           "-t", f"{seconds:.4f}", "-map", "0:a:0", "-ac", "1", "-ar", "4000",
                           "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype="<f4")


def compare_audio(reel, offset, src, clip, checks, notes):
    """Compare source samples to the rendered track inside each clip, away from fades."""
    duration = clip["end"] - clip["start"]
    length = min(1.0, duration / 3)
    margin = min(.18, duration / 8)
    pos = (duration - length) / 2
    out = audio_sample(reel, offset + pos - margin, length + 2 * margin)
    label = f"audio content at {offset + pos:.2f}s ({src['name']})"
    if not src["info"]["has_audio"]:
        checks[label + " is silent"] = bool(out.size and np.max(np.abs(out)) < .003)
        return
    ref = audio_sample(src["path"], clip["start"] + pos, length)
    if not ref.size or len(out) < len(ref):
        notes.append(label + ": insufficient samples to judge")
        return
    ref = ref - ref.mean()
    if np.linalg.norm(ref) < .001:
        notes.append(label + ": quiet source, sync not judged")
        return
    # Normalized sliding correlation, allowing for AAC priming at the joins.
    energy = np.concatenate(([0.0], np.cumsum(out.astype(float) ** 2)))
    denom = np.sqrt(np.maximum(energy[len(ref):] - energy[:-len(ref)], 1e-15))
    corr = np.correlate(out, ref, mode="valid") / (np.linalg.norm(ref) * denom)
    peak = float(corr.max())
    matches = np.where(corr >= max(.55, peak * .97))[0]
    checks[label + " matches source"] = peak >= .55
    if len(matches):
        if (matches[-1] - matches[0]) / 4000 > .10:
            notes.append(label + ": repetitive sound, exact sync not judged")
        else:
            lag = float(np.argmax(corr)) / 4000 - margin
            checks[f"{label} offset {lag:+.3f}s"] = abs(lag) <= .10


def eval_rate(value: str) -> float:
    n, d = value.split("/")
    return float(n) / float(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reel", type=Path)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--workspace", type=Path, help="the app workspace the reel was exported from")
    where.add_argument("--state", type=Path, help="a saved copy of the app's /api/state for that reel")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.workspace:
        project = Project(args.workspace)
        sources = {s["id"]: s for s in project.sources}
        clips = project.reel()
        settings = project.settings
    else:
        st = json.loads(args.state.read_text())
        settings = st.get("settings", {})
        sources = {s["id"]: s for s in st["sources"]}
        clips = [c for c in st["clips"] if c["status"] == "kept"]  # state lists clips in reel order

    checks: dict[str, bool] = {}
    notes: list[str] = []
    dec = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(args.reel), "-f", "null", "-"],
                         capture_output=True, text=True)
    checks["plays: decodes without errors"] = dec.returncode == 0 and not dec.stderr.strip()

    info = ffprobe(args.reel)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    checks["format: H.264 yuv420p + AAC"] = (v["codec_name"] == "h264" and v.get("pix_fmt") == "yuv420p"
                                            and a is not None and a["codec_name"] == "aac")
    checks["format: faststart"] = faststart(args.reel)
    total = float(info["format"]["duration"])
    checks["upright output: rotation baked into pixels"] = not any(
        item.get("rotation", 0) % 360 for item in v.get("side_data_list", []))
    checks["stream format: 30 fps, 48 kHz stereo"] = (abs(eval_rate(v["r_frame_rate"]) - 30) < .01
        and a is not None and a["sample_rate"] == "48000" and a["channels"] == 2)
    intro = TITLE_SECONDS if settings.get("template") == "recap" and settings.get("title") else 0
    expected = intro + sum(c["end"] - c["start"] for c in clips)
    checks[f"duration: {total:.2f}s vs clips {expected:.2f}s"] = abs(total - expected) < 0.15
    if a is not None:
        va = abs(float(v["duration"]) - float(a["duration"]))
        checks[f"audio sync: audio and video lengths differ by {va:.3f}s"] = va < 0.1

    offset = float(intro)
    rows = []
    for i, c in enumerate(clips, 1):
        s = sources[c["source_id"]]
        d = c["end"] - c["start"]
        for edge, t_reel, t_src in (("start", offset + INSET, c["start"] + INSET),
                                    ("end", offset + d - INSET, c["end"] - INSET)):
            got, clear, diff = match_time(args.reel, t_reel, s["path"], t_src, s["info"]["duration"],
                                          c.get("crop"), settings.get("template") == "recap" and any(s.get("scoreboard", {}).values()))
            row = {"clip": i, "source": s["name"], "edge": edge, "reel_t": round(t_reel, 2),
                   "expected_source_t": round(t_src, 2), "matched_source_t": None if got is None else round(got, 2),
                   "clear": clear, "difference": round(diff, 1)}
            rows.append(row)
            name = f"clip {i} {edge}: reel {t_reel:.2f}s shows {s['name']} at {t_src:.2f}s"
            if diff > MISMATCH:
                checks[f"{name} (no frame within {SEARCH}s looks like it)"] = False
                continue
            if got is None or not clear:
                notes.append(f"{name}: no clear frame match (still shot?), not judged")
                continue
            checks[f"{name} (matched {got:.2f}s)"] = bool(abs(got - t_src) <= 0.15)
        if a is not None:
            compare_audio(args.reel, offset, s, c, checks, notes)
        offset += d

    picture_log = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(args.reel), "-vf", "scale=320:-2,blackdetect=d=0.5:pix_th=0.10,freezedetect=n=-55dB:d=1", "-an", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    black = re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", picture_log)
    for frozen in re.findall(r"freeze_start: ([\d.]+)", picture_log):
        notes.append(f"review frozen picture starting at {float(frozen):.2f}s")
    silence = re.findall(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(args.reel), "-af", "silencedetect=noise=-50dB:d=1.5", "-vn", "-f", "null", "-"],
        capture_output=True, text=True).stderr)
    for b0, b1 in black:
        notes.append(f"black video {float(b0):.2f}-{float(b1):.2f}s")
    for end, dur in silence:
        notes.append(f"silence {float(end) - float(dur):.2f}-{float(end):.2f}s")

    print(f"{args.reel}: {total:.2f}s, {v['width']}x{v['height']} @ {v.get('r_frame_rate')}, {len(clips)} clips")
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    for n in notes:
        print(f"  [note] {n}")
    if args.json:
        args.json.write_text(json.dumps({"reel": str(args.reel), "checks": checks, "notes": notes, "edges": rows},
                                        indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
