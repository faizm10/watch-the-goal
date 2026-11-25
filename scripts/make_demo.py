"""Generate synthetic, redistributable demo footage.

Renders a simulated 5-a-side pickup game from a fixed, wide tripod angle, plus
three "phone clips" of moments from the same game shot from other spots:

  sunday_match_wide.mp4      6 min, 1280x720 25 fps, H.264 + audio (the tripod)
  phone_goal_portrait.mov    portrait, stored landscape with rotation metadata, HEVC + audio
  phone_goal_60fps.mp4       1920x1080 60 fps, H.264 + audio
  phone_chance_noaudio.mp4   1280x720 30 fps, H.264, no audio track

The script also writes labels.json with the ground truth (goal times, the
decoy shout and the near miss) so detection can be scored.

Everything is drawn from code with a fixed seed: no real people, no real
recordings, safe to share.

    python scripts/make_demo.py [--out demo/footage] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PITCH_W, PITCH_H = 60.0, 36.0  # metres
GOAL_HALF = 2.2  # half the goal mouth (pickup goals are small)
SIM_FPS = 25
SR = 48000

# Ground truth for the generated match (seconds).
GOALS = [
    {"t": 70.0, "team": "blue", "side": "right"},
    {"t": 176.0, "team": "red", "side": "left"},
    {"t": 296.0, "team": "blue", "side": "right"},
]
DECOY_SHOUT = 128.0  # a loud argument, no goal
NEAR_MISS = 236.0  # a shot that goes wide of the left goal, no cheer


# ---------------------------------------------------------------------------
# Simulation: a scripted ball path; players follow it.


@dataclass
class Key:
    t: float
    x: float
    y: float
    team: str  # team in possession after this keyframe
    mode: str  # "carry" | "pass" | "shot" | "dead"


def build_script(duration: float, rng: np.random.Generator) -> list[Key]:
    keys: list[Key] = [Key(0.0, 30.0, 18.0, "blue", "carry")]

    def free_play(until: float) -> None:
        t, team = keys[-1].t, keys[-1].team
        x, y = keys[-1].x, keys[-1].y
        while t < until - 3.5:
            dt = float(rng.uniform(1.2, 3.2))
            mode = "pass" if rng.random() < 0.55 else "carry"
            dist = rng.uniform(6, 16) if mode == "pass" else rng.uniform(2, 6)
            ang = rng.uniform(0, 2 * math.pi)
            x = float(np.clip(x + dist * math.cos(ang), 12, PITCH_W - 12))
            y = float(np.clip(y + dist * math.sin(ang), 4, PITCH_H - 4))
            t += dt
            if rng.random() < 0.18:
                team = "red" if team == "blue" else "blue"
            keys.append(Key(t, x, y, team, mode))
        # Drift back to midfield before a scripted moment.
        keys.append(Key(until, 30 + float(rng.uniform(-3, 3)), 18 + float(rng.uniform(-4, 4)), keys[-1].team, "carry"))

    def attack(t_goal: float, team: str, side: str, scores: bool) -> None:
        gx = PITCH_W if side == "right" else 0.0
        sign = 1 if side == "right" else -1
        defending = "red" if team == "blue" else "blue"
        # Defending team has it in midfield, loses it (turnover), then a run,
        # an assist across, and the finish.
        free_play(t_goal - 13)
        keys.append(Key(t_goal - 11.0, 30 - 6 * sign, 14.0, defending, "carry"))
        keys.append(Key(t_goal - 9.0, 30 - 3 * sign, 16.0, team, "carry"))  # turnover
        keys.append(Key(t_goal - 6.0, 30 + 10 * sign, 10.0, team, "carry"))
        keys.append(Key(t_goal - 4.2, 30 + 14 * sign, 7.0, team, "pass"))  # the assist
        keys.append(Key(t_goal - 2.2, gx - 9 * sign, 21.0, team, "carry"))
        keys.append(Key(t_goal - 1.0, gx - 7 * sign, 20.0, team, "shot"))
        if scores:
            keys.append(Key(t_goal, gx + 0.8 * sign, 18.3, team, "dead"))
            keys.append(Key(t_goal + 9, gx + 0.8 * sign, 18.3, team, "dead"))
            keys.append(Key(t_goal + 14, 30.0, 18.0, team, "dead"))
            keys.append(Key(t_goal + 22, 30.0, 18.0, defending, "carry"))  # kickoff
        else:
            keys.append(Key(t_goal, gx + 2.5 * sign, 18 + 6.5, team, "dead"))  # wide
            keys.append(Key(t_goal + 5, gx - 0.5 * sign, 26.0, team, "dead"))
            keys.append(Key(t_goal + 9, gx - 3 * sign, 18.0, defending, "carry"))  # goal kick

    events = [(g["t"], g["team"], g["side"], True) for g in GOALS]
    events.append((NEAR_MISS, "red", "left", False))
    for t, team, side, scores in sorted(events):
        attack(t, team, side, scores)
    free_play(duration + 4)
    return keys


class Match:
    """Frame-by-frame positions of ball and players at SIM_FPS."""

    def __init__(self, duration: float, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.duration = duration
        self.keys = build_script(duration, rng)
        n = int(duration * SIM_FPS) + 2
        self.ball = np.zeros((n, 3))  # x, y, height
        self.players = np.zeros((n, 10, 2))
        self.team = np.array(["blue"] * 5 + ["red"] * 5)
        home_blue = np.array([[4, 18], [16, 9], [16, 27], [26, 13], [26, 23]], float)
        home_red = home_blue.copy()
        home_red[:, 0] = PITCH_W - home_red[:, 0]
        self.home = np.concatenate([home_blue, home_red])
        pos = self.home.copy()
        wander = rng.normal(0, 1, (10, 2))
        goals_by_time = {g["t"]: g for g in GOALS}
        ki = 0
        for f in range(n):
            t = f / SIM_FPS
            while ki + 1 < len(self.keys) - 1 and self.keys[ki + 1].t <= t:
                ki += 1
            k0, k1 = self.keys[ki], self.keys[min(ki + 1, len(self.keys) - 1)]
            u = 0.0 if k1.t == k0.t else min(1.0, (t - k0.t) / (k1.t - k0.t))
            if k0.mode in ("pass", "shot"):
                u = 1 - (1 - u) ** 2  # ball slows as it rolls
            bx, by = k0.x + (k1.x - k0.x) * u, k0.y + (k1.y - k0.y) * u
            h = 1.2 * math.sin(math.pi * u) if k0.mode == "shot" else 0.0
            self.ball[f] = (bx, by, h)

            # Celebration / reset phases around goals.
            phase, scorer = None, None
            for gt, g in goals_by_time.items():
                if gt <= t < gt + 8:
                    phase, scorer = "celebrate", g
                elif gt + 8 <= t < gt + 22:
                    phase, scorer = "reset", g

            wander += rng.normal(0, 0.15, (10, 2))
            wander *= 0.97
            ball = np.array([bx, by])
            targets = np.empty((10, 2))
            speed = np.full(10, 6.5)
            attacking_shift = np.where(self.team == k0.team, 6.0, -4.0) * np.where(self.team == "blue", 1, -1)
            for i in range(10):
                home = self.home[i] + np.array([attacking_shift[i], 0])
                targets[i] = home + 0.45 * (ball - home) + wander[i] * 2.5
            if phase is None:
                for team in ("blue", "red"):
                    idx = np.flatnonzero(self.team == team)
                    d = np.linalg.norm(pos[idx] - ball, axis=1)
                    near = idx[np.argsort(d)[:2]]
                    if team == k0.team and k0.mode == "carry":
                        targets[near[0]] = ball - np.array([0.6, 0.3])
                        speed[near[0]] = 9.0
                    elif team == k1.team and k0.mode == "pass":
                        targets[near[0]] = np.array([k1.x, k1.y])
                        speed[near[0]] = 7.5
                    else:
                        targets[near[0]] = ball
                        speed[near[0]] = 6.0
            elif phase == "celebrate":
                corner = np.array([PITCH_W - 3 if scorer["side"] == "right" else 3, 33.0])
                for i in np.flatnonzero(self.team == scorer["team"]):
                    if i not in (0, 5):  # keepers stay home
                        targets[i] = corner + wander[i] * 1.2
                        speed[i] = 7.0
                speed[self.team != scorer["team"]] = 1.2
            else:  # walk back slowly for the restart
                targets = self.home + wander * 0.8
                speed[:] = 1.4
            step = targets - pos
            dist = np.linalg.norm(step, axis=1, keepdims=True)
            pos = pos + step / np.maximum(dist, 1e-6) * np.minimum(dist, speed[:, None] / SIM_FPS)
            pos[:, 0] = np.clip(pos[:, 0], 0.5, PITCH_W - 0.5)
            pos[:, 1] = np.clip(pos[:, 1], 0.5, PITCH_H - 0.5)
            self.players[f] = pos

    def at(self, t: float):
        f = t * SIM_FPS
        a = int(np.clip(math.floor(f), 0, len(self.ball) - 2))
        u = f - a
        return (self.ball[a] * (1 - u) + self.ball[a + 1] * u,
                self.players[a] * (1 - u) + self.players[a + 1] * u)


# ---------------------------------------------------------------------------
# Rendering


COLORS = {"blue": (40, 110, 230), "red": (225, 60, 50)}


class Camera:
    """Projects pitch coordinates (metres) to pixels with a simple elevated-sideline look."""

    def __init__(self, w: int, h: int, center, span: float, tilt: float = 0.55):
        self.w, self.h = w, h
        self.cx, self.cy = center
        self.span = span  # metres visible across the frame width
        self.tilt = tilt

    def project(self, x, y):
        s = self.w / self.span
        px = (np.asarray(x) - self.cx) * s + self.w / 2
        py = (np.asarray(y) - self.cy) * s * self.tilt + self.h / 2
        return px, py, s


def render_pitch(cam: Camera, t: float = 0.0) -> np.ndarray:
    h, w = cam.h, cam.w
    ys, xs = np.mgrid[0:h, 0:w]
    s = w / cam.span
    mx = (xs - w / 2) / s + cam.cx
    my = (ys - h / 2) / (s * cam.tilt) + cam.cy
    img = np.zeros((h, w, 3), np.float32)
    stripes = (np.floor(mx / 5) % 2).astype(np.float32)
    grass = np.array([52, 128, 60], np.float32)
    img[:] = grass
    img += (stripes[..., None] * np.array([10, 14, 8], np.float32))
    noise = np.random.default_rng(1).normal(0, 4, (h, w, 1)).astype(np.float32)
    img += noise
    outside = (mx < -3) | (mx > PITCH_W + 3) | (my < -2) | (my > PITCH_H + 2)
    img[outside] = img[outside] * 0.75 + np.array([20, 30, 15]) * 0.25
    lw = 0.18
    lines = (
        ((np.abs(mx) < lw) | (np.abs(mx - PITCH_W) < lw) | (np.abs(mx - PITCH_W / 2) < lw))
        & (my > -lw) & (my < PITCH_H + lw)
    ) | (((np.abs(my) < lw) | (np.abs(my - PITCH_H) < lw)) & (mx > -lw) & (mx < PITCH_W + lw))
    circle = np.abs(np.hypot(mx - PITCH_W / 2, my - PITCH_H / 2) - 5) < lw
    boxes = np.zeros_like(lines)
    for gx, d in ((0, 1), (PITCH_W, -1)):
        inside_x = (mx - gx) * d
        boxes |= ((np.abs(inside_x - 7) < lw) & (np.abs(my - PITCH_H / 2) < 7)) | (
            (np.abs(np.abs(my - PITCH_H / 2) - 7) < lw) & (inside_x > 0) & (inside_x < 7))
    img[lines | circle | boxes] = (235, 240, 235)
    # Sky / treeline band for wide shots.
    top = my < -6
    img[top] = np.array([150, 185, 215], np.float32)
    trees = (my >= -6) & (my < -3)
    img[trees] = np.array([30, 70, 40], np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_disc(img, cx, cy, r, color, alpha=1.0):
    h, w = img.shape[:2]
    x0, x1 = int(max(0, cx - r - 1)), int(min(w, cx + r + 2))
    y0, y1 = int(max(0, cy - r - 1)), int(min(h, cy + r + 2))
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    cov = np.clip(r + 0.5 - np.hypot(xs - cx, ys - cy), 0, 1)[..., None] * alpha
    patch = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (patch * (1 - cov) + np.array(color, np.float32) * cov).astype(np.uint8)


def draw_ellipse(img, cx, cy, rx, ry, color, alpha):
    h, w = img.shape[:2]
    x0, x1 = int(max(0, cx - rx - 1)), int(min(w, cx + rx + 2))
    y0, y1 = int(max(0, cy - ry - 1)), int(min(h, cy + ry + 2))
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = ((xs - cx) / max(rx, 0.5)) ** 2 + ((ys - cy) / max(ry, 0.5)) ** 2 <= 1
    patch = img[y0:y1, x0:x1].astype(np.float32)
    patch[inside] = patch[inside] * (1 - alpha) + np.array(color, np.float32) * alpha
    img[y0:y1, x0:x1] = patch.astype(np.uint8)


def draw_goal(img, cam: Camera, side: str, shake: float):
    gx = 0.0 if side == "left" else PITCH_W
    depth = -1.6 if side == "left" else 1.6
    px, py, s = cam.project(np.array([gx, gx, gx + depth, gx + depth]),
                            np.array([PITCH_H / 2 - GOAL_HALF, PITCH_H / 2 + GOAL_HALF,
                                      PITCH_H / 2 - GOAL_HALF, PITCH_H / 2 + GOAL_HALF]))
    height = 2.0 * s * 0.9
    # Net: a lattice behind the goal mouth, wobbling after a goal.
    x_lo, x_hi = int(min(px)), int(max(px))
    y_lo, y_hi = int(min(py) - height), int(max(py))
    h, w = img.shape[:2]
    x_lo, x_hi, y_lo, y_hi = max(0, x_lo), min(w, x_hi + 1), max(0, y_lo), min(h, y_hi + 1)
    if x_lo < x_hi and y_lo < y_hi:
        ys, xs = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        spacing = max(3.0, s * 0.25)
        wob = shake * np.sin(ys / 3.0 + xs / 5.0) * 3
        net = ((np.mod(xs + wob, spacing) < 1) | (np.mod(ys + wob * 0.6, spacing) < 1))
        patch = img[y_lo:y_hi, x_lo:x_hi]
        patch[net] = (patch[net] * 0.35 + 255 * 0.65).astype(np.uint8)
    # Posts and bar.
    for i in (0, 1):
        x, y = px[i], py[i]
        for yy in range(int(y - height), int(y) + 1):
            draw_disc(img, x, yy, max(1.0, s * 0.07), (250, 250, 250))
    for xx in np.linspace(px[0], px[1], int(abs(px[1] - px[0]) + 2)):
        yy = np.interp(xx, [px[0], px[1]], [py[0], py[1]]) - height
        draw_disc(img, xx, yy, max(1.0, s * 0.07), (250, 250, 250))


def render_frame(base, cam: Camera, match: Match, t: float, label: str | None) -> np.ndarray:
    img = base.copy()
    ball, players = match.at(t)
    shake = {"left": 0.0, "right": 0.0}
    for g in GOALS:
        if 0 <= t - g["t"] < 2.5:
            shake[g["side"]] = (1 - (t - g["t"]) / 2.5) * 2.5
    for side in ("left", "right"):
        draw_goal(img, cam, side, shake[side])
    order = np.argsort(players[:, 1])
    px, py, s = cam.project(players[:, 0], players[:, 1])
    for i in order:
        r = max(2.0, s * 0.45)
        draw_ellipse(img, px[i] + r * 0.4, py[i] + r * 0.2, r * 1.1, r * 0.45, (20, 50, 20), 0.45)
        color = COLORS[match.team[i]]
        if i in (0, 5):
            color = (240, 200, 40)  # keepers in yellow
        draw_disc(img, px[i], py[i] - r * 1.6, r, color)
        draw_disc(img, px[i], py[i] - r * 3.0, r * 0.62, (235, 190, 150))
    bx, by, s = cam.project(ball[0], ball[1])
    br = max(1.5, s * 0.16)
    draw_ellipse(img, bx, by, br, br * 0.45, (20, 50, 20), 0.5)
    draw_disc(img, bx, by - ball[2] * s * 0.6 - br, br, (250, 250, 250))
    if label:
        draw_label(img, label)
    return img


# A tiny 3x5 bitmap font, enough for a corner watermark.
FONT = {
    "A": "010101111101101", "C": "011100100100011", "D": "110101101101110", "E": "111100110100111",
    "H": "101101111101101", "I": "111010010010111", "M": "101111111101101", "N": "110101101101101",
    "O": "010101101101010", "S": "011100010001110", "T": "111010010010010", "Y": "101101010010010",
    "F": "111100110100100", "G": "011100101101011", "R": "110101110101101", "L": "100100100100111",
    "U": "101101101101111", "P": "110101110100100", "K": "101101110101101", "W": "101101111111101",
    " ": "000000000000000", "-": "000000111000000",
}


def draw_label(img, text: str, scale: int | None = None):
    h, w = img.shape[:2]
    scale = scale or max(2, w // 400)
    x, y = 12, h - 12 - 5 * scale
    for ch in text.upper():
        bits = FONT.get(ch, FONT[" "])
        for r in range(5):
            for c in range(3):
                if bits[r * 3 + c] == "1":
                    img[y + r * scale:y + (r + 1) * scale, x + c * scale:x + (c + 1) * scale] = (255, 255, 255)
        x += 4 * scale


# ---------------------------------------------------------------------------
# Audio


def audio_track(match: Match, start: float, duration: float, near: str | None, seed: int,
                level: float = 1.0) -> np.ndarray:
    """Ambient wind/park noise, ball strikes, player calls, and cheers after goals."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    white = rng.normal(0, 1, n).astype(np.float32)
    # Brown-ish wind: integrated noise, high-passed by subtracting a slow average.
    wind = np.cumsum(white) / 300
    wind -= np.convolve(wind, np.ones(4801) / 4801, mode="same")
    out = 0.012 * white + 0.06 * wind.astype(np.float32)

    def add(at: float, sig: np.ndarray):
        i = int((at - start) * SR)
        if i >= n or i + len(sig) <= 0:
            return
        a, b = max(i, 0), min(i + len(sig), n)
        out[a:b] += sig[a - i:b - i]

    def voice(dur, f0, gain, vib=5.0):
        tt = np.arange(int(dur * SR)) / SR
        f = f0 * (1 + 0.04 * np.sin(2 * np.pi * vib * tt) + rng.uniform(-0.05, 0.05))
        phase = 2 * np.pi * np.cumsum(f) / SR
        sig = sum(np.sin(k * phase) / k for k in range(1, 6))
        sig += 0.3 * rng.normal(0, 1, len(tt))
        env = np.minimum(1, tt / 0.05) * np.minimum(1, (dur - tt) / 0.25).clip(0)
        return (gain * sig * env).astype(np.float32)

    # Ball strikes at every pass and shot.
    for k in match.keys:
        if k.mode in ("pass", "shot") and start <= k.t < start + duration:
            tt = np.arange(int(0.08 * SR)) / SR
            thump = np.sin(2 * np.pi * 140 * tt) * np.exp(-tt * 60) * (0.35 if k.mode == "shot" else 0.15)
            add(k.t, thump.astype(np.float32))
    # Occasional calls ("man on!", "here!").
    t = start
    while t < start + duration:
        t += float(rng.uniform(5, 12))
        add(t, voice(float(rng.uniform(0.25, 0.5)), float(rng.uniform(180, 320)), 0.05))
    # Cheers: several voices, loud, a few seconds.
    for g in GOALS:
        for _ in range(7):
            add(g["t"] + float(rng.uniform(0.1, 0.6)),
                voice(float(rng.uniform(1.8, 3.6)), float(rng.uniform(160, 420)), 0.16))
    # Decoy: a loud argument with no goal.
    for _ in range(3):
        add(DECOY_SHOUT + float(rng.uniform(0, 0.8)), voice(float(rng.uniform(1.2, 2.0)), float(rng.uniform(150, 260)), 0.2))
    # Near miss: a groan, quieter than a cheer.
    add(NEAR_MISS + 0.3, voice(0.9, 140, 0.06, vib=2))
    out *= level
    stereo = np.stack([out, np.roll(out, 37)], axis=1)
    return np.clip(stereo, -1, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Writers


KICKOFF = "2026-05-17T10:00:00Z"  # when the tripod started recording


def stamp(offset: float) -> str:
    """Camera-style creation_time, ``offset`` seconds after kickoff."""
    from datetime import datetime, timedelta

    start = datetime.fromisoformat(KICKOFF.replace("Z", "+00:00"))
    return (start + timedelta(seconds=offset)).strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def write_video(path: Path, frames, w: int, h: int, fps: float, audio: np.ndarray | None,
                codec: str = "libx264", rotate_display: int = 0, created: str = KICKOFF) -> None:
    """Pipe RGB frames to ffmpeg. ``rotate_display`` stores frames rotated and
    tags the stream so players turn them upright (like a phone held sideways)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if audio is not None:
            wav = tmp / "a.f32"
            audio.astype("<f4").tofile(wav)
            cmd += ["-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(wav)]
        vcodec = ["-c:v", codec, "-preset", "fast", "-pix_fmt", "yuv420p"]
        if codec == "libx264":
            vcodec += ["-crf", "21"]
        else:
            vcodec += ["-crf", "26", "-tag:v", "hvc1", "-x265-params", "log-level=error"]
        cmd += vcodec
        if audio is not None:
            cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += ["-metadata", f"creation_time={created}"]
        target = tmp / f"raw{path.suffix}" if rotate_display else path
        cmd.append(str(target))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"ffmpeg failed writing {path}")
        if rotate_display:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", str(rotate_display),
                            "-i", str(target), "-c", "copy",
                            "-metadata", f"creation_time={created}", str(path)], check=True)


def tripod(match: Match, out: Path, w: int, h: int, label: str) -> None:
    cam = Camera(w, h, (PITCH_W / 2, PITCH_H / 2 - 1.5), span=PITCH_W + 8, tilt=0.62)
    base = render_pitch(cam)
    n = int(match.duration * SIM_FPS)
    frames = (render_frame(base, cam, match, f / SIM_FPS, label) for f in range(n))
    write_video(out, frames, w, h, SIM_FPS, audio_track(match, 0, match.duration, None, 11))


def phone(match: Match, out: Path, start: float, duration: float, w: int, h: int, fps: float,
          span: float, focus_side: str, audio: bool, codec: str, rotate: int, seed: int, label: str) -> None:
    """A handheld clip that pans to follow the ball, with a little shake."""
    rng = np.random.default_rng(seed)
    gx = PITCH_W - 8 if focus_side == "right" else 8
    n = int(duration * fps)
    jitter = np.cumsum(rng.normal(0, 0.05, (n, 2)), axis=0) * 0.3
    pitch_cache: dict = {}

    def frames():
        cx, cy = gx, PITCH_H / 2
        for f in range(n):
            t = start + f / fps
            ball, _ = match.at(t)
            cx += (0.55 * ball[0] + 0.45 * gx - cx) * 0.06
            cy += (ball[1] - cy) * 0.05
            key = (round(cx + jitter[f, 0], 1), round(cy + jitter[f, 1], 1))
            cam = Camera(w, h, key, span=span, tilt=0.7)
            if key not in pitch_cache:
                if len(pitch_cache) > 64:
                    pitch_cache.clear()
                pitch_cache[key] = render_pitch(cam)
            img = render_frame(pitch_cache[key], cam, match, t, label)
            yield np.ascontiguousarray(np.rot90(img, k=1)) if rotate else img

    sw, sh = (h, w) if rotate else (w, h)
    sound = audio_track(match, start, duration, focus_side, seed, level=1.6) if audio else None
    write_video(out, frames(), sw, sh, fps, sound, codec=codec, rotate_display=rotate, created=stamp(start))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("demo/footage"))
    ap.add_argument("--quick", action="store_true", help="smaller frames, for CI")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    duration = 360.0
    match = Match(duration)
    big = 1 if args.quick else 2
    label = "synthetic demo"

    print("rendering sunday_match_wide.mp4 (tripod, 6 min)...")
    tripod(match, args.out / "sunday_match_wide.mp4", 640 * big, 360 * big, label)
    g2, g3, g1 = GOALS[1]["t"], GOALS[2]["t"], GOALS[0]["t"]
    print("rendering phone_goal_portrait.mov...")
    # Rotation -90 = stored sideways, displayed upright (what iPhones write).
    phone(match, args.out / "phone_goal_portrait.mov", g2 - 9, 14, 540 * big, 960 * big, 30,
          span=20, focus_side="left", audio=True, codec="libx265", rotate=-90, seed=21, label=label)
    print("rendering phone_goal_60fps.mp4...")
    phone(match, args.out / "phone_goal_60fps.mp4", g3 - 5, 9, 960 * big, 540 * big, 60,
          span=26, focus_side="right", audio=True, codec="libx264", rotate=0, seed=22, label=label)
    print("rendering phone_chance_noaudio.mp4...")
    phone(match, args.out / "phone_chance_noaudio.mp4", g1 - 14, 22, 640 * big, 360 * big, 30,
          span=30, focus_side="right", audio=False, codec="libx264", rotate=0, seed=23, label=label)

    labels = {
        "video": "sunday_match_wide.mp4",
        "goals": [g["t"] for g in GOALS],
        "goal_details": GOALS,
        "not_goals": {"decoy_shout": DECOY_SHOUT, "near_miss": NEAR_MISS},
        "goal_areas": tripod_goal_areas(640 * big, 360 * big),
        "buildup": "each goal: turnover at t-9s, assist at t-4.2s, shot at t-1s",
        # For scripts/eval_highlights.py: the whole move, and the cheer right after.
        "goal_moments": [{"t": g["t"], "buildup_start": g["t"] - 9.0, "reaction_end": g["t"] + 3.0} for g in GOALS],
        "moments": [{"t": NEAR_MISS, "label": "near_miss"}, {"t": DECOY_SHOUT, "label": "other"}],
        "fully_labeled": True,
        "use_goal_areas": False,
    }
    (args.out / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"done: {args.out}")


def tripod_goal_areas(w: int, h: int) -> list[dict]:
    """Where the goals sit in the tripod frame, as the UI would store them."""
    cam = Camera(w, h, (PITCH_W / 2, PITCH_H / 2 - 1.5), span=PITCH_W + 8, tilt=0.62)
    areas = []
    for gx in (0.0, PITCH_W):
        px, py, s = cam.project(np.array([gx - 3.5, gx + 3.5]),
                                np.array([PITCH_H / 2 - GOAL_HALF - 2.5, PITCH_H / 2 + GOAL_HALF + 1.5]))
        y0 = py[0] - 2.2 * s
        areas.append({"x": round(float(px[0] / w), 4), "y": round(float(y0 / h), 4),
                      "w": round(float((px[1] - px[0]) / w), 4), "h": round(float((py[1] - y0) / h), 4)})
    return areas


if __name__ == "__main__":
    main()
