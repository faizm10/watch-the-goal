---
name: ffmpeg-essentials
description: Inspect and normalise phone and camera footage for Watch the Goal with ffprobe and FFmpeg - rotation, variable frame rate, odd timestamps, missing or mono audio - and trim frame-accurately. Use when an imported clip plays sideways, drifts out of sync, starts at the wrong time, won't preview, has no sound, or when changing watch_the_goal/media.py or export.py.
---

# FFmpeg essentials (Watch the Goal)

Probe first. Never write over a source file: the app only reads footage, and so should you. Put scratch output in `bench/` or a temp folder (both git-ignored).

## Inspect

```bash
# What the app sees: display size after rotation, fps, duration, audio
uv run python -c "import sys, json; from watch_the_goal.media import probe; print(json.dumps(probe(sys.argv[1]).to_dict(), indent=1))" clip.mov

# Raw streams: codecs, rotation side data, frame rates, start times
ffprobe -v error -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,start_time,sample_rate,channels:stream_side_data=rotation:stream_tags=rotate -show_entries format=duration,start_time -of json clip.mov
```

What to look for:

| Field | Problem it reveals |
|---|---|
| `side_data rotation` / `tags.rotate` | Portrait phone clip stored sideways. Players and FFmpeg apply it; OpenCV only with `CAP_PROP_ORIENTATION_AUTO` (the app sets it). |
| `r_frame_rate` ≠ `avg_frame_rate` | Variable frame rate (phones). Export forces constant 30 fps. |
| `start_time` > 0 | Timestamps don't start at zero (edit lists, trimmed uploads). The app's times are relative to the first frame. |
| no audio stream, or `channels: 1` | Export adds silence or upmixes to stereo so the reel concatenates cleanly. |
| width or height odd | Encoders need even sizes; export scales with `force_divisible_by=2`. |

## Debug rotation

```bash
# Frame as players show it (rotation applied) vs as stored
ffmpeg -v error -ss 2 -i clip.mov -frames:v 1 shown.jpg
ffmpeg -v error -noautorotate -ss 2 -i clip.mov -frames:v 1 stored.jpg
# Same check through OpenCV, the path the detectors use
uv run python -c "import sys, cv2; from watch_the_goal.vision.frames import sample_frames; [(cv2.imwrite('cv.jpg', f), print(f.shape)) for _, f in sample_frames(sys.argv[1], [2.0])]" clip.mov
```

`shown.jpg` and `cv.jpg` must match. If OpenCV's is sideways, the OpenCV build lacks orientation support; tell the user rather than rotating by hand.

## Debug timestamps and sync

```bash
# Packet timestamps of the first video and audio packets
ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,dts_time -read_intervals %+#5 -of csv clip.mp4
ffprobe -v error -select_streams a:0 -show_entries packet=pts_time -read_intervals %+#5 -of csv clip.mp4
# Audio vs video length (a gap > 0.1 s will show as drift in a long reel)
ffprobe -v error -show_entries stream=codec_type,duration -of csv=p=0 clip.mp4
```

## Normalise (what export does per clip)

`watch_the_goal/export.py:segment_command` is the source of truth. Equivalent by hand, for one clip into a 1920x1080 reel:

```bash
mkdir -p out
ffmpeg -v error -y -ss 30 -t 15 -i clip.mov \
  -filter_complex "[0:v]split=2[bgsrc][fgsrc];[bgsrc]scale=480:270:force_original_aspect_ratio=increase,crop=480:270,boxblur=8:2,eq=brightness=-0.06,scale=1920:1080[bg];[fgsrc]scale=1920:1080:force_original_aspect_ratio=decrease:force_divisible_by=2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps=30,format=yuv420p[v];[0:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=0:15,afade=t=in:d=0.05,afade=t=out:st=14.88:d=0.12[a]" \
  -map "[v]" -map "[a]" -t 15 -c:v libx264 -preset veryfast -crf 20 -profile:v high -c:a aac -b:a 160k -video_track_timescale 30000 out/segment.mp4
```

Clip without audio: add `-f lavfi -t 15 -i anullsrc=r=48000:cl=stereo` and map that input's audio instead.

## Trim

```bash
# Frame-accurate (re-encode). Use this for anything a person will watch.
ffmpeg -v error -y -ss 2794 -i match.mp4 -t 20 -c:v libx264 -crf 20 -c:a aac out/goal.mp4
# Fast, keyframe-aligned (start can be seconds early). Only for quick looks.
ffmpeg -v error -y -ss 2794 -i match.mp4 -t 20 -c copy out/goal-rough.mp4
```

Check a trim landed where intended by comparing frames: `uv run python scripts/check_reel.py` does this for whole reels (see the `video-qa` skill).

## Make test fixtures

Generate them; never commit real footage. `tests/conftest.py` and `scripts/make_demo.py` show how (lavfi `testsrc2`, `sine`, rotation metadata via `-display_rotation`, VFR via `setpts`).

```bash
ffmpeg -v error -y -f lavfi -i testsrc2=size=1280x720:rate=30 -f lavfi -i sine=frequency=440 -t 5 -c:v libx264 -pix_fmt yuv420p -c:a aac fixture.mp4
ffmpeg -v error -y -display_rotation 90 -i fixture.mp4 -c copy portrait.mov   # phone-style rotation tag
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Preview won't play in the browser (HEVC, odd codec) | The app makes an H.264 proxy (`media.py`); check `ffmpeg -codecs` has libx264 |
| Clip sideways in the reel | Probe rotation; the export relies on FFmpeg autorotate, so don't pass `-noautorotate` |
| Audio drifts late in a long reel | Each segment must have audio exactly as long as video: `apad` + `-t`, then concat with `-c copy` |
| `height not divisible by 2` | `scale=...:force_divisible_by=2` or `scale=trunc(iw/2)*2:trunc(ih/2)*2` |
| Plays in VLC, not on a phone | `-c:v libx264 -pix_fmt yuv420p -movflags +faststart` |

## Check a recap cut

Run `uv run pytest -q tests/test_boundaries.py tests/test_export.py` for timestamped fixtures and normalized phone streams. Use the vertical Playwright command in the webapp-testing skill to exercise FFmpeg crop, fit, and the two-second title. Compare the saved title/goal preview and export PNGs at the same source time.
