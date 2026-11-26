# Changelog

All notable changes to Watch the Goal are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-25

The first release.

### Highlights
- Import whole matches and phone clips by path, upload or drag and drop. Originals
  are read in place and never modified.
- Suggested moments with plain-language reasons, from crowd noise, motion in
  marked goal areas, play stopping after a burst of action, and, on
  auto-panning cameras like Veo, the camera following play to one end
  ([#6](https://github.com/faizm10/watch-the-goal/pull/6)).
- Review, keep, skip and re-trim clips, with keyboard shortcuts throughout.
- Tag goals, saves, near misses, chances and tackles as you watch, and re-tune
  detection from your calls ([#2](https://github.com/faizm10/watch-the-goal/pull/2)).
- Auto-fill the reel to about 45, 60 or 90 seconds, with one-step undo.
- Trim a recording to kick-off, halftime and full time
  ([#3](https://github.com/faizm10/watch-the-goal/pull/3),
  [#4](https://github.com/faizm10/watch-the-goal/pull/4)).
- Timeline controls: zoom, drag to scrub with a frame preview, playback speed,
  ±10 s skip and frame stepping ([#5](https://github.com/faizm10/watch-the-goal/pull/5)).
- Match recaps with an opening title, scoreboards, animated captions, fades
  between clips and an end card. The words can be drafted offline from tags and
  detection, or optionally by Claude, OpenAI or Gemini with your own key; only
  text is sent ([#7](https://github.com/faizm10/watch-the-goal/pull/7)).
- Export one MP4 in 16:9, 9:16 or 1:1, with every clip rotated upright and
  normalised to 30 fps and stereo AAC.
- Optional local extras: a player detector (YOLOX or RF-DETR Nano) and AI review
  by a vision model run with Ollama.

[Unreleased]: https://github.com/faizm10/watch-the-goal/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/faizm10/watch-the-goal/releases/tag/v0.1.0
