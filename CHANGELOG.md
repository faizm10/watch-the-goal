# Changelog

All notable changes to Watch the Goal are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- AI review asks a second, narrower question about each moment: are players
  celebrating a goal? A "yes" makes it a likely goal, with what the model saw
  as the reason. In testing on a real match it never called a non-goal a
  celebration; it adds about 30 s per suggestion.
- Edit, remove and add clips from the Reel panel: **✎** opens a clip in the
  editor, **✕** takes it out of the reel, and **+ Clip at playhead** adds one.
- A **Suggestions** list under the reel shows every suggested moment not yet
  in it, strongest first, with **+** to add one and **Add all** to add them
  all. Auto-fill stops at the length you pick, so the rest are no longer
  hidden.

### Changed
- Suggested clips are 5 to 15 s long: about 10 s of buildup and 4 s of
  reaction, instead of 17 to 30 s. Untrimmed suggestions in existing projects
  are shortened when the project opens; clips you trimmed yourself are kept.

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
