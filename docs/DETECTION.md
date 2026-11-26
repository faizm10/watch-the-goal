# How detection works

Watch the Goal suggests moments; it never decides for you. Every suggestion
shows the evidence that produced it ("crowd noise +14 dB above the surrounding
level for 2.5s", "motion near the marked left goal") and starts out as a
candidate you keep or skip. There is no confidence score, because the detector
has no calibrated way to produce one.

The code:

- [`watch_the_goal/signals.py`](../watch_the_goal/signals.py) decodes the video
  with ffmpeg and turns it into small time series.
- [`watch_the_goal/detect.py`](../watch_the_goal/detect.py) turns those series
  into moments and ranks them. It is pure numpy, so you can test it without
  any video.
- [`watch_the_goal/pipeline.py`](../watch_the_goal/pipeline.py) runs the
  steps for a long recording, including the optional player detector
  ([`vision/`](../watch_the_goal/vision/)), and saves debug overlays.
- [`watch_the_goal/review.py`](../watch_the_goal/review.py) is the optional AI
  review that runs after detection.

## 1. Signals

| Signal | How | Rate |
| --- | --- | --- |
| Loudness | mono 8 kHz audio, RMS level in dBFS | every 0.25 s |
| Motion | frames scaled to 192 px on the long side, greyscale; share of pixels whose brightness changed by more than 14 levels since the previous frame | 5 per second |
| Goal-area motion | the same, but only inside each goal box you drew | 5 per second |
| Camera pan | horizontal shift of the background behind the pitch (the band 12–42% down the frame) between those frames, by phase correlation; weak matches count as no pan | 5 per second |

Audio and video are decoded in parallel, streamed, and never held in memory
whole. A 51-minute 1080p phone recording takes about 45 s on an M-series Mac.

## 2. Events

Each signal is compared to its own *local* normal level, a rolling median over
60 s, scaled by a robust spread (MAD). That way a loud indoor dome and a quiet
park are each judged against themselves.

- **Crowd noise**: loudness more than 3.5 robust deviations *and* 6 dB above
  the local level, sustained for at least 0.75 s. The duration rule stops ball
  strikes and single shouts from counting.
- **Motion near a marked goal**: goal-box motion more than 3.5 deviations above
  normal and covering at least 2% more of the box, for at least 1 s.
- **Play stopped after a burst**: average motion over the next 10 s drops below
  60% of the previous 8 s, after a busier-than-usual spell. Goals are usually
  followed by a restart, so this mostly *supports* other evidence.
- **Camera followed play to one end**: auto-panning cameras (Veo, Trace,
  Pixellot) follow the ball, so where the camera points says where play is.
  Summing the pan gives the camera's direction. Over the surrounding 3
  minutes its sweep runs from the 5th to the 95th percentile, and the
  middle of that is 0. The camera counts as at an end beyond 75% of the way
  to either side, held for at least 3 s while play is busy (motion at least
  0.9× the median). A camera that sweeps less than 0.15 frame widths (a
  phone or tripod) never triggers this.

## 3. Moments

1. Crowd noise and goal-box motion start moments. A play stoppage only adds
   to one, except in footage with no audio, no goal boxes and a fixed camera,
   where it is the only evidence left. A spell at one end joins the moment
   it led to if it ended between 4 s before and 14 s after that moment's
   noise. After a goal the camera often stays at that end through the
   celebration. On its own, a spell at one end starts a moment only if play
   stopped right after (a shot out, a save, a goal). Play reaches a final
   third every minute or so, and most of those spells fizzle out.
2. Events within 6 s of each other are one moment.
3. The moment's time ("the finish") is the goal-box motion if there was any,
   otherwise half a second before the loudest reaction.
4. With goal boxes marked, noise that has neither goal-box motion nor a
   stoppage after it is dropped. In testing, that is usually an argument.
5. The clip runs from **10 s before** the moment, to keep the turnover and the
   assist, to 4 s or more after it, depending on how long the reaction lasts.
   If play is still surging at the 10 s mark (smoothed motion at least 1.3×
   the recording's median), the start walks back to where the surge began,
   at most 11 s before the moment. A cut in the middle of a counterattack is
   the most common way a highlight loses its buildup. Every suggested clip
   is **5 to 15 s** long, so a reel is a run of short moments: past 15 s the
   reaction is cut first, then buildup beyond 10 s. Two overlapping moments
   merge into one clip only if it stays within 15 s; otherwise the later one
   becomes its own clip, starting where the first ends.
6. Candidates are ordered by strength (how many kinds of evidence agree, and
   how strongly). With a player detector, the strongest 3× as many as will be
   kept are re-checked (section 4).
7. At most one suggestion per 90 s of footage (minimum 4) is kept and numbered
   by rank (#1 strongest). Overlapping clips from the same recording are
   merged, keeping the better rank and both sets of reasons.

Short clips (2 minutes or less) skip all of this and go straight into the reel.
The detector only *suggests a trim*, and only from audio or goal-box evidence.
A handheld phone pans and zooms, so whole-frame motion says little about where
the goal is, and a wrong trim would cut the moment the clip was filmed for.

## 4. Player evidence (optional)

Choose a detector in **Analysis** (or `detect --detector`). Both run on the CPU
through OpenCV's DNN module and only keep the COCO "person" class, behind one
interface (`vision/detectors.py`): **YOLOX-S** from OpenCV Zoo (640×640) and
**RF-DETR Nano** (384×384). Person detection per frame is kept separate from
event scoring: the detector only returns boxes, and `vision/players.py` turns
boxes into counts.

For each re-checked candidate, one frame per second from 6 s before to 2 s
after the moment is searched, plus 40 frames spread over the whole file to
learn what is *usual* for this recording. Per frame:

- **Players near a goal**: people whose feet (bottom centre of the box) are
  inside the marked goal box widened by half its size on every side.
- **Bunch**: the most people within two body heights of any one person, a
  rough "goalmouth scramble or celebration" cue that works without goal boxes.
- **Lens blocked**: a person taller than 35% of the frame covers at least 30%
  of a goal box.

Then:

- "N players near the left goal (usually M)" is added when the peak is at
  least 2 and at least 2 above usual. It strengthens the moment and stretches
  the clip to end at least half a post-roll after that frame (giving up
  buildup if needed to stay within 15 s).
- Otherwise "N players bunched together (usually M)" is added when the peak
  is at least 4 and at least 3 above usual.
- If the lens was blocked, "motion near the marked goal" is withdrawn; a
  moment with nothing else behind it is dropped.

Tracking (ByteTrack) was evaluated and not added: counts at 1 fps don't need
identities, and tracking needs about 10 fps, which multiplies detector time by
about 9 for no change in which moments clear the thresholds. See
[BENCHMARKS.md](BENCHMARKS.md).

## 5. AI review (optional)

After detection, a local vision model can look at the strongest few
suggestions that you haven't touched (default 6; Analysis dialog or
`detect --review`). It talks to [Ollama](https://ollama.com) on this machine
only: the endpoint must be a loopback address.

- **Input:** 12 frames from 20 s before to 8 s after the moment (at most
  40 s), 768 px on the long side, each with its timestamp burned in and listed
  in the prompt, so the model sees the buildup and can name times.
- **Output:** JSON constrained by a schema: `label` (likely goal, save, near
  miss, other, uncertain), `start` and `end` inside the window it was shown,
  and a short `reason` about what is visible. Anything else (bad JSON, unknown
  label, times outside the window, start after end, empty reason) is rejected.
- **Effect:** suggestions are re-ranked (likely goals first, then saves and
  near misses, then unreviewed or uncertain, then "other"); the verdict shows
  as an extra reason; where the proposed start and end make the clip
  *longer*, up to 15 s, they become a **suggested trim** you apply with one click. The
  AI can add buildup or aftermath but never cut into the detector's clip: in
  testing its trims cut the buildup or the reaction as often as they added to
  them. Clips you've kept, skipped or trimmed are never
  touched, including when you edit them while the review is running.
- **Celebration check:** a second, narrower question about the same
  moment: 7 frames from 2 s before to 12 s after it, at 1024 px, asking only
  whether players are celebrating a goal (running together, arms up,
  hugging, the keeper fetching the ball from the net). A "yes" makes the
  suggestion a likely goal, with the model's description as the reason
  ("players celebrate: …"); "no" and "unsure" change nothing. The model
  can't see the ball go in on wide amateur footage, but it can see a
  celebration. It adds about 30 s per suggestion.
- **Failure:** model missing, Ollama not running, a timeout or an invalid
  answer leave every suggestion exactly as detection made it, and the source
  card says what happened. A failed celebration check keeps the first look.

Tested with Qwen3.5 9B (Apache-2.0). It takes about a minute per suggestion
on an Apple M4, half of it the celebration check. On real footage it
recognised ordinary play and a save reliably and moved the one real goal to
the top, but its first look called that goal a near miss; the celebration
check then marks it a likely goal (BENCHMARKS.md). On the drawn demo match it
got most labels wrong. It needs an 8 GB model and several minutes per
recording, so it is off by default.

## 6. Your calls

A call is you saying what really happened at a moment: goal, save, near
miss, chance created, tackle, or not a highlight. Everything but the last
counts as a highlight, so a tackle that detection missed can lower a bar
too. Tackles are often quiet, though, so many will get the "too little crowd
noise to learn from" note instead. [`watch_the_goal/coach.py`](../watch_the_goal/coach.py)
learns from calls for **one recording at a time**. Nothing carries over to
other recordings, and the defaults never change.

- **Explain.** For any moment, the readout compares each signal with its bar
  over the window where its evidence would be: crowd noise from 4 s before to
  8 s after, held for 0.75 s; goal-box movement held for 1 s; a stoppage
  afterwards. It then says what the detector does with the moment: suggested,
  a candidate cut by the cap, filtered (noise without goal-mouth action once
  goals are marked), or missed. The live view shows the same numbers every
  0.5 s.
- **Re-tune.** Each bar (crowd noise, goal-box movement) is scaled from the
  default, never from the previous tuning, so the same calls always give the
  same bars.
  - A highlight it missed **lowers** a bar to just under that highlight's
    evidence, but no lower than half the default. Each highlight counts
    toward the signal that saw more of it.
  - A highlight with too little evidence for either signal is reported and
    left alone.
  - Lowering only adds suggestions to review. If it also lets back in a
    moment you rejected, the change says so.
  - "Not a highlight" calls can **raise** a bar, with three limits. It needs
    at least one confirmed highlight. It never goes past a confirmed
    highlight. And it drops at most one suggestion you haven't reviewed per
    rejection it honours. Without that last limit, one goal and three
    rejections on the Worker FC recording raised the bar enough to drop 20
    unreviewed suggestions, including four windows labelled "unclear". With
    it, the bar stays, and the note says the smallest raise would still drop
    8 unreviewed suggestions.
- **Resemble.** Each candidate is compared with your calls on four numbers:
  loudest rise, seconds of noise, goal-box movement, and the drop in play
  afterwards. Each number is scaled by its robust spread across this
  recording. A candidate within distance 1 of up to three calls gains or loses
  1.5 strength, about one extra kind of evidence. It also gets a reason
  naming the nearest call ("like your goal at 46:50", "like 20:43, which you
  called not a highlight"). This happens before the cap, so a lookalike of a
  goal can make the cut.
- **Signals are cached** per recording (`cache/<source>-signals.npz`, keyed by
  the goal boxes). Re-tuning and re-analysing take well under a second, and
  only changing the goal boxes decodes the video again. Changing the goal
  boxes also clears learned tuning, because it was learned from the old
  evidence. Your calls stay.

Measured on the two real recordings, with calls taken from their hand labels
(not a held-out test; it checks the mechanics):

| Recording | Calls | Bars after re-tune | Suggestions | Result |
| --- | --- | --- | --- | --- |
| Worker FC (Veo, 57 min) | goal 46:50; not a highlight at 4:53, 15:48, 32:28, 36:39 | unchanged (raise blocked: would drop 8 unreviewed) | 31 → 29 | goal still found; 7 candidates re-ranked by resemblance |
| Indoor dome (51 min, no goal boxes) | save 20:16 (missed); not a highlight at 20:43, 31:41, 41:38 | crowd noise at 84% of default | 3 → 11 | save now found on its own (#2); 2 candidates re-ranked by resemblance |

Suggestion counts are the pipeline's own output. The app shows one or two
fewer, because it doesn't re-suggest a moment a clip you kept, skipped or
added already covers. Neither recording is fully labelled, so whether the
re-ranked candidates are better is unmeasured.

Live detection sends the frame the browser is showing (960 px on the long
side, JPEG) to `POST /api/sources/<id>/live`. One frame is in flight at a time,
so the overlay updates as fast as the detector allows. YOLOX took 57–69 ms a
frame and RF-DETR Nano 81 ms on an Apple M4. Boxes older than 0.6 s fade, so
a slow detector never looks like it's tracking.

## 7. What we measured

[BENCHMARKS.md](BENCHMARKS.md) has the numbers, the labelled footage, the
hardware and the commands. In short: the synthetic match checks the mechanics
(3/3 goals with buildup in every configuration). On real footage the one
confirmed goal is found but its clip starts 4.5 s after the attack began;
with the goal marked, player detectors trim the indoor recording's
suggestions from 24 to 15 and lift its one save from #19 to #2; AI review
puts the goal and the save at #1 but mislabels the goal.

### The camera-pan cue on two Veo matches

Two full matches (about 56 and 58 minutes). The labels are the calls
made in the app: 2 goals, 1 chance and 2 tackles in one match, and 1 goal
in the other, found by watching the stoppages. Ranks are among the shown
suggestions, before and after the camera-pan cue:

| | Before | After |
| --- | --- | --- |
| Match 1: goal 46:50 | #3 | #2 |
| Match 1: goal 56:12 | #7 | #4 |
| Match 1: chance 51:35 | #1 | #1 |
| Match 1: tackles 15:16, 42:50 | #5, #9 | #3, #5 |
| Match 1: false alarm 4:53 | #2 | #10 |
| Match 2: goal 35:31 | #1 | #1 |
| Match 2: suggestions | 6 | 35 |

In match 1 all five labelled highlights are now the top five. Match 2 was
filmed at night with a quiet microphone, so crowd noise found only 6
moments. The camera's movement adds the attacks that ended in a stoppage.
Spot checks show a mix of shots, chances and ordinary build-up that ran
out, so tag them as you go.

### Tried and dropped (2026-11)

Measured on the Veo match (2 goals, a chance, 2 tackles and 4 "not a
highlight" from the calls above), 10 strongest suggestions, Qwen3.5 9B:

- **Restart frames for the AI.** Four extra frames 15 to 60 s after the
  moment, so the model could see a kick-off from the centre spot (a goal) or
  a goal kick (not). On this footage the camera stays at the goal end and
  never shows the centre circle, and the second goal came near full time.
  Labels right fell from 2 of 6 to 1 of 6 and the second goal dropped from
  #3 to #34.
- **Soccer cues in the main prompt** (ball in the net, keeper fetching it,
  how celebrations look): 1 of 6 labels right instead of 2, ranks unchanged.
  The narrow celebration question above worked where this didn't.
- **Celebration frames.** The answer depends on exactly which frames the
  model sees. Across four samplings each goal got a "yes" 3 and 1 times out
  of 4, and the eight other moments never did (0 of 32, plus 0 of 4 on the
  drawn match). So a "yes" is trusted and a "no" means nothing.
- **Referee whistles.** A whistle is easy to find (a steady 3–4.5 kHz tone
  holding nearly all the energy in that band): 36 in the match, including
  half time and one 6 s after the second goal. But whistles also come for
  fouls and offsides, and whether one followed a moment didn't separate
  goals from the rest.
- **Following the play in portrait.** A 9:16 crop that pans with the action,
  from motion once the camera's pan is taken out (or from RF-DETR player
  boxes), in the spirit of AutoFlip. It tracked play on the drawn match but
  on the Veo goal it framed empty grass during the goalmouth scramble and the
  celebration. Without finding the ball it isn't good enough to ship.

## 8. Where to improve it

Good first projects, roughly in order of payoff:

- **Tell cheering from talking.** Loudness alone can't separate a celebration
  from spectators chatting near the phone. Spectral features (broadband energy,
  many overlapping voices, rising pitch) or a small pretrained audio tagger
  (e.g. YAMNet-sized, CPU-only) would cut false positives a lot.
- **Ignore people near the camera without a detector.** The player detector's
  lens veto removes most of these, but costs 1.5 to 2.5 s per minute of
  footage. Masking large blobs that touch the frame edge, or weighting motion
  by distance (row in the frame), would get some of that for free.
- **Distant players.** Full-frame detectors miss most players far across a
  wide shot. Tiling only around the marked goals, rather than the whole frame,
  would raise recall where it matters at a fraction of the cost.
- **Ball and net cues.** With goal boxes marked, detecting the net bulging or
  a small fast object entering the box is a stronger signal than generic
  motion. OpenCV background subtraction is a reasonable start.
- **Restart detection.** After a goal, players drift back into two halves and
  the ball goes to the centre. Spatial motion distributions over 20 to 40 s
  could recognise that pattern.
- **Camera-motion compensation** for handheld clips (global motion estimation,
  e.g. OpenCV `estimateAffinePartial2D` on tracked features) so motion signals
  mean something when the phone pans.
- **Learn thresholds from labelled matches.** `scripts/eval_detection.py`
  scores any labelled recording. A handful of labelled pickup games would let
  thresholds be fitted instead of hand-picked.

Whatever you add, keep two rules: every suggestion must say *why* in words a
player understands, and nothing may claim a confidence the system can't back up.
