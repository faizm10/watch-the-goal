"""Evaluate highlight suggestions end to end, per detector, with and without AI review.

    python scripts/eval_highlights.py LABELS.json [LABELS.json ...] \\
        [--detectors none,yolox,rfdetr-nano] [--review] [--out bench/highlights]

Each LABELS.json describes one recording (paths are relative to the file):

    {
      "video": "match.mp4",
      "goals": [{"t": 2810.5, "buildup_start": 2794, "reaction_end": 2814}],   # or plain seconds
      "goal_areas": [{"x": .., "y": .., "w": .., "h": ..}],                     # optional
      "use_goal_areas": true,                                                    # optional
      "fully_labeled": false,       # true only if every goal in the file is listed
      "moments": [{"t": 1219, "label": "save"}, {"t": 1385, "label": "other"}]  # optional
    }

``moments`` labels individual non-goal suggestions (save, near_miss, chance, tackle, other) so
precision can be measured on footage that hasn't been labelled end to end.
Labels: likely_goal is implied by "goals".

Measured per recording and config:

* goals found (a suggestion contains the finish), with the buildup kept
  (starts at or before ``buildup_start``, default finish - 6 s) and the
  reaction (ends at or after ``reaction_end``, default finish + 2 s)
* false suggestions (contain no goal), split by labelled kind
* top-5: how many of the five strongest suggestions are a goal, save or near miss
* rank of each goal, and processing time per minute of footage

With --review, each config is also scored after AI review: its reranking,
and its proposed trims as if the user applied them (as the app proposes them,
only ever extending the clip, and unwidened for comparison). Verdicts are cached in
the output folder, so every config is judged on identical answers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watch_the_goal import pipeline, review, signals  # noqa: E402
from watch_the_goal.media import probe  # noqa: E402

HIGHLIGHT = {"goal", "save", "near_miss", "chance", "tackle"}
AI_TO_TRUTH = {"likely_goal": "goal", "save": "save", "near_miss": "near_miss", "other": "other"}


def load_labels(path: Path) -> dict:
    data = json.loads(path.read_text())
    goals = []
    for g in data.get("goal_moments") or data.get("goals", []):
        g = {"t": float(g)} if isinstance(g, (int, float)) else dict(g)
        g.setdefault("buildup_start", g["t"] - 6.0)
        g.setdefault("reaction_end", g["t"] + 2.0)
        goals.append(g)
    data["goals"] = goals
    data["video_path"] = str((path.parent / data["video"]).resolve())
    data["areas"] = data.get("goal_areas", []) if data.get("use_goal_areas") else []
    return data


def cached_signals(labels: dict, info, out: Path) -> tuple[signals.Signals, float]:
    stem = Path(labels["video_path"]).stem + ("_goals" if labels["areas"] else "")
    f = out / "signals" / f"{stem}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False)
        goal = [d[k] for k in sorted(d.files) if k.startswith("goal")]
        audio = d["audio"] if d["audio"].size else None
        pan = d["pan"] if "pan" in d.files else np.zeros(0)
        return signals.Signals(duration=float(d["duration"]), audio_db=audio, motion=d["motion"],
                               goal_motion=goal, pan=pan), float(d["seconds"])
    t0 = time.time()
    sig = signals.extract(info.path, info.width, info.height, info.duration, info.has_audio, labels["areas"])
    seconds = time.time() - t0
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, duration=sig.duration, seconds=seconds, motion=sig.motion, pan=sig.pan,
             audio=sig.audio_db if sig.audio_db is not None else np.zeros(0),
             **{f"goal{i}": g for i, g in enumerate(sig.goal_motion)})
    return sig, seconds


def truth_at(labels: dict, start: float, end: float) -> str | None:
    """What a suggestion [start, end] actually contains, if labelled."""
    if any(start <= g["t"] <= end for g in labels["goals"]):
        return "goal"
    for m in labels.get("moments", []):
        if start <= m["t"] <= end:
            return m["label"]
    return "other" if labels.get("fully_labeled") else None


def score(clips: list[dict], labels: dict) -> dict:
    """``clips``: dicts with start, end, rank (1 = strongest)."""
    ranked = sorted(clips, key=lambda c: c["rank"])
    goals = []
    for g in labels["goals"]:
        hit = next((c for c in ranked if c["start"] <= g["t"] <= c["end"]), None)
        goals.append({
            "t": g["t"], "found": hit is not None, "rank": hit["rank"] if hit else None,
            "buildup": bool(hit and hit["start"] <= g["buildup_start"]),
            "reaction": bool(hit and hit["end"] >= g["reaction_end"]),
            "clip": [hit["start"], hit["end"]] if hit else None,
        })
    kinds = [truth_at(labels, c["start"], c["end"]) for c in ranked]
    false = [k for k in kinds if k != "goal"]
    top5 = kinds[:5]
    return {
        "suggestions": len(ranked),
        "goals": goals,
        "found": sum(g["found"] for g in goals),
        "found_with_buildup": sum(g["found"] and g["buildup"] and g["reaction"] for g in goals),
        "missed": sum(not g["found"] for g in goals),
        "false": len(false),
        "false_by_label": {k: false.count(k) for k in ("save", "near_miss", "other", None) if false.count(k)},
        "top5_highlights": sum(k in HIGHLIGHT for k in top5),
        "top5_unlabelled": sum(k is None for k in top5),
    }


def run_review(clips: list[dict], labels: dict, info, cfg: dict, cache: dict, cache_file: Path) -> dict:
    """Review the strongest clips (cached), rerank, and score the AI's trims."""
    client = review.OllamaClient(cfg["endpoint"], cfg["model"], cfg["timeout"])
    client.check()
    t_spent, agree, judged, failed = 0.0, 0, 0, 0
    for c in review.pick(clips, cfg["max_candidates"]):
        w0, w1 = review.window_for(c["start"], c["end"], c["anchor"], info.duration)
        prompts = zlib.crc32((review.PROMPT + review.CELEBRATION_PROMPT + str(review.CELEBRATION_AFTER)).encode())
        key = f"{Path(info.path).name}|{w0}|{w1}|{c['anchor']}|{cfg['model']}|{cfg['frames']}|{prompts}"
        if key not in cache:
            t0 = time.time()
            try:
                cache[key] = {"status": "done", **review.review_moment(client, info.path, w0, w1, cfg["frames"],
                                                                       c["anchor"], info.duration).to_dict()}
            except review.ReviewError as exc:
                cache[key] = {"status": "failed", "message": str(exc)}
            cache[key]["seconds"] = round(time.time() - t0, 1)
            cache_file.write_text(json.dumps(cache, indent=1))
        rec = cache[key]
        t_spent += rec["seconds"]
        c["review"] = rec
        if rec["status"] != "done":
            failed += 1
            continue
        truth = truth_at(labels, c["start"], c["end"])
        if truth is not None and rec["label"] != "uncertain":
            judged += 1
            agree += AI_TO_TRUTH.get(rec["label"]) == truth
    review.rerank(clips)
    # As the app proposes them: the AI may extend a clip, never cut into it.
    trimmed = [dict(c, **dict(zip(("start", "end"), review.widen(c, c["review"]), strict=True)))
               if (c.get("review") or {}).get("status") == "done" else c for c in clips]
    raw = [dict(c, start=c["review"]["start"], end=c["review"]["end"])
           if (c.get("review") or {}).get("status") == "done" else c for c in clips]
    return {"reranked": score(clips, labels), "with_ai_trims": score(trimmed, labels),
            "with_raw_ai_trims": score(raw, labels),
            "review_seconds": round(t_spent, 1), "label_agreement": [agree, judged], "failed": failed,
            "verdicts": [{"clip": [c["start"], c["end"]], "truth": truth_at(labels, c["start"], c["end"]),
                          **c["review"]} for c in clips if c.get("review")]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+", type=Path)
    ap.add_argument("--detectors", default="none,yolox,rfdetr-nano")
    ap.add_argument("--review", action="store_true", help="also score AI review (needs Ollama running)")
    ap.add_argument("--review-model", default=review.DEFAULTS["model"])
    ap.add_argument("--review-max", type=int, default=review.DEFAULTS["max_candidates"])
    ap.add_argument("--out", type=Path, default=Path("bench/highlights"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_file = args.out / "review_cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    cfg = review.config({"model": args.review_model, "max_candidates": args.review_max, "timeout": 600})

    report = []
    for lf in args.labels:
        labels = load_labels(lf)
        info = probe(labels["video_path"])
        sig, sig_seconds = cached_signals(labels, info, args.out)
        minutes = info.duration / 60
        for det in args.detectors.split(","):
            res = pipeline.analyze(info.path, info.to_dict(), labels["areas"],
                                   pipeline.Options(detector=det, debug_dir=args.out / "debug" / f"{Path(info.path).stem}_{det}"),
                                   sig=sig)
            if res.notes:
                print(f"{lf.name} {det}: {'; '.join(res.notes)}", file=sys.stderr)
            clips = [m.to_dict() | {"id": str(i), "origin": "detected", "status": "candidate", "edited": False}
                     for i, m in enumerate(res.moments)]
            row = {
                "video": Path(info.path).name, "minutes": round(minutes, 1), "detector": det,
                "goal_areas": bool(labels["areas"]), "fully_labeled": bool(labels.get("fully_labeled")),
                "labelled_goals": len(labels["goals"]),
                "seconds_per_minute": round((sig_seconds + res.stats.get("players_seconds", 0)) / minutes, 2),
                "detector_seconds_per_minute": round(res.stats.get("players_seconds", 0) / minutes, 2),
                "dropped_by_players": res.stats.get("dropped_by_players", 0),
                "baseline": score(clips, labels),
                "suggestions": [{"rank": c["rank"], "start": c["start"], "end": c["end"], "reasons": c["reasons"],
                                 "truth": truth_at(labels, c["start"], c["end"])} for c in sorted(clips, key=lambda c: c["rank"])],
            }
            if args.review:
                try:
                    row["review"] = run_review(clips, labels, info, cfg, cache, cache_file)
                    row["review"]["seconds_per_minute"] = round(row["review"]["review_seconds"] / minutes, 2)
                except review.ReviewError as exc:
                    row["review"] = {"error": str(exc)}
            report.append(row)
            print_row(row)
    (args.out / "results.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out / 'results.json'} and overlays under {args.out / 'debug'}")
    return 0


def print_row(row: dict) -> None:
    b = row["baseline"]
    goals = ", ".join(f"{g['t']:.0f}s→#{g['rank']}{'' if g['buildup'] else ' (buildup cut)'}" if g["found"]
                      else f"{g['t']:.0f}s missed" for g in b["goals"]) or "none labelled"
    print(f"\n{row['video']} · {row['detector']}{' · goal areas' if row['goal_areas'] else ''} · "
          f"{row['seconds_per_minute']} s/min (detector {row['detector_seconds_per_minute']})")
    print(f"  suggestions {b['suggestions']}  goals {b['found']}/{row['labelled_goals']} "
          f"(with buildup+reaction {b['found_with_buildup']})  false {b['false']} {b['false_by_label']}  "
          f"top-5 highlights {b['top5_highlights']} (unlabelled {b['top5_unlabelled']})  [{goals}]")
    r = row.get("review")
    if r and "error" not in r:
        rr, at = r["reranked"], r["with_ai_trims"]
        goals = ", ".join(f"{g['t']:.0f}s→#{g['rank']}" for g in rr["goals"] if g["found"])
        print(f"  + AI review: top-5 highlights {rr['top5_highlights']}  goals [{goals}]  "
              f"AI trims keep buildup+reaction {at['found_with_buildup']}/{row['labelled_goals']} "
              f"(unwidened {r['with_raw_ai_trims']['found_with_buildup']})  "
              f"labels right {r['label_agreement'][0]}/{r['label_agreement'][1]}  "
              f"failed {r['failed']}  {r['seconds_per_minute']} s/min")
    elif r:
        print(f"  + AI review unavailable: {r['error']}")


if __name__ == "__main__":
    raise SystemExit(main())
