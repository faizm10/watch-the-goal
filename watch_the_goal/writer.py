"""Words for a reel: an opening title, a caption on each clip and an end card.

Drafted from what the project already knows: your calls (goal, save, chance,
...), the detector's reasons, match times and the scoreboards. Optionally an
AI model rewrites them, with your own key for Claude, OpenAI or Gemini. Only
that text is sent. Footage and frames never leave this machine.

A draft is a proposal: the recap dialog shows it, and nothing changes until
it is saved.
"""

from __future__ import annotations

import json
import os

TITLE_LIMIT = 64
CAPTION_LIMIT = 60
END_LIMIT = 64

PROVIDERS = {
    # name: (label, default model, environment variables the SDK reads, pip package)
    "claude": ("Claude", "claude-opus-5-5", ("ANTHROPIC_API_KEY",), "anthropic"),
    "openai": ("OpenAI", "gpt-5.5", ("OPENAI_API_KEY",), "openai"),
    "gemini": ("Gemini", "gemini-flash-latest", ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "google-genai"),
}
# Claude models that accept server-side refusal fallbacks.
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")

CALL_TEXT = {"goal": "GOAL", "save": "Big save", "near_miss": "So close", "chance": "Chance", "tackle": "Big tackle"}
REASON_TEXT = (  # detector reason prefix -> caption when nobody tagged the clip
    ("the camera followed play", "Pressure"),
    ("crowd noise", "The touchline erupts"),
    ("motion near the marked", "Goalmouth action"),
    ("play stopped", "Play stops"),
)

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "captions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"clip": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["clip", "text"],
                "additionalProperties": False,
            },
        },
        "end": {"type": "string"},
    },
    "required": ["title", "captions", "end"],
    "additionalProperties": False,
}

SYSTEM = (
    "You write the on-screen text for a short highlight reel of an amateur soccer match. "
    "Write like a friendly club social media account: short, specific and upbeat. "
    "Never invent players, names, scores or events that the notes don't give you."
)

PROMPT = """Here is what we know about each clip in the reel, in order.

{facts}

Write:
- title: the opening card, at most {title} characters. Use the team names if you have them.
- captions: one per clip, at most {caption} characters each, as {{"clip": <number>, "text": "..."}}. Start with what happened (a tag beats a detector note) and include the match minute when it is given, like "GOAL · 47'".
- end: the closing card, at most {end} characters. Use the final score if you have it, otherwise a short sign-off.

Answer with JSON only."""


class WriterError(Exception):
    """The text couldn't be written; nothing in the project changed."""


# --------------------------------------------------------------------- facts

def match_minute(src: dict, t: float) -> int | None:
    """Minute of the match at ``t`` once the recording is trimmed to it, else None."""
    play = src.get("play")
    if not play:
        return None
    t = min(max(t, play["start"]), play["end"])
    brk = play.get("break")
    if brk and t >= brk[1]:
        t -= brk[1] - brk[0]
    elif brk and t > brk[0]:
        t = brk[0]
    return int((t - play["start"]) // 60) + 1


def clock(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def facts(project) -> dict:
    """Plain facts about the kept clips: the only thing a provider ever sees."""
    sources = {s["id"]: s for s in project.sources}
    clips = []
    for n, c in enumerate(project.reel(), 1):
        src = sources[c["source_id"]]
        at = c["anchor"] if c.get("anchor") is not None else (c["start"] + c["end"]) / 2
        call = next((x["label"] for x in src.get("calls", []) if x.get("clip_id") == c["id"]), None)
        reasons = [r for r in c.get("reasons", []) if r and r != "added by you"]
        clips.append({
            "clip": n, "id": c["id"], "recording": src["name"],
            "minute": match_minute(src, at), "time": clock(at),
            "seconds": round(c["end"] - c["start"], 1),
            "tag": call if call in CALL_TEXT else None, "detector": reasons[:3],
        })
    boards = [s.get("scoreboard") or {} for s in project.sources]
    board = next((b for b in boards if b.get("home") or b.get("away")), {})
    score = next((b["score"] for b in boards if b.get("score")), "")
    return {"home": board.get("home", ""), "away": board.get("away", ""), "score": score, "clips": clips}


def describe(f: dict) -> str:
    lines = []
    if f["home"] or f["away"]:
        lines.append(f"Teams: {f['home'] or '?'} v {f['away'] or '?'}")
    if f["score"]:
        lines.append(f"Final score: {f['score']}")
    for c in f["clips"]:
        when = f"minute {c['minute']}" if c["minute"] else f"{c['time']} into the recording"
        what = [f"tagged {c['tag'].replace('_', ' ')}"] if c["tag"] else []
        what += [f"detector: {r}" for r in c["detector"]]
        lines.append(f"Clip {c['clip']}: {when}, {c['seconds']}s long. " + ("; ".join(what) or "no notes") + ".")
    return "\n".join(lines)


# --------------------------------------------------------------------- drafts

def from_detection(f: dict) -> dict:
    """Text from tags and the detector alone. Needs no key and no network."""
    teams = " v ".join(t for t in (f["home"], f["away"]) if t)
    title = teams or "Match highlights"
    captions = {}
    for c in f["clips"]:
        text = CALL_TEXT.get(c["tag"])
        if not text:
            reason = " ".join(c["detector"]).lower()
            text = next((t for prefix, t in REASON_TEXT if prefix in reason), "")
        if text and c["minute"]:
            text += f" · {c['minute']}'"
        captions[c["id"]] = text
    if f["score"]:
        end = f"Full time · {f['home']} {f['score']} {f['away']}".replace("  ", " ").strip()
    else:
        end = "Full time"
    return {"title": title, "captions": captions, "end": end}


def tidy(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[:limit - 1].rstrip(" ,.;:-") + "…"
    return text


def validate(raw: str, f: dict) -> dict:
    """Parse a provider's answer and keep only what fits the reel."""
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        raise WriterError("The model didn't answer with JSON. Try again, or use tags and detection.") from None
    if not isinstance(obj, dict) or not isinstance(obj.get("captions"), list):
        raise WriterError("The model's answer was missing the captions. Try again.")
    ids = {c["clip"]: c["id"] for c in f["clips"]}
    captions = {cid: "" for cid in ids.values()}
    for item in obj["captions"]:
        if isinstance(item, dict) and item.get("clip") in ids:
            captions[ids[item["clip"]]] = tidy(item.get("text"), CAPTION_LIMIT)
    return {"title": tidy(obj.get("title"), TITLE_LIMIT), "captions": captions, "end": tidy(obj.get("end"), END_LIMIT)}


# --------------------------------------------------------------------- providers

def _sdk(provider: str):
    package = PROVIDERS[provider][3]
    try:
        if provider == "claude":
            import anthropic as sdk
        elif provider == "openai":
            import openai as sdk
        else:
            from google import genai as sdk
    except ImportError:
        raise WriterError(f"{PROVIDERS[provider][0]} needs its SDK. Install it with: "
                          f"uv sync --extra ai  (or pip install {package})") from None
    return sdk


def _claude(sdk, key: str | None, model: str, prompt: str) -> str:
    # Without a key the SDK finds ANTHROPIC_API_KEY or an `ant auth login` profile itself.
    client = sdk.Anthropic(api_key=key) if key else sdk.Anthropic()
    extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if model in FALLBACK_MODELS else {}
    try:
        msg = client.beta.messages.create(
            model=model, max_tokens=4000, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            **extra,
        )
    except sdk.AuthenticationError:
        raise WriterError("Claude rejected the API key.") from None
    except sdk.APIStatusError as exc:
        raise WriterError(f"Claude returned an error ({exc.status_code}): {exc.message}") from None
    except sdk.APIConnectionError:
        raise WriterError("Couldn't reach Claude. Check the internet connection.") from None
    except TypeError:  # the SDK found no key, token or profile
        raise WriterError("Paste a Claude API key, or set ANTHROPIC_API_KEY before starting the app.") from None
    if msg.stop_reason == "refusal":
        raise WriterError("Claude declined to write this. Use tags and detection instead.")
    return next((b.text for b in msg.content if b.type == "text"), "")


def _openai(sdk, key: str | None, model: str, prompt: str) -> str:
    client = sdk.OpenAI(api_key=key) if key else sdk.OpenAI()
    try:
        resp = client.responses.create(
            model=model, instructions=SYSTEM, input=prompt,
            text={"format": {"type": "json_schema", "name": "reel_text", "schema": SCHEMA, "strict": True}},
        )
    except sdk.AuthenticationError:
        raise WriterError("OpenAI rejected the API key.") from None
    except sdk.APIStatusError as exc:
        raise WriterError(f"OpenAI returned an error ({exc.status_code}): {exc.message}") from None
    except sdk.APIConnectionError:
        raise WriterError("Couldn't reach OpenAI. Check the internet connection.") from None
    return resp.output_text


def _gemini(sdk, key: str | None, model: str, prompt: str) -> str:
    from google.genai import errors

    try:
        client = sdk.Client(api_key=key) if key else sdk.Client()
        resp = client.models.generate_content(
            model=model, contents=prompt,
            config={"system_instruction": SYSTEM, "response_mime_type": "application/json",
                    "response_json_schema": SCHEMA},
        )
    except errors.APIError as exc:
        raise WriterError(f"Gemini returned an error ({exc.code}): {exc.message}") from None
    except ValueError as exc:  # no key anywhere
        raise WriterError(f"Gemini: {exc}") from None
    return resp.text or ""


CALLS = {"claude": _claude, "openai": _openai, "gemini": _gemini}


def has_key(provider: str) -> bool:
    return any(os.environ.get(v) for v in PROVIDERS[provider][2])


def draft(f: dict, provider: str = "detection", key: str | None = None, model: str | None = None) -> dict:
    if not f["clips"]:
        raise WriterError("Keep some clips first: the text is written for the clips in the reel.")
    if provider == "detection":
        return {**from_detection(f), "provider": "detection"}
    if provider not in PROVIDERS:
        raise WriterError(f"Unknown provider {provider!r}.")
    label, default, env, _ = PROVIDERS[provider]
    if not key and not has_key(provider) and provider != "claude":
        raise WriterError(f"Paste a {label} API key, or set {env[0]} before starting the app.")
    prompt = PROMPT.format(facts=describe(f), title=TITLE_LIMIT, caption=CAPTION_LIMIT, end=END_LIMIT)
    raw = CALLS[provider](_sdk(provider), key or None, model or default, prompt)
    return {**validate(raw, f), "provider": provider}
