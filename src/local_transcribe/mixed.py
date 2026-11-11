"""Find the stretches spoken in a second language, and put them back in that language.

    local-transcribe mixed TRANSCRIPT.json --other en [--threshold 0.15]
    local-transcribe mixed TRANSCRIPT.json --other en --reuse --apply 0.9

Whisper decodes a run in one language. A few words of another language survive, written in
their own script. A whole sentence in the other language does not: it is *translated* into
the run's language, and the transcript reads fluently and is wrong.

Segments are grouped into windows of at most 28 s and Whisper is asked for the language of
each. Where the other language scores at least ``threshold``, the window is transcribed
again in that language. Every such window goes to ``X.mixed.jsonl``, one JSON object per
line with both readings, for a person or an agent to judge.

``--apply P`` writes ``X.mixed.transcript.json`` with every listed window that scores at
least ``P`` replaced by its second reading. A replacement swaps the *whole* window, up to
28 s, for one segment. So a window that holds both languages loses its main-language half:
delete that line from the listing (or edit its ``other_text``) before applying. ``--reuse``
applies the listing as it stands on disk instead of looking again, which is what makes
editing it worthwhile. The listing belongs to the transcript it was made from.

On the recording this was written for (114 min, Russian with English mixed in) a threshold
of 0.15 flagged 11 windows. The four above 0.95 were real translations. One at 0.83 held
both languages at once. The rest were the main language with a few foreign words, already
right. Hence 0.9 as the suggested value, and hence: read the listing before applying it.

All heavy imports are inside functions so the rest of the CLI stays light.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from local_transcribe.transcribe import DEFAULT_MODEL, SAMPLE_RATE, _log, ensure_ffmpeg

DEFAULT_THRESHOLD = 0.15
DEFAULT_MAX_WINDOW = 28.0
DEFAULT_MAX_PAUSE = 2.0
MIN_WINDOW = 1.0
SAME_TIME = 0.01


def group_windows(
    segments: Sequence[dict[str, Any]],
    max_window: float = DEFAULT_MAX_WINDOW,
    max_pause: float = DEFAULT_MAX_PAUSE,
) -> list[tuple[int, int]]:
    """Runs of consecutive segments as ``(first, last)`` indices: at most ``max_window`` s long, split at pauses over ``max_pause`` s."""
    windows: list[tuple[int, int]] = []
    first = 0
    for i, s in enumerate(segments):
        if i > first and (
            float(s["end"]) - float(segments[first]["start"]) > max_window
            or float(s["start"]) - float(segments[i - 1]["end"]) > max_pause
        ):
            windows.append((first, i - 1))
            first = i
    if segments:
        windows.append((first, len(segments) - 1))
    return windows


def find_candidates(
    transcript: dict[str, Any],
    *,
    other: str,
    threshold: float = DEFAULT_THRESHOLD,
    audio: Path | str | None = None,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    log: Callable[[str], None] = _log,
) -> list[dict[str, Any]]:
    """Windows where ``other`` scores at least ``threshold``, each with its text in both languages."""
    ensure_ffmpeg()
    import whisperx

    from local_transcribe.probe import load_whisper

    main = transcript.get("language") or "en"
    segments = transcript["segments"]
    wav = whisperx.load_audio(str(audio or transcript["audio"]))
    model, device = load_whisper(model_name, device)
    windows = group_windows(segments)
    log(f"{len(windows)} windows; looking for {other} inside {main} ({device})")

    found: list[dict[str, Any]] = []
    for first, last in windows:
        inside = segments[first : last + 1]
        start, end = float(inside[0]["start"]), max(float(s["end"]) for s in inside)
        if end - start < MIN_WINDOW:
            continue
        chunk = wav[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        _, _, scores = model.detect_language(chunk)
        scores = dict(scores)
        p_other = float(scores.get(other, 0.0))
        if p_other < threshold:
            continue
        decoded, _ = model.transcribe(chunk, language=other, vad_filter=False, beam_size=5, condition_on_previous_text=False)
        found.append(
            {
                "first": first,
                "last": last,
                "start": round(start, 3),
                "end": round(end, 3),
                "other": other,
                "p_other": round(p_other, 3),
                "p_main": round(float(scores.get(main, 0.0)), 3),
                "text": " ".join(s["text"] for s in inside),
                "other_text": " ".join(s.text.strip() for s in decoded),
            }
        )
    log(f"{len(found)} windows scored {threshold} or more for {other}")
    return found


def apply_candidates(transcript: dict[str, Any], candidates: Sequence[dict[str, Any]], *, min_prob: float) -> dict[str, Any]:
    """A copy of ``transcript`` where each window scoring ``min_prob`` or more is one segment in the other language.

    The segments ``first..last`` of a window are replaced by one that takes the speaker who
    held most of the window and carries ``language``; it stays ``filled`` if all of them
    were. Windows below ``min_prob`` and windows with an empty second reading are left
    alone. A candidate whose ``first`` segment does not start where the candidate does was
    made from another transcript: that is an error, nothing is applied.
    """
    out = copy.deepcopy(transcript)
    segments = out["segments"]
    chosen = [c for c in candidates if float(c["p_other"]) >= min_prob and c["other_text"].strip()]
    for c in chosen:
        first, last = int(c["first"]), int(c["last"])
        if not 0 <= first <= last < len(segments) or abs(float(segments[first]["start"]) - float(c["start"])) > SAME_TIME:
            raise ValueError(f"the window at {c['start']} s does not match this transcript: the listing was made from another one")
    for c in sorted(chosen, key=lambda c: -int(c["first"])):
        first, last = int(c["first"]), int(c["last"])
        inside = segments[first : last + 1]
        held: dict[str | None, float] = {}
        for s in inside:
            held[s.get("speaker")] = held.get(s.get("speaker"), 0.0) + float(s["end"]) - float(s["start"])
        merged = {
            "start": c["start"],
            "end": c["end"],
            "text": c["other_text"].strip(),
            "speaker": max(held, key=lambda k: held[k]),
            "language": c["other"],
        }
        if all(s.get("filled") for s in inside):
            merged["filled"] = True
        segments[first : last + 1] = [merged]
    out["mixed"] = {"min_prob": min_prob, "candidates": len(candidates), "replaced": len(chosen)}
    return out


def write_candidates(candidates: Sequence[dict[str, Any]], path: Path | str) -> None:
    Path(path).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in candidates), encoding="utf-8")


def read_candidates(path: Path | str) -> list[dict[str, Any]]:
    """The listing back from disk, as edited: blank lines are skipped."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sibling(transcript_path: Path | str, suffix: str) -> Path:
    path = Path(transcript_path)
    stem = path.name[: -len(".transcript.json")] if path.name.endswith(".transcript.json") else path.stem
    return path.with_name(stem + suffix)


def default_candidates_path(transcript_path: Path | str) -> Path:
    """``X.transcript.json`` -> ``X.mixed.jsonl``."""
    return _sibling(transcript_path, ".mixed.jsonl")


def default_output(transcript_path: Path | str) -> Path:
    """``X.transcript.json`` -> ``X.mixed.transcript.json``; anything else ``X.json`` -> ``X.mixed.json``."""
    path = Path(transcript_path)
    if path.name.endswith(".transcript.json"):
        return _sibling(path, ".mixed.transcript.json")
    return path.with_name(path.stem + ".mixed" + path.suffix)
