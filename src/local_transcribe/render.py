"""transcript.json -> text a person can read and correct.

    local-transcribe render TRANSCRIPT.json [-o OUT.txt] [--name SPEAKER_00=Alice ...]

Each paragraph is ``[h:mm:ss] SPEAKER: text``. A new paragraph starts when the speaker
changes and, within one speaker, about every ``--max-paragraph`` seconds (60 by default; 0
for one paragraph per turn), so that every stretch of text has a timestamp near it to go
back to the audio with. Segments added by gapfill carry no speaker and come out as
``FILLED``: who said them is decided from the context, by whoever corrects the text.
Segments the diarizer left unlabelled come out as ``UNKNOWN``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from local_transcribe.probe import format_time

FILLED = "FILLED"
UNKNOWN = "UNKNOWN"
DEFAULT_MAX_PARAGRAPH = 60.0


def speaker_label(segment: Mapping[str, Any]) -> str:
    return segment.get("speaker") or (FILLED if segment.get("filled") else UNKNOWN)


def turns(transcript: Mapping[str, Any], max_seconds: float = 0.0) -> list[dict[str, Any]]:
    """Consecutive segments of one speaker joined into ``{"start", "speaker", "text"}``.

    With ``max_seconds`` above 0 a paragraph also ends at the first segment that starts
    that long after the paragraph did.
    """
    out: list[dict[str, Any]] = []
    for s in transcript["segments"]:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        label, start = speaker_label(s), float(s["start"])
        same = out and out[-1]["speaker"] == label and not (max_seconds > 0 and start - out[-1]["start"] >= max_seconds)
        if same:
            out[-1]["text"] += " " + text
        else:
            out.append({"start": start, "speaker": label, "text": text})
    return out


def speaker_stats(transcript: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Per speaker label: number of segments and minutes of speech, most minutes first."""
    stats: dict[str, dict[str, float]] = {}
    for s in transcript["segments"]:
        entry = stats.setdefault(speaker_label(s), {"segments": 0, "minutes": 0.0})
        entry["segments"] += 1
        entry["minutes"] += (float(s["end"]) - float(s["start"])) / 60
    for entry in stats.values():
        entry["minutes"] = round(entry["minutes"], 1)
    return dict(sorted(stats.items(), key=lambda kv: -kv[1]["minutes"]))


def unlabelled_share(transcript: Mapping[str, Any]) -> float:
    """The share of the transcribed time that has no speaker: ``FILLED`` plus ``UNKNOWN``."""
    total = unlabelled = 0.0
    for s in transcript["segments"]:
        length = float(s["end"]) - float(s["start"])
        total += length
        if speaker_label(s) in (FILLED, UNKNOWN):
            unlabelled += length
    return unlabelled / total if total else 0.0


def to_text(transcript: Mapping[str, Any], names: Mapping[str, str] | None = None, max_seconds: float = 0.0) -> str:
    """The transcript as paragraphs, with speaker labels replaced by ``names`` where given."""
    names = names or {}
    paragraphs = [f"[{format_time(t['start'])}] {names.get(t['speaker'], t['speaker'])}: {t['text']}" for t in turns(transcript, max_seconds)]
    return "\n\n".join(paragraphs) + "\n"


def parse_names(pairs: list[str] | None) -> dict[str, str]:
    """``["SPEAKER_00=Alice", ...]`` -> ``{"SPEAKER_00": "Alice", ...}``."""
    names: dict[str, str] = {}
    for pair in pairs or []:
        label, sep, name = pair.partition("=")
        if not sep or not label or not name:
            raise ValueError(f"expected LABEL=Name, got {pair!r}")
        names[label] = name
    return names


def default_output(transcript_path: Path | str) -> Path:
    """``X.transcript.json`` -> ``X.txt``; anything else ``X.json`` -> ``X.txt``."""
    path = Path(transcript_path)
    stem = path.name[: -len(".transcript.json")] if path.name.endswith(".transcript.json") else path.stem
    return path.with_name(stem + ".txt")
