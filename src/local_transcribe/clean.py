"""Drop the junk a transcription pass leaves; never rewrite what stays.

    local-transcribe clean TRANSCRIPT.json [-o OUT.json]   -> OUT.json (default: X.clean.transcript.json)

Four rules, from an audit of 42 room recordings:

- empty: a segment with no word in it ("." twelve times in a row);
- stock: a filled segment that is nothing but one of Whisper's phrases for a
  quiet window ("Thank you for watching.", "We'll be right back.", and in
  Russian "Субтитры сделал DimaTorzok", "Продолжение следует..."). In the
  diarized pass a lone "Thank you." is usually someone finishing a talk, so
  only filled segments are judged;
- duplicates: a filled segment of four words or more whose word trigrams are
  at least half found in unfilled segments within 30 s of it. The gap window
  started where a neighbour's words were cut short and read them again; 8-27 %
  of what gapfill added was such copies;
- loops: four or more identical segments in a row collapse to the first. Both
  passes loop (one sentence written 19 times running).

A segment dropped by an earlier rule is counted there only. Kept segments are
returned exactly as they came in. ``transcribe`` and ``gapfill`` apply this to
what they write; the command cleans transcripts made before.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

LOOP_RUN = 4
DUP_MIN_WORDS = 4
DUP_SHARE = 0.5
DUP_WINDOW = 30.0
STOCK_PHRASE = re.compile(
    r"^(thank you( very much| so much| for watching| for listening|,? guys)?"
    r"|thanks(,? guys| for watching)?|bye(-bye)?|you|you're welcome"
    r"|we'll be right back|god bless you|i'll see you next time"
    r"|субтитры (сделал|создавал|делал) \w+|продолжение следует|спасибо за просмотр)[.!?…]*$",
    re.IGNORECASE,
)
WORD = re.compile(r"\w+(?:'\w+)*")


def _trigrams(text: str) -> tuple[set[tuple[str, ...]], int]:
    words = WORD.findall(text.lower())
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}, len(words)


def is_edge_duplicate(segment: dict[str, Any], unfilled: list[dict[str, Any]]) -> bool:
    """Whether a filled segment mostly repeats unfilled speech within ``DUP_WINDOW`` s of its edges."""
    grams, n_words = _trigrams(segment["text"])
    if n_words < DUP_MIN_WORDS:
        return False
    start, end = float(segment["start"]), float(segment["end"])
    near: set[tuple[str, ...]] = set()
    for u in unfilled:
        if float(u["start"]) - end <= DUP_WINDOW and start - float(u["end"]) <= DUP_WINDOW:
            near |= _trigrams(u["text"])[0]
    return len(grams & near) >= DUP_SHARE * len(grams)


def clean(transcript: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``transcript`` without junk segments and with a ``clean`` record of what went."""
    out = copy.deepcopy(transcript)
    counts = {"empty": 0, "loops": 0, "stock": 0, "duplicates": 0}
    unfilled = [s for s in out["segments"] if not s.get("filled")]

    kept: list[dict[str, Any]] = []
    for s in out["segments"]:
        text = (s.get("text") or "").strip()
        if not WORD.search(text):
            counts["empty"] += 1
        elif s.get("filled") and STOCK_PHRASE.match(text):
            counts["stock"] += 1
        elif s.get("filled") and is_edge_duplicate(s, unfilled):
            counts["duplicates"] += 1
        else:
            kept.append(s)

    segments: list[dict[str, Any]] = []
    i = 0
    while i < len(kept):
        key = kept[i]["text"].strip().lower()
        j = i
        while j + 1 < len(kept) and kept[j + 1]["text"].strip().lower() == key:
            j += 1
        if j - i + 1 >= LOOP_RUN:
            segments.append(kept[i])
            counts["loops"] += j - i
        else:
            segments.extend(kept[i : j + 1])
        i = j + 1

    out["segments"] = segments
    out["clean"] = counts
    return out


def default_output(transcript_path: Path | str) -> Path:
    """``X.transcript.json`` -> ``X.clean.transcript.json``; anything else ``X.json`` -> ``X.clean.json``."""
    path = Path(transcript_path)
    if path.name.endswith(".transcript.json"):
        return path.with_name(path.name[: -len(".transcript.json")] + ".clean.transcript.json")
    return path.with_name(path.stem + ".clean" + path.suffix)
