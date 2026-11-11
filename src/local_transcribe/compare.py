"""How much two transcripts of the same audio disagree, and where.

    local-transcribe compare A.json B.json [-o REPORT.txt]

Two uses. After changing the device, the compute type, the model or a setting: is the new
transcript the same text, or only a similar one. And as a second opinion: every place where
two passes disagree is a place neither can be trusted.

What the numbers mean was measured once, on a 114-minute recording transcribed with the same
model on a GPU (float16) and on a CPU (int8): 4.7 % of the words disagreed, in both
directions and mostly where both passes were wrong (names, switches of language). Of 163
numbers, eight places differed and none of them had two different values. So a few percent
of disagreement between two honest passes is the noise floor, not a regression, and
agreement between them does not make a hard spot right.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Mapping, Sequence

from local_transcribe.probe import format_time

TOKEN = re.compile(r"[0-9]+(?:[.,][0-9]+)?|[^\W\d_]+(?:'[^\W\d_]+)*")


def tokens(segments: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str], list[float]]:
    """Words as written, the same words normalised for comparison, and the start of the segment each came from."""
    shown: list[str] = []
    norm: list[str] = []
    when: list[float] = []
    for s in segments:
        for m in TOKEN.finditer(s.get("text") or ""):
            shown.append(m.group(0))
            norm.append(m.group(0).lower().replace("ё", "е"))
            when.append(float(s["start"]))
    return shown, norm, when


def diff_counts(a: Sequence[str], b: Sequence[str]) -> dict[str, Any]:
    """Words the same, substituted, only in ``a``, only in ``b``; ``disagreement`` is their sum over ``len(a)``."""
    same = substituted = only_a = only_b = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            same += i2 - i1
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            substituted += n
            only_a += (i2 - i1) - n
            only_b += (j2 - j1) - n
        elif tag == "delete":
            only_a += i2 - i1
        else:
            only_b += j2 - j1
    disagreement = (substituted + only_a + only_b) / len(a) if a else 0.0
    return {
        "words_a": len(a),
        "words_b": len(b),
        "same": same,
        "substituted": substituted,
        "only_a": only_a,
        "only_b": only_b,
        "disagreement": round(disagreement, 4),
    }


def differences(
    a_segments: Sequence[Mapping[str, Any]], b_segments: Sequence[Mapping[str, Any]], context: int = 6
) -> list[dict[str, Any]]:
    """Every place the two differ, with its time in ``a`` and the words around it.

    Nothing is filtered out: a rule that hides "word endings" also hides 15000 against 15005.
    """
    a_shown, a_norm, a_when = tokens(a_segments)
    b_shown, b_norm, _ = tokens(b_segments)
    out: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a_norm, b_norm, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        out.append(
            {
                "time": a_when[min(i1, len(a_when) - 1)] if a_when else 0.0,
                "before": " ".join(a_shown[max(0, i1 - context) : i1]),
                "a": " ".join(a_shown[i1:i2]),
                "b": " ".join(b_shown[j1:j2]),
                "after": " ".join(a_shown[i2 : i2 + context]),
            }
        )
    return out


def report(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    """The comparison as text: sizes, the word counts, the differences that involve a number, then all of them."""
    counts = diff_counts(tokens(a["segments"])[1], tokens(b["segments"])[1])
    found = differences(a["segments"], b["segments"])
    with_numbers = [d for d in found if re.search(r"\d", d["a"] + d["b"])]

    def line(d: dict[str, Any]) -> str:
        return f"[{format_time(d['time'])}] …{d['before']} ⟦{d['a']} ⇒ {d['b']}⟧ {d['after']}…"

    n = max(1, counts["words_a"])
    lines = [
        f"A: {len(a['segments'])} segments, {counts['words_a']} words",
        f"B: {len(b['segments'])} segments, {counts['words_b']} words",
        f"same {counts['same']} ({counts['same'] / n:.1%}), substituted {counts['substituted']} ({counts['substituted'] / n:.1%}), "
        f"only in A {counts['only_a']} ({counts['only_a'] / n:.1%}), only in B {counts['only_b']} ({counts['only_b'] / n:.1%})",
        f"disagreement {counts['disagreement']:.1%} of A's words",
        "",
        f"differences involving a number: {len(with_numbers)}",
        *[line(d) for d in with_numbers],
        "",
        f"all differences: {len(found)}  (A ⇒ B)",
        *[line(d) for d in found],
    ]
    return "\n".join(lines) + "\n"
