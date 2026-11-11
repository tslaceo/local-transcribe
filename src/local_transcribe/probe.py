"""Short model probes: which language is spoken where, and a second opinion on a doubtful spot.

    local-transcribe langs AUDIO [--step 120]
    local-transcribe redecode AUDIO --at 0:53:50 [--at ...] [--langs ru,en] [--prompt "..."]

``langs`` asks Whisper for the language of a 30 s window every ``step`` seconds. Run it
before ``transcribe`` to choose ``--lang``: a run decodes in one language, and a recording
that mixes two has to be run in the one that carries it (see ``local_transcribe.mixed`` for
the other).

``redecode`` transcribes a short window around a timestamp again, once per language asked
for and, with ``--prompt``, once more in the first language with that prompt. A second full
pass with the same model repeats most of its own mistakes, because the audio and the chunk
boundaries are the same. A window cut at other places, another language, or a vocabulary
prompt gives a reading that is actually independent. A prompt pulls the model towards its
own words: a prompted reading is a hint, never the tie-breaker.

All heavy imports are inside functions so the rest of the CLI stays light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from local_transcribe.transcribe import DEFAULT_MODEL, SAMPLE_RATE, _log, ensure_ffmpeg

WINDOW = 30.0
DEFAULT_STEP = 120.0
DEFAULT_BEFORE = 5.0
DEFAULT_LENGTH = 18.0
SILENCE = 1e-3


def parse_time(text: str) -> float:
    """``"5025"``, ``"83:45"`` or ``"1:23:45"`` -> seconds."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(p.replace(".", "", 1).isdigit() for p in parts):
        raise ValueError(f"not a time: {text!r} (use seconds, m:ss or h:mm:ss)")
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + float(p)
    return seconds


def format_time(seconds: float) -> str:
    """Seconds -> ``h:mm:ss``."""
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}"


def summarize_languages(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Share of the non-silent windows each language won, largest first."""
    heard = [r["language"] for r in rows if r.get("language")]
    if not heard:
        return {}
    shares = {lang: heard.count(lang) / len(heard) for lang in set(heard)}
    return dict(sorted(shares.items(), key=lambda kv: -kv[1]))


def load_whisper(model_name: str = DEFAULT_MODEL, device: str | None = None, compute_type: str | None = None):
    """A faster-whisper model on the GPU if CTranslate2 sees one, else on the CPU with int8 weights."""
    import ctranslate2
    from faster_whisper import WhisperModel

    if device is None:
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_name, device=device, compute_type=compute_type), device


def language_profile(
    audio: Path | str,
    *,
    step: float = DEFAULT_STEP,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    log: Callable[[str], None] = _log,
) -> list[dict[str, Any]]:
    """The language Whisper hears in a 30 s window every ``step`` seconds of ``audio``."""
    ensure_ffmpeg()
    import numpy as np
    import whisperx

    wav = whisperx.load_audio(str(audio))
    model, device = load_whisper(model_name, device)
    log(f"language of a {WINDOW:.0f} s window every {step:.0f} s ({device})")
    rows: list[dict[str, Any]] = []
    t = 0.0
    while t * SAMPLE_RATE < len(wav):
        chunk = wav[int(t * SAMPLE_RATE) : int((t + WINDOW) * SAMPLE_RATE)]
        if len(chunk) < SAMPLE_RATE or float(np.abs(chunk).mean()) < SILENCE:
            rows.append({"start": t, "language": None, "probabilities": {}})
        else:
            language, _, scores = model.detect_language(chunk)
            top = sorted(scores, key=lambda kv: -kv[1])[:3]
            rows.append({"start": t, "language": language, "probabilities": {k: round(float(v), 3) for k, v in top}})
        t += step
    return rows


def redecode(
    audio: Path | str,
    spots: Sequence[float],
    *,
    languages: Sequence[str],
    prompt: str | None = None,
    before: float = DEFAULT_BEFORE,
    length: float = DEFAULT_LENGTH,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    log: Callable[[str], None] = _log,
) -> list[dict[str, Any]]:
    """Transcribe ``length`` s starting ``before`` s ahead of each spot, once per language (+ once prompted)."""
    if not languages:
        raise ValueError("give at least one language")
    ensure_ffmpeg()
    import whisperx

    wav = whisperx.load_audio(str(audio))
    model, device = load_whisper(model_name, device)
    log(f"re-decoding {len(spots)} spots in {', '.join(languages)}{' + prompt' if prompt else ''} ({device})")

    def read(chunk, language: str, initial_prompt: str | None) -> str:
        segments, _ = model.transcribe(
            chunk, language=language, beam_size=5, vad_filter=False, condition_on_previous_text=False, initial_prompt=initial_prompt
        )
        return " ".join(s.text.strip() for s in segments)

    out: list[dict[str, Any]] = []
    for at in spots:
        start = max(0.0, at - before)
        chunk = wav[int(start * SAMPLE_RATE) : int((start + length) * SAMPLE_RATE)]
        readings = [{"language": lang, "prompted": False, "text": read(chunk, lang, None)} for lang in languages]
        if prompt:
            readings.insert(1, {"language": languages[0], "prompted": True, "text": read(chunk, languages[0], prompt)})
        out.append({"at": at, "start": start, "end": start + length, "readings": readings})
    return out


def format_redecode(results: Sequence[dict[str, Any]]) -> str:
    """The readings as text: one block per spot, one line per reading."""
    blocks = []
    for r in results:
        lines = [f"=== {format_time(r['at'])} (window {format_time(r['start'])}-{format_time(r['end'])})"]
        for reading in r["readings"]:
            tag = reading["language"] + ("+prompt" if reading["prompted"] else "")
            lines.append(f"  {tag:<10}: {reading['text']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
