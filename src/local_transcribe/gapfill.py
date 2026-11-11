"""Re-transcribe the holes a batched WhisperX run left in transcript.json.

    local-transcribe gapfill TRANSCRIPT.json [-o OUT.json]   -> OUT.json (default: X.gapfill.transcript.json)

On room recordings from a phone the batched pass sometimes keeps only half
the speech: whole stretches of 20-35 s vanish mid-sentence while the audio
there is as loud as the rest. Every gap of at least ``min_gap`` seconds
between segments (and before the first, after the last) is cut into windows
of at most ``max_window`` seconds, and each window is transcribed on its own
by faster-whisper with VAD off. What comes back is merged into the
transcript with ``speaker: null`` and ``filled: true``; the original
segments are kept as they were.

Whisper invents text on silence and loops on noise, so a filled segment is
kept only if it looks like speech: not likely-silent *and* low-confidence at
once, not over-compressible (a repetition loop), at least 0.2 s long, and not
one of the stock phrases it says to a quiet window ("Thank you.", "Thanks for
watching!") standing alone. The merged transcript then goes through
``local_transcribe.clean``: loops across segments and windows that re-read the edge of
their neighbour are dropped there.

All heavy imports are inside functions so the rest of the CLI stays light.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Iterable

from local_transcribe.clean import STOCK_PHRASE, clean
from local_transcribe.transcribe import DEFAULT_MODEL, SAMPLE_RATE, _log, ensure_ffmpeg

DEFAULT_MIN_GAP = 1.5
DEFAULT_MAX_WINDOW = 28.0
MIN_WINDOW = 0.5
MIN_SEGMENT = 0.2
NO_SPEECH_PROB = 0.6
LOW_LOGPROB = -0.8
MAX_COMPRESSION = 2.4


def coverage(segments: Iterable[dict[str, Any]]) -> float:
    """Seconds of audio under segments (overlaps counted twice; segments rarely overlap)."""
    return sum(max(0.0, float(s["end"]) - float(s["start"])) for s in segments)


def find_gaps(segments: list[dict[str, Any]], duration: float, min_gap: float = DEFAULT_MIN_GAP) -> list[tuple[float, float]]:
    """Stretches of at least ``min_gap`` s that no segment covers, from 0 to ``duration``."""
    gaps: list[tuple[float, float]] = []
    covered_to = 0.0
    for s in sorted(segments, key=lambda s: float(s["start"])) + [{"start": duration, "end": duration}]:
        start = float(s["start"])
        if start - covered_to >= min_gap:
            gaps.append((covered_to, start))
        covered_to = max(covered_to, float(s["end"]))
    return gaps


def split_windows(gaps: list[tuple[float, float]], max_window: float = DEFAULT_MAX_WINDOW) -> list[tuple[float, float]]:
    """Cut each gap into windows of at most ``max_window`` s; a tail under 0.5 s is dropped."""
    windows: list[tuple[float, float]] = []
    for g0, g1 in gaps:
        t = g0
        while g1 - t >= MIN_WINDOW:
            end = min(g1, t + max_window)
            windows.append((t, end))
            t = end
    return windows


def keep_segment(
    text: str,
    *,
    start: float,
    end: float,
    no_speech_prob: float,
    avg_logprob: float,
    compression_ratio: float,
) -> bool:
    """Whether a re-transcribed segment looks like speech rather than a hallucination."""
    if not text.strip() or end - start < MIN_SEGMENT or STOCK_PHRASE.match(text.strip()):
        return False
    if no_speech_prob > NO_SPEECH_PROB and avg_logprob < LOW_LOGPROB:
        return False
    return compression_ratio <= MAX_COMPRESSION


def merge_filled(
    transcript: dict[str, Any],
    added: list[dict[str, Any]],
    *,
    model_name: str,
    min_gap: float,
    windows: int,
) -> dict[str, Any]:
    """A copy of ``transcript`` with ``added`` merged in by start time, cleaned, and a ``gapfill`` record."""
    out = copy.deepcopy(transcript)
    filled = [
        {"start": round(float(a["start"]), 3), "end": round(float(a["end"]), 3), "text": a["text"].strip(), "speaker": None, "filled": True}
        for a in added
    ]
    out["segments"] = sorted(out["segments"] + filled, key=lambda s: float(s["start"]))
    out = clean(out)
    kept = sum(1 for s in out["segments"] if s.get("filled"))
    out["gapfill"] = {"model": model_name, "min_gap": min_gap, "windows": windows, "added": kept}
    return out


def default_output(transcript_path: Path | str) -> Path:
    """``X.transcript.json`` -> ``X.gapfill.transcript.json``; anything else ``X.json`` -> ``X.gapfill.json``."""
    path = Path(transcript_path)
    if path.name.endswith(".transcript.json"):
        return path.with_name(path.name[: -len(".transcript.json")] + ".gapfill.transcript.json")
    return path.with_name(path.stem + ".gapfill" + path.suffix)


def gapfill(
    transcript: dict[str, Any],
    *,
    audio: Path | str | None = None,
    model_name: str = DEFAULT_MODEL,
    min_gap: float = DEFAULT_MIN_GAP,
    max_window: float = DEFAULT_MAX_WINDOW,
    device: str | None = None,
    compute_type: str | None = None,
    log: Callable[[str], None] = _log,
) -> dict[str, Any]:
    """Re-transcribe every gap of ``transcript`` and return the merged transcript."""
    ensure_ffmpeg()
    import torch
    import whisperx
    from faster_whisper import WhisperModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    segments = transcript["segments"]
    duration = float(transcript["duration"])
    gaps = find_gaps(segments, duration, min_gap)
    windows = split_windows(gaps, max_window)
    log(
        f"covered {coverage(segments) / 60:.1f} of {duration / 60:.1f} min; "
        f"{len(gaps)} gaps ({sum(b - a for a, b in gaps) / 60:.1f} min) -> {len(windows)} windows"
    )

    wav = whisperx.load_audio(str(audio or transcript["audio"]))
    log(f"loading {model_name} ({device}, {compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    language = transcript.get("language") or "en"

    added: list[dict[str, Any]] = []
    for i, (w0, w1) in enumerate(windows, 1):
        chunk = wav[int(w0 * SAMPLE_RATE) : int(w1 * SAMPLE_RATE)]
        decoded, _ = model.transcribe(
            chunk, language=language, vad_filter=False, beam_size=5, condition_on_previous_text=False, temperature=0.0
        )
        for s in decoded:
            start, end = w0 + s.start, min(w1, w0 + s.end)
            if keep_segment(
                s.text,
                start=start,
                end=end,
                no_speech_prob=s.no_speech_prob,
                avg_logprob=s.avg_logprob,
                compression_ratio=s.compression_ratio,
            ):
                added.append({"start": start, "end": end, "text": s.text})
        if i % 25 == 0:
            log(f"  {i}/{len(windows)} windows, {len(added)} segments added")

    out = merge_filled(transcript, added, model_name=model_name, min_gap=min_gap, windows=len(windows))
    log(f"added {len(added)}; covered now {coverage(out['segments']) / 60:.1f} min")
    return out
