"""audio -> transcript.json with WhisperX.

- model ``large-v3``, language given by flag (English by default);
- speech found by silero VAD by default: WhisperX's own default, pyannote,
  left 20-35 s stretches of phone-in-the-room speech out of every chunk list
  (a third of one recording); ``--vad pyannote`` brings it back;
- word-level alignment (the point of WhisperX): segment times come from it;
- diarization with ``pyannote/speaker-diarization-3.1`` when a HuggingFace
  token is available (``HF_TOKEN`` in the environment or ``.env``). No token
  is a *mode*, not an error: the step runs without speaker labels and says so;
- loops (one sentence written 4+ times in a row) and empty segments are
  dropped on the way out (``local_transcribe.clean``).

Two hours of audio should take minutes on the RTX 4090. If the GPU runs out of
memory, the batch size is halved and the run retried; the model stays large-v3.
Without a GPU the same code runs on the CPU with int8 weights, at about 0.6 of
the recording's length on an i9-13900K (see README).

All heavy imports are inside functions so the rest of the CLI stays light.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

DEFAULT_MODEL = "large-v3"
DEFAULT_BATCH_SIZE = 16
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SAMPLE_RATE = 16000
VAD_METHODS = ("silero", "pyannote")
DEFAULT_VAD = "silero"

# Where `winget install Gyan.FFmpeg` puts its binaries. A shell opened before
# the install does not have it on PATH yet; we look there as a fallback.
WINGET_PACKAGES = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"

# The folder that holds pyproject.toml and .env when the code runs from a clone.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

T = TypeVar("T")


def _log(message: str) -> None:
    """A progress line on stderr, with the clock: the stages are long and silent."""
    import time

    print(f"{time.strftime('%H:%M:%S')} {message}", file=sys.stderr, flush=True)


def ensure_ffmpeg() -> Path:
    """Return the ffmpeg binary, adding the winget install to PATH if needed."""
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    if WINGET_PACKAGES.is_dir():
        for candidate in sorted(WINGET_PACKAGES.glob(f"Gyan.FFmpeg*/**/bin/{exe}")):
            os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
            return candidate
    raise FileNotFoundError(
        "ffmpeg not found on PATH. Install it (winget install Gyan.FFmpeg, brew install ffmpeg, "
        "apt install ffmpeg) and open a new shell."
    )


def load_hf_token(dotenv_path: Path | str | None = None) -> str | None:
    """HF_TOKEN from the environment, else from .env. None when absent or blank.

    ``.env`` is looked for in the current directory and above it, then next to
    pyproject.toml, so ``uv run --project <clone>`` from another folder finds it too.
    """
    from dotenv import dotenv_values, find_dotenv

    token = os.environ.get("HF_TOKEN")
    if token is None:
        if dotenv_path is not None:
            places = [Path(dotenv_path)]
        else:
            places = [Path(p) for p in (find_dotenv(usecwd=True), PROJECT_ROOT / ".env") if p]
        for path in places:
            if path.is_file():
                token = dotenv_values(path).get("HF_TOKEN")
                if (token or "").strip():
                    break
    token = (token or "").strip()
    return token or None


def with_batch_backoff(
    run: Callable[[int], T],
    batch_size: int,
    oom_errors: tuple[type[BaseException], ...],
) -> T:
    """Call ``run(batch_size)``, halving the batch size on out-of-memory errors."""
    size = batch_size
    while True:
        try:
            return run(size)
        except oom_errors:
            if size <= 1:
                raise
            _log(f"out of GPU memory at batch size {size}; retrying with {size // 2}")
            size //= 2


def restore_torch_threads(wanted: int, log: Callable[[str], None] = _log) -> int:
    """Give torch its CPU threads back; returns the count now in force. Never lowers it.

    silero-vad runs ``torch.set_num_threads(1)`` when it is imported, for the whole
    process, so on CPU the alignment and the diarization after it ran on one core.
    """
    import torch

    if torch.get_num_threads() < wanted:
        torch.set_num_threads(wanted)
        log(f"torch back to {wanted} CPU threads (silero-vad had left it at 1)")
    return torch.get_num_threads()


def to_transcript(
    *,
    audio: Path | str,
    language: str,
    duration: float,
    segments: Sequence[dict[str, Any]],
    diarized: bool,
) -> dict[str, Any]:
    """The transcript.json shape, with loops and empty segments dropped (``local_transcribe.clean``)."""
    from local_transcribe.clean import clean

    return clean(
        {
            "audio": str(audio),
            "language": language,
            "duration": round(float(duration), 3),
            "diarized": diarized,
            "segments": [
                {
                    "start": round(float(seg["start"]), 3),
                    "end": round(float(seg["end"]), 3),
                    "text": (seg.get("text") or "").strip(),
                    "speaker": seg.get("speaker"),
                }
                for seg in segments
            ],
        }
    )


def transcribe(
    audio: Path | str,
    *,
    language: str = "en",
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    diarize: bool = True,
    hf_token: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    vad: str = DEFAULT_VAD,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    log: Callable[[str], None] = _log,
) -> dict[str, Any]:
    if vad not in VAD_METHODS:
        raise ValueError(f"vad must be one of {', '.join(VAD_METHODS)}, got {vad!r}")
    ensure_ffmpeg()
    import gc

    import torch

    torch_threads = torch.get_num_threads()  # read before whisperx loads silero-vad, which sets it to 1

    import whisperx
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    log(f"device: {device} ({compute_type})")

    wav = whisperx.load_audio(str(audio))
    duration = len(wav) / SAMPLE_RATE
    log(f"audio: {audio} ({duration / 60:.1f} min)")

    log(f"loading {model_name} (VAD: {vad})")
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language, vad_method=vad)

    def run(size: int) -> dict[str, Any]:
        if device == "cuda":
            torch.cuda.empty_cache()
        return model.transcribe(wav, batch_size=size, language=language)

    log(f"transcribing (batch size {batch_size})")
    result = with_batch_backoff(run, batch_size, oom_errors=(torch.cuda.OutOfMemoryError,))
    language = result.get("language") or language

    if device == "cpu":  # the GPU path is left exactly as it was
        restore_torch_threads(torch_threads, log)

    log("aligning words")
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, wav, device, return_char_alignments=False)

    diarized = False
    if diarize and hf_token:
        log(f"diarizing with {DIARIZATION_MODEL}")
        pipeline = DiarizationPipeline(model_name=DIARIZATION_MODEL, token=hf_token, device=device)
        diarization = pipeline(wav, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        result = assign_word_speakers(diarization, result)
        diarized = True
    elif diarize:
        log("HF_TOKEN not set: continuing without diarization (no speaker labels). Put the token in .env to enable it.")
    else:
        log("diarization disabled (--no-diarize)")

    del model, align_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return to_transcript(audio=audio, language=language, duration=duration, segments=result["segments"], diarized=diarized)
