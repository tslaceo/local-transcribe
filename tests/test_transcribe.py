"""Tests for local_transcribe.transcribe.

Fast tests cover the parts around the model: token discovery, output shape,
batch-size backoff, ffmpeg lookup. The end-to-end run on a synthesised speech
sample is marked slow (GPU + model download).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from local_transcribe.transcribe import (
    DIARIZATION_MODEL,
    ensure_ffmpeg,
    load_hf_token,
    to_transcript,
    with_batch_backoff,
)


def test_load_hf_token_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert load_hf_token(dotenv_path=tmp_path / ".env") == "hf_env"


def test_load_hf_token_from_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_file\n", encoding="utf-8")
    assert load_hf_token(dotenv_path=tmp_path / ".env") == "hf_file"


def test_missing_token_is_none_not_an_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No token means 'no diarization', never a crash."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert load_hf_token(dotenv_path=tmp_path / ".env") is None
    monkeypatch.setenv("HF_TOKEN", "   ")
    assert load_hf_token(dotenv_path=tmp_path / ".env") is None


def test_no_token_is_hardcoded_anywhere() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root / "src").rglob("*.py"):
        assert "hf_" not in path.read_text(encoding="utf-8").replace("hf_token", "").replace("HF_TOKEN", ""), path


def test_to_transcript_shape() -> None:
    raw = [
        {"start": 12.3, "end": 18.9, "text": " Hello there. ", "speaker": "SPEAKER_00", "words": [{"word": "Hello"}]},
        {"start": 19.0, "end": 20.0, "text": "Yes."},
    ]
    out = to_transcript(audio="meeting.m4a", language="en", duration=7180.4, segments=raw, diarized=True)
    assert out["audio"] == "meeting.m4a"
    assert out["language"] == "en"
    assert out["duration"] == 7180.4
    assert out["diarized"] is True
    assert out["segments"] == [
        {"start": 12.3, "end": 18.9, "text": "Hello there.", "speaker": "SPEAKER_00"},
        {"start": 19.0, "end": 20.0, "text": "Yes.", "speaker": None},
    ]


def test_to_transcript_drops_loops_and_empty_segments() -> None:
    """The batched pass loops too (mirrors a real case: one sentence nine times in a row)."""
    raw = [{"start": float(i), "end": i + 1.0, "text": "I'm going to check with the other team.", "speaker": "SPEAKER_01"} for i in range(9)]
    raw += [{"start": 9.0, "end": 9.5, "text": "."}, {"start": 10.0, "end": 14.0, "text": "Back to the agenda.", "speaker": "SPEAKER_01"}]
    out = to_transcript(audio="meeting.m4a", language="en", duration=60.0, segments=raw, diarized=True)
    assert [s["text"] for s in out["segments"]] == ["I'm going to check with the other team.", "Back to the agenda."]
    assert out["clean"] == {"empty": 1, "loops": 8, "stock": 0, "duplicates": 0}


def test_batch_backoff_halves_on_oom_and_gives_up_at_zero() -> None:
    class FakeOOM(RuntimeError):
        pass

    calls: list[int] = []

    def run(batch_size: int) -> str:
        calls.append(batch_size)
        if batch_size > 4:
            raise FakeOOM("CUDA out of memory")
        return "ok"

    assert with_batch_backoff(run, 16, oom_errors=(FakeOOM,)) == "ok"
    assert calls == [16, 8, 4]

    def always(batch_size: int) -> str:
        raise FakeOOM("no")

    with pytest.raises(FakeOOM):
        with_batch_backoff(always, 2, oom_errors=(FakeOOM,))


def test_batch_backoff_does_not_swallow_other_errors() -> None:
    def run(batch_size: int) -> str:
        raise ValueError("unrelated")

    with pytest.raises(ValueError):
        with_batch_backoff(run, 8, oom_errors=(MemoryError,))


def test_restore_torch_threads_undoes_silero_and_never_lowers() -> None:
    import torch

    from local_transcribe.transcribe import restore_torch_threads

    before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)  # what importing silero-vad does to the whole process
        assert restore_torch_threads(4, log=lambda _: None) == 4
        assert torch.get_num_threads() == 4
        assert restore_torch_threads(2, log=lambda _: None) == 4
    finally:
        torch.set_num_threads(before)


def test_ensure_ffmpeg_finds_winget_install_when_not_on_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_bin = tmp_path / "Packages" / "Gyan.FFmpeg_x" / "ffmpeg-9.0.1-full_build" / "bin"
    fake_bin.mkdir(parents=True)
    exe = fake_bin / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    exe.write_bytes(b"")
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    monkeypatch.setattr("local_transcribe.transcribe.WINGET_PACKAGES", tmp_path / "Packages")
    found = ensure_ffmpeg()
    assert found == exe
    assert str(fake_bin) in os.environ["PATH"]


def test_ensure_ffmpeg_missing_is_a_clear_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("local_transcribe.transcribe.WINGET_PACKAGES", tmp_path / "none")
    with pytest.raises(FileNotFoundError, match="ffmpeg"):
        ensure_ffmpeg()


def test_diarization_model_is_the_one_from_the_design() -> None:
    assert DIARIZATION_MODEL == "pyannote/speaker-diarization-3.1"


def test_silero_is_the_default_vad() -> None:
    """pyannote's VAD dropped 20-35 s stretches of room speech; silero kept them."""
    from local_transcribe.cli import build_parser
    from local_transcribe.transcribe import DEFAULT_VAD, VAD_METHODS

    assert DEFAULT_VAD == "silero"
    assert set(VAD_METHODS) == {"silero", "pyannote"}
    assert build_parser().parse_args(["transcribe", "a.wav"]).vad == "silero"
    assert build_parser().parse_args(["transcribe", "a.wav", "--vad", "pyannote"]).vad == "pyannote"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["transcribe", "a.wav", "--vad", "webrtc"])


# --- slow: real model on the GPU ----------------------------------------------


@pytest.mark.slow
def test_transcribe_end_to_end_without_diarization(speech_wav: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from local_transcribe.transcribe import transcribe

    monkeypatch.delenv("HF_TOKEN", raising=False)
    result = transcribe(speech_wav, language="en", diarize=True, hf_token=None)
    assert result["language"] == "en"
    assert result["duration"] > 3
    assert result["diarized"] is False
    text = " ".join(seg["text"] for seg in result["segments"]).lower()
    assert "quarterly budget" in text
    assert all(seg["speaker"] is None for seg in result["segments"])
    assert all(seg["start"] <= seg["end"] for seg in result["segments"])


@pytest.mark.slow
def test_transcribe_end_to_end_with_diarization_if_token_present(speech_wav: Path) -> None:
    from local_transcribe.transcribe import transcribe

    token = load_hf_token()
    if not token:
        pytest.skip("HF_TOKEN not set; diarization cannot be exercised")
    result = transcribe(speech_wav, language="en", diarize=True, hf_token=token)
    assert result["diarized"] is True
    assert any(seg["speaker"] for seg in result["segments"])
