"""Tests for local_transcribe.gapfill.

Fast tests cover the arithmetic around the model: where the holes are, how
they are cut into windows, which re-transcribed segments are kept, and the
shape of the output. The end-to-end run on a synthesised speech sample is
marked slow (GPU + model download).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_transcribe.gapfill import (
    coverage,
    default_output,
    find_gaps,
    keep_segment,
    merge_filled,
    split_windows,
)


def seg(start: float, end: float, text: str = "x", speaker: str | None = "SPEAKER_00") -> dict:
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def test_find_gaps_between_before_first_and_after_last() -> None:
    segments = [seg(3.0, 10.0), seg(10.5, 20.0), seg(40.0, 50.0)]
    assert find_gaps(segments, duration=60.0, min_gap=1.5) == [(0.0, 3.0), (20.0, 40.0), (50.0, 60.0)]


def test_find_gaps_ignores_short_pauses() -> None:
    segments = [seg(0.0, 10.0), seg(11.0, 20.0)]
    assert find_gaps(segments, duration=20.5, min_gap=1.5) == []


def test_find_gaps_measures_from_the_furthest_end_so_far() -> None:
    """A short segment inside a long one must not open a false hole."""
    segments = [seg(0.0, 30.0), seg(5.0, 6.0), seg(31.0, 40.0)]
    assert find_gaps(segments, duration=40.0, min_gap=1.5) == []


def test_find_gaps_on_an_empty_transcript_is_the_whole_recording() -> None:
    assert find_gaps([], duration=12.0, min_gap=1.5) == [(0.0, 12.0)]


def test_split_windows_cuts_long_gaps_and_drops_slivers() -> None:
    gaps = [(0.0, 3.0), (20.0, 80.2)]
    assert split_windows(gaps, max_window=28.0) == [(0.0, 3.0), (20.0, 48.0), (48.0, 76.0), (76.0, 80.2)]
    # a tail under half a second is not worth a model call
    assert split_windows([(0.0, 28.3)], max_window=28.0) == [(0.0, 28.0)]


def test_keep_segment_drops_the_usual_hallucinations() -> None:
    assert keep_segment("Today we look at the budget.", start=0.0, end=3.0, no_speech_prob=0.1, avg_logprob=-0.3, compression_ratio=1.2)
    assert not keep_segment("", start=0.0, end=3.0, no_speech_prob=0.1, avg_logprob=-0.3, compression_ratio=1.2)
    # silence read as words: likely no speech and a low-confidence decode
    assert not keep_segment("Thank you.", start=0.0, end=2.0, no_speech_prob=0.9, avg_logprob=-1.1, compression_ratio=0.8)
    # the repetition loop: the same phrase over and over compresses too well
    assert not keep_segment("the the the the the the the the the the", start=0.0, end=9.0, no_speech_prob=0.1, avg_logprob=-0.2, compression_ratio=3.1)
    # a blip shorter than 0.2 s
    assert not keep_segment("Yes.", start=5.0, end=5.1, no_speech_prob=0.1, avg_logprob=-0.2, compression_ratio=1.0)


def test_keep_segment_drops_stock_phrases_said_alone() -> None:
    """Whisper's favourite words for a quiet window; in one audit they were 6-22 % of what gapfill added."""
    for text in ["Thank you.", " thank you very much!", "Thanks for watching!", "Bye.", "you"]:
        assert not keep_segment(text, start=0.0, end=1.5, no_speech_prob=0.2, avg_logprob=-0.3, compression_ratio=0.9), text
    # the same words inside a real sentence stay
    assert keep_segment("Thank you for the question, it is a good one.", start=0.0, end=3.0, no_speech_prob=0.2, avg_logprob=-0.3, compression_ratio=1.1)


def test_keep_segment_drops_the_stock_phrases_the_audit_found() -> None:
    """In an audit of 42 recordings "Thank you for watching." alone was 73 filled segments."""
    for text in ["Thank you for watching.", "We'll be right back.", "Thank you so much.", "Thanks, guys.", "God bless you.",
                 "I'll see you next time.", "Thank you for listening.", "Bye-bye.", "You're welcome.", "Thank you, guys."]:
        assert not keep_segment(text, start=0.0, end=1.5, no_speech_prob=0.2, avg_logprob=-0.3, compression_ratio=0.9), text


def test_keep_segment_keeps_quiet_but_confident_speech() -> None:
    """High no-speech probability alone is not enough to drop a confident decode."""
    assert keep_segment("Any other questions?", start=0.0, end=2.0, no_speech_prob=0.7, avg_logprob=-0.4, compression_ratio=1.0)


def test_merge_filled_marks_added_segments_and_keeps_order() -> None:
    transcript = {"audio": "a.wav", "language": "en", "duration": 60.0, "diarized": True,
                  "segments": [seg(0.0, 10.0), seg(40.0, 50.0)]}
    added = [{"start": 12.0, "end": 20.0, "text": "filled"}]
    out = merge_filled(transcript, added, model_name="large-v3", min_gap=1.5, windows=3)
    assert [s["start"] for s in out["segments"]] == [0.0, 12.0, 40.0]
    assert out["segments"][1] == {"start": 12.0, "end": 20.0, "text": "filled", "speaker": None, "filled": True}
    assert "filled" not in out["segments"][0]
    assert out["gapfill"] == {"model": "large-v3", "min_gap": 1.5, "windows": 3, "added": 1}
    assert out["diarized"] is True and out["duration"] == 60.0
    assert len(transcript["segments"]) == 2  # input left alone


def test_merge_filled_drops_windows_that_reread_their_neighbour() -> None:
    """A gap window starts where the last segment's words were cut short and reads them again."""
    transcript = {"audio": "a.wav", "language": "en", "duration": 60.0, "diarized": True,
                  "segments": [seg(0.0, 10.0, "But with that, I thank you for your attention and see you next week.")]}
    added = [{"start": 10.5, "end": 13.0, "text": "I thank you for your attention and see you"},
             {"start": 30.0, "end": 34.0, "text": "One more thing about the deadline."}]
    out = merge_filled(transcript, added, model_name="large-v3", min_gap=1.5, windows=2)
    assert [s["text"] for s in out["segments"]] == [transcript["segments"][0]["text"], "One more thing about the deadline."]
    assert out["gapfill"]["added"] == 1
    assert out["clean"]["duplicates"] == 1


def test_coverage_sums_segment_time() -> None:
    assert coverage([seg(0.0, 10.0), seg(12.0, 20.0)]) == pytest.approx(18.0)


def test_default_output_sits_next_to_the_input() -> None:
    assert default_output(Path("s/rec-part1.transcript.json")) == Path("s/rec-part1.gapfill.transcript.json")
    assert default_output(Path("meeting.json")) == Path("meeting.gapfill.json")


def test_cli_registers_gapfill() -> None:
    from local_transcribe.cli import build_parser

    args = build_parser().parse_args(["gapfill", "t.transcript.json"])
    assert args.func.__name__ == "cmd_gapfill"
    assert args.min_gap == 1.5
    assert args.max_window == 28.0
    assert args.output is None and args.audio is None


# --- slow: real model on the GPU ----------------------------------------------


@pytest.mark.slow
def test_gapfill_recovers_speech_the_transcript_lost(speech_wav: Path) -> None:
    from local_transcribe.gapfill import gapfill

    empty = {"audio": str(speech_wav), "language": "en", "duration": 6.0, "diarized": False, "segments": []}
    out = gapfill(empty)
    text = " ".join(s["text"] for s in out["segments"]).lower()
    assert "quarterly budget" in text
    assert all(s["filled"] and s["speaker"] is None for s in out["segments"])
    assert out["gapfill"]["windows"] == 1
