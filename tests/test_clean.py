"""Tests for local_transcribe.clean.

The filters only ever drop segments; kept segments come out exactly as they
went in. The cases mirror real ones from an audit of 42 room recordings.
"""

from __future__ import annotations

from pathlib import Path

from local_transcribe.clean import clean, default_output


def seg(start: float, end: float, text: str, speaker: str | None = "SPEAKER_00", filled: bool = False) -> dict:
    s = {"start": start, "end": end, "text": text, "speaker": speaker}
    if filled:
        s["filled"] = True
        s["speaker"] = None
    return s


def transcript(segments: list[dict]) -> dict:
    return {"audio": "a.wav", "language": "en", "duration": 7200.0, "diarized": True, "segments": segments}


def texts(t: dict) -> list[str]:
    return [s["text"] for s in t["segments"]]


def test_a_long_loop_collapses_to_its_first_segment() -> None:
    """Mirrors a real case: the silero pass wrote one sentence 19 times in a row."""
    loop = [seg(5790.0 + i, 5791.0 + i, "You know, it's a long-term trend.") for i in range(19)]
    out = clean(transcript([seg(5780.0, 5789.0, "So the order of steps matters.")] + loop + [seg(5810.0, 5815.0, "Next point.")]))
    assert texts(out) == ["So the order of steps matters.", "You know, it's a long-term trend.", "Next point."]
    assert out["segments"][1] == loop[0]
    assert out["clean"]["loops"] == 18


def test_three_in_a_row_is_not_a_loop() -> None:
    """People do say "Okay. Okay. Okay." — under four repeats stays."""
    segs = [seg(0.0, 1.0, "Okay."), seg(1.0, 2.0, "Okay."), seg(2.0, 3.0, "Okay."), seg(3.0, 9.0, "Let's start.")]
    assert texts(clean(transcript(segs))) == ["Okay.", "Okay.", "Okay.", "Let's start."]


def test_a_loop_counts_filled_and_unfilled_alike() -> None:
    segs = [seg(i, i + 0.5, "Yes.", filled=True) for i in range(15)]
    assert texts(clean(transcript(segs))) == ["Yes."]


def test_punctuation_only_segments_are_dropped() -> None:
    segs = [seg(0.0, 1.0, "."), seg(1.0, 2.0, "..."), seg(2.0, 3.0, ""), seg(3.0, 4.0, " ? "), seg(4.0, 5.0, "Okay.")]
    out = clean(transcript(segs))
    assert texts(out) == ["Okay."]
    assert out["clean"]["empty"] == 4


def test_stock_phrases_go_only_when_filled_and_alone() -> None:
    segs = [
        seg(0.0, 2.0, "Thank you for watching.", filled=True),
        seg(3.0, 4.0, "We'll be right back.", filled=True),
        seg(5.0, 6.0, "Thank you, guys.", filled=True),
        seg(7.0, 8.0, "God bless you.", filled=True),
        seg(9.0, 10.0, "I'll see you next time.", filled=True),
        seg(11.0, 12.0, "Thank you so much.", filled=True),
        seg(13.0, 14.0, "Thank you for listening.", filled=True),
        seg(15.0, 16.0, "Bye-bye.", filled=True),
        seg(17.0, 18.0, "You're welcome.", filled=True),
        # someone finishing a talk: real, from the diarized pass
        seg(20.0, 21.0, "Thank you."),
        # the same words inside a real sentence, even when filled
        seg(30.0, 33.0, "Thank you for the question, that is the core of it.", filled=True),
    ]
    out = clean(transcript(segs))
    assert texts(out) == ["Thank you.", "Thank you for the question, that is the core of it."]
    assert out["clean"]["stock"] == 9


def test_a_filled_segment_repeating_its_neighbour_is_dropped() -> None:
    """Mirrors a real case: the gap window re-read the end of the line before it."""
    segs = [
        seg(976.0, 980.5, "So the rule of thumb is that most of the effort goes into the last part of the work."),
        seg(981.0, 984.0, "that most of the effort goes into the last part of the work.", filled=True),
        seg(988.0, 992.0, "So a good plan leaves room for that last part."),
    ]
    out = clean(transcript(segs))
    assert texts(out) == [segs[0]["text"], segs[2]["text"]]
    assert out["clean"]["duplicates"] == 1


def test_filled_speech_with_new_words_stays() -> None:
    segs = [
        seg(100.0, 110.0, "Why is it a problem? That is the question I ask every team."),
        seg(112.0, 118.0, "And in my opinion this is the most important question of the whole review.", filled=True),
    ]
    assert len(clean(transcript(segs))["segments"]) == 2


def test_a_repeat_far_away_is_not_an_edge_duplicate() -> None:
    """A speaker saying the same sentence again a minute later is speech, not a window artefact."""
    segs = [
        seg(100.0, 104.0, "Why is it a problem for this group of users?"),
        seg(180.0, 184.0, "Why is it a problem for this group of users?", filled=True),
    ]
    assert len(clean(transcript(segs))["segments"]) == 2


def test_short_filled_segments_are_not_judged_as_duplicates() -> None:
    """Under four words there are too few trigrams to tell a copy from a coincidence."""
    segs = [seg(0.0, 3.0, "and see you next week"), seg(3.5, 4.5, "next week", filled=True)]
    assert len(clean(transcript(segs))["segments"]) == 2


def test_unfilled_segments_are_never_judged_as_duplicates() -> None:
    segs = [seg(0.0, 3.0, "the issue is still open in the second report"), seg(3.0, 6.0, "the issue is still open in the second report")]
    assert len(clean(transcript(segs))["segments"]) == 2


def test_kept_segments_are_unchanged_and_the_input_is_left_alone() -> None:
    segs = [seg(0.0, 2.0, "  Hello there. "), seg(2.0, 2.5, ".", filled=True), seg(3.0, 5.0, "Today: the budget.", filled=True)]
    t = transcript(segs)
    out = clean(t)
    assert out["segments"] == [segs[0], segs[2]]
    assert len(t["segments"]) == 3 and "clean" not in t
    assert out["duration"] == 7200.0 and out["diarized"] is True


def test_clean_reports_every_rule_even_when_nothing_was_dropped() -> None:
    out = clean(transcript([seg(0.0, 5.0, "Nothing to clean here.")]))
    assert out["clean"] == {"empty": 0, "loops": 0, "stock": 0, "duplicates": 0}


def test_default_output_sits_next_to_the_input() -> None:
    assert default_output(Path("s/x.silero.gapfill.transcript.json")) == Path("s/x.silero.gapfill.clean.transcript.json")
    assert default_output(Path("meeting.json")) == Path("meeting.clean.json")


def test_cli_registers_clean() -> None:
    from local_transcribe.cli import build_parser

    args = build_parser().parse_args(["clean", "t.transcript.json"])
    assert args.func.__name__ == "cmd_clean"
    assert args.output is None
