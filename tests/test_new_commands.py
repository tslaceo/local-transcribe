"""Tests for the parts around the model in probe, mixed, render, compare, fixes and the command line.

Nothing here loads a model: the arithmetic and the shapes are tested, the model calls are not.
All the text in these tests is invented.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_transcribe.clean import clean
from local_transcribe.compare import diff_counts, differences, report, tokens
from local_transcribe.fixes import apply_fixes
from local_transcribe.mixed import apply_candidates, default_candidates_path, group_windows, read_candidates, write_candidates
from local_transcribe.mixed import default_output as mixed_output
from local_transcribe.probe import format_redecode, format_time, parse_time, summarize_languages
from local_transcribe.render import default_output as render_output
from local_transcribe.render import parse_names, speaker_stats, to_text, turns, unlabelled_share


def seg(start: float, end: float, text: str = "x", speaker: str | None = "SPEAKER_00", filled: bool = False) -> dict:
    s = {"start": start, "end": end, "text": text, "speaker": speaker}
    if filled:
        s["filled"] = True
        s["speaker"] = None
    return s


def candidate(first: int, last: int, start: float, end: float, p: float, other_text: str) -> dict:
    return {"first": first, "last": last, "start": start, "end": end, "other": "en", "p_other": p, "p_main": 0.0, "text": "…", "other_text": other_text}


# --- the package itself -------------------------------------------------------


def test_pyannote_usage_reporting_is_off_unless_asked_for() -> None:
    """pyannote-audio 4.x phones home by default; importing this package says no first."""
    import local_transcribe  # noqa: F401

    assert os.environ["PYANNOTE_METRICS_ENABLED"] in ("false", "true")  # "true" only if the user exported it
    if os.environ["PYANNOTE_METRICS_ENABLED"] == "false":
        from pyannote.audio.telemetry.metrics import is_metrics_enabled

        assert not is_metrics_enabled()


def test_hf_token_is_found_next_to_pyproject_when_run_from_elsewhere(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import local_transcribe.transcribe as transcribe_mod

    clone, elsewhere = tmp_path / "clone", tmp_path / "recordings"
    clone.mkdir()
    elsewhere.mkdir()
    (clone / ".env").write_text("HF_TOKEN=from_the_clone\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(transcribe_mod, "PROJECT_ROOT", clone)
    monkeypatch.chdir(elsewhere)
    assert transcribe_mod.load_hf_token() == "from_the_clone"
    (elsewhere / ".env").write_text("HF_TOKEN=from_here\n", encoding="utf-8")
    assert transcribe_mod.load_hf_token() == "from_here"  # the current directory wins


# --- probe ------------------------------------------------------------------


def test_parse_time_takes_seconds_minutes_and_hours() -> None:
    assert parse_time("5025") == 5025.0
    assert parse_time("83:45") == 5025.0
    assert parse_time("1:23:45") == 5025.0
    assert parse_time("0:53:50") == 3230.0
    for bad in ["", "abc", "1:2:3:4", "1:xx", "-5"]:
        with pytest.raises(ValueError):
            parse_time(bad)


def test_format_time_is_h_mm_ss() -> None:
    assert format_time(0) == "0:00:00"
    assert format_time(5025.9) == "1:23:45"


def test_summarize_languages_ignores_silent_windows() -> None:
    rows = [{"language": "ru"}, {"language": "ru"}, {"language": "en"}, {"language": None}]
    assert summarize_languages(rows) == {"ru": pytest.approx(2 / 3), "en": pytest.approx(1 / 3)}
    assert summarize_languages([{"language": None}]) == {}


def test_format_redecode_marks_the_prompted_reading() -> None:
    results = [
        {
            "at": 3230.0,
            "start": 3225.0,
            "end": 3243.0,
            "readings": [
                {"language": "ru", "prompted": False, "text": "раз"},
                {"language": "ru", "prompted": True, "text": "два"},
                {"language": "en", "prompted": False, "text": "three"},
            ],
        }
    ]
    text = format_redecode(results)
    assert "=== 0:53:50 (window 0:53:45-0:54:03)" in text
    assert "ru        : раз" in text and "ru+prompt : два" in text and "en        : three" in text


# --- mixed ------------------------------------------------------------------


def test_group_windows_splits_on_length_and_on_pauses() -> None:
    segments = [seg(0, 10), seg(10, 20), seg(20, 30), seg(33, 35), seg(35.5, 36)]
    assert group_windows(segments, max_window=28.0, max_pause=2.0) == [(0, 1), (2, 2), (3, 4)]
    assert group_windows([]) == []
    assert group_windows([seg(0, 40)]) == [(0, 0)]


def test_apply_candidates_replaces_only_confident_windows() -> None:
    transcript = {
        "audio": "a.wav",
        "language": "ru",
        "duration": 60.0,
        "segments": [
            seg(0, 5, "сначала по-русски"),
            seg(5, 9, "мы встречаемся в четверг", "SPEAKER_00"),
            seg(9, 12, "и смотрим бюджет", "SPEAKER_01"),
            seg(12, 20, "дальше снова по-русски"),
            seg(20, 25, "тут оба языка сразу"),
        ],
    }
    candidates = [
        candidate(1, 2, 5.0, 12.0, 0.97, " we meet on Thursday and look at the budget "),
        candidate(4, 4, 20.0, 25.0, 0.83, "both at once"),
    ]
    out = apply_candidates(transcript, candidates, min_prob=0.9)
    assert [s["text"] for s in out["segments"]] == [
        "сначала по-русски",
        "we meet on Thursday and look at the budget",
        "дальше снова по-русски",
        "тут оба языка сразу",
    ]
    assert out["segments"][1] == {"start": 5.0, "end": 12.0, "text": out["segments"][1]["text"], "speaker": "SPEAKER_00", "language": "en"}
    assert out["mixed"] == {"min_prob": 0.9, "candidates": 2, "replaced": 1}
    assert len(transcript["segments"]) == 5  # input left alone


def test_apply_candidates_takes_exactly_its_own_segments() -> None:
    """Segments overlap after diarization and gapfill: a neighbour that starts inside the window's time is not part of it."""
    transcript = {"segments": [seg(0, 15, "один"), seg(10, 29, "два"), seg(25, 30, "три"), seg(31, 33, "четыре")]}
    out = apply_candidates(transcript, [candidate(0, 1, 0.0, 29.0, 0.99, "one two")], min_prob=0.9)
    assert [s["text"] for s in out["segments"]] == ["one two", "три", "четыре"]


def test_apply_candidates_applies_several_windows_and_keeps_filled() -> None:
    transcript = {"segments": [seg(0, 5, "а"), seg(5, 9, "б", filled=True), seg(9, 12, "в", filled=True), seg(12, 20, "г"), seg(20, 25, "д")]}
    candidates = [candidate(1, 2, 5.0, 12.0, 0.95, "filled words"), candidate(4, 4, 20.0, 25.0, 0.99, "the end")]
    out = apply_candidates(transcript, candidates, min_prob=0.9)
    assert [s["text"] for s in out["segments"]] == ["а", "filled words", "г", "the end"]
    assert out["segments"][1]["filled"] is True and out["segments"][1]["speaker"] is None
    assert "filled" not in out["segments"][3]


def test_apply_candidates_refuses_a_listing_from_another_transcript() -> None:
    transcript = {"segments": [seg(0, 5, "а"), seg(5, 9, "б")]}
    with pytest.raises(ValueError, match="another"):
        apply_candidates(transcript, [candidate(1, 1, 7.5, 9.0, 0.99, "shifted")], min_prob=0.9)
    with pytest.raises(ValueError, match="another"):
        apply_candidates(transcript, [candidate(1, 5, 5.0, 9.0, 0.99, "too long")], min_prob=0.9)


def test_apply_candidates_skips_an_empty_second_reading() -> None:
    transcript = {"segments": [seg(0, 5, "слова")]}
    out = apply_candidates(transcript, [candidate(0, 0, 0.0, 5.0, 0.99, "  ")], min_prob=0.9)
    assert [s["text"] for s in out["segments"]] == ["слова"]
    assert out["mixed"]["replaced"] == 0


def test_the_listing_survives_a_round_trip_and_an_edit(tmp_path: Path) -> None:
    path = tmp_path / "x.mixed.jsonl"
    candidates = [candidate(0, 0, 0.0, 5.0, 0.99, "один"), candidate(1, 1, 5.0, 9.0, 0.5, "two")]
    write_candidates(candidates, path)
    assert read_candidates(path) == candidates
    path.write_text(path.read_text(encoding="utf-8").splitlines()[1] + "\n\n", encoding="utf-8")  # first line deleted by hand
    assert read_candidates(path) == candidates[1:]


def test_mixed_outputs_sit_next_to_the_input() -> None:
    assert default_candidates_path(Path("s/x.transcript.json")) == Path("s/x.mixed.jsonl")
    assert mixed_output(Path("s/x.transcript.json")) == Path("s/x.mixed.transcript.json")
    assert mixed_output(Path("meeting.json")) == Path("meeting.mixed.json")


# --- render -----------------------------------------------------------------


def test_turns_join_one_speaker_and_label_filled_and_unknown() -> None:
    transcript = {
        "segments": [
            seg(0, 2, "Hello."),
            seg(2, 4, "Shall we start?"),
            seg(4, 5, "Yes.", "SPEAKER_01"),
            seg(5, 6, "one moment", filled=True),
            seg(6, 7, "who is this", None),
            seg(7, 8, "  "),
        ]
    }
    assert turns(transcript) == [
        {"start": 0.0, "speaker": "SPEAKER_00", "text": "Hello. Shall we start?"},
        {"start": 4.0, "speaker": "SPEAKER_01", "text": "Yes."},
        {"start": 5.0, "speaker": "FILLED", "text": "one moment"},
        {"start": 6.0, "speaker": "UNKNOWN", "text": "who is this"},
    ]


def test_a_long_turn_is_cut_into_paragraphs_with_their_own_timestamps() -> None:
    transcript = {"segments": [seg(0, 20, "a"), seg(20, 50, "b"), seg(50, 70, "c"), seg(70, 100, "d"), seg(100, 130, "e")]}
    assert [(t["start"], t["text"]) for t in turns(transcript, max_seconds=60)] == [(0.0, "a b c"), (70.0, "d e")]
    assert [(t["start"], t["text"]) for t in turns(transcript)] == [(0.0, "a b c d e")]


def test_to_text_uses_names_and_timestamps() -> None:
    transcript = {"segments": [seg(3725, 3730, "Hello."), seg(3730, 3731, "Yes.", "SPEAKER_01")]}
    text = to_text(transcript, {"SPEAKER_00": "Alice"})
    assert text == "[1:02:05] Alice: Hello.\n\n[1:02:10] SPEAKER_01: Yes.\n"


def test_speaker_stats_and_the_share_without_a_speaker() -> None:
    transcript = {"segments": [seg(0, 120), seg(120, 150, speaker="SPEAKER_01"), seg(150, 180, filled=True), seg(180, 200, speaker=None)]}
    assert speaker_stats(transcript) == {
        "SPEAKER_00": {"segments": 1, "minutes": 2.0},
        "SPEAKER_01": {"segments": 1, "minutes": 0.5},
        "FILLED": {"segments": 1, "minutes": 0.5},
        "UNKNOWN": {"segments": 1, "minutes": 0.3},
    }
    assert unlabelled_share(transcript) == pytest.approx(50 / 200)
    assert unlabelled_share({"segments": []}) == 0.0


def test_parse_names_and_render_output() -> None:
    assert parse_names(["SPEAKER_00=Alice", "SPEAKER_01=Bob B."]) == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob B."}
    assert parse_names(None) == {}
    with pytest.raises(ValueError):
        parse_names(["Alice"])
    assert render_output(Path("s/x.mixed.transcript.json")) == Path("s/x.mixed.txt")
    assert render_output(Path("meeting.json")) == Path("meeting.txt")


# --- compare ----------------------------------------------------------------


def test_tokens_keep_numbers_whole_and_fold_case() -> None:
    shown, norm, when = tokens([seg(7, 9, "It's 75 euro, или 1,5 Ёлки.")])
    assert shown == ["It's", "75", "euro", "или", "1,5", "Ёлки"]
    assert norm == ["it's", "75", "euro", "или", "1,5", "елки"]
    assert when == [7.0] * 6


def test_diff_counts_adds_up() -> None:
    a = "we meet on thursday at eight in the office".split()
    b = "we meet thursday at nine in the big office".split()
    counts = diff_counts(a, b)
    assert counts == {"words_a": 9, "words_b": 9, "same": 7, "substituted": 1, "only_a": 1, "only_b": 1, "disagreement": round(3 / 9, 4)}
    assert diff_counts([], [])["disagreement"] == 0.0


def test_differences_hide_nothing_however_small() -> None:
    a = [seg(10, 20, "the contract costs 15000 euro per day for the meals")]
    b = [seg(10, 20, "the contrast costs 15005 euro per days for the meals")]
    found = differences(a, b)
    assert [(d["a"], d["b"]) for d in found] == [("contract", "contrast"), ("15000", "15005"), ("day", "days")]
    assert found[1]["time"] == 10.0 and found[1]["before"].endswith("costs") and found[1]["after"].startswith("euro")


def test_report_lists_number_differences_first() -> None:
    a = {"segments": [seg(0, 5, "we have 20 days and a cat")]}
    b = {"segments": [seg(0, 5, "we have 28 days and a dog")]}
    text = report(a, b)
    assert "differences involving a number: 1" in text
    assert "all differences: 2" in text
    assert "⟦20 ⇒ 28⟧" in text and "⟦cat ⇒ dog⟧" in text


# --- fixes ------------------------------------------------------------------


def test_fixes_apply_in_order_and_report_every_miss() -> None:
    text = "the meating is on thursday. the meating room is free."
    fixed, misses = apply_fixes(text, [{"old": "meating", "new": "meeting", "count": 2}, {"old": "meeting room", "new": "big room"}])
    assert fixed == "the meeting is on thursday. the big room is free." and misses == []
    fixed, misses = apply_fixes(text, [{"old": "meating", "new": "meeting"}, {"old": "friday", "new": "monday"}, {"old": "", "new": "x"}])
    assert fixed == text  # a miss changes nothing for that fix
    assert misses == ["fix 1: 'meating' occurs 2 times, expected 1", "fix 2: 'friday' occurs 0 times, expected 1", "fix 3: empty 'old'"]


# --- clean: the Russian stock phrases ----------------------------------------


def test_russian_subtitle_phrases_are_stock_when_filled() -> None:
    segments = [
        seg(0, 2, "Субтитры сделал DimaTorzok", filled=True),
        seg(3, 4, "Продолжение следует...", filled=True),
        seg(5, 6, "Субтитры создавал DimaTorzok", filled=True),
        seg(7, 8, "Спасибо за просмотр!", filled=True),
        seg(9, 12, "Продолжение следует в четверг, я пришлю приглашение.", filled=True),
        seg(13, 14, "Продолжение следует."),
    ]
    out = clean({"segments": segments})
    assert [s["text"] for s in out["segments"]] == ["Продолжение следует в четверг, я пришлю приглашение.", "Продолжение следует."]
    assert out["clean"]["stock"] == 4


# --- command line -------------------------------------------------------------


def test_every_command_is_registered() -> None:
    from local_transcribe.cli import build_parser

    parser = build_parser()
    cases = {
        "cmd_transcribe": ["transcribe", "a.wav"],
        "cmd_gapfill": ["gapfill", "t.json"],
        "cmd_clean": ["clean", "t.json"],
        "cmd_langs": ["langs", "a.wav"],
        "cmd_mixed": ["mixed", "t.json", "--other", "en"],
        "cmd_redecode": ["redecode", "a.wav", "--at", "0:53:50", "--at", "90"],
        "cmd_render": ["render", "t.json"],
        "cmd_fix": ["fix", "t.txt", "fixes.json"],
        "cmd_compare": ["compare", "a.json", "b.json"],
    }
    for name, argv in cases.items():
        assert parser.parse_args(argv).func.__name__ == name
    assert parser.parse_args(cases["cmd_mixed"]).apply is None
    assert parser.parse_args(cases["cmd_redecode"]).at == ["0:53:50", "90"]
    assert parser.parse_args(cases["cmd_render"]).max_paragraph == 60.0
    with pytest.raises(SystemExit):
        parser.parse_args(["mixed", "t.json"])  # the second language is not guessed


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict]]:
    """transcribe and gapfill without a model: record what they were called with."""
    import local_transcribe.gapfill as gapfill_mod
    import local_transcribe.transcribe as transcribe_mod

    transcript = {"audio": "a.wav", "language": "en", "duration": 60.0, "diarized": False, "segments": [seg(0.0, 10.0, "kept", None)]}
    calls: dict[str, list[dict]] = {"transcribe": [], "gapfill": []}

    def fake_transcribe(audio, **kw) -> dict:
        calls["transcribe"].append(kw)
        return dict(transcript)

    def fake_gapfill(t: dict, **kw) -> dict:
        calls["gapfill"].append(kw)
        filled = {"start": 20.0, "end": 25.0, "text": "found", "speaker": None, "filled": True}
        return {**t, "segments": t["segments"] + [filled], "gapfill": {"model": "large-v3", "min_gap": 1.5, "windows": 2, "added": 1}}

    monkeypatch.setattr(transcribe_mod, "transcribe", fake_transcribe)
    monkeypatch.setattr(gapfill_mod, "gapfill", fake_gapfill)
    return calls


def test_transcribe_fills_the_gaps_by_default(tmp_path: Path, fake_models: dict, capsys: pytest.CaptureFixture[str]) -> None:
    from local_transcribe.cli import main

    out = tmp_path / "t.json"
    assert main(["transcribe", "a.wav", "--no-diarize", "-o", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["gapfill"]["added"] == 1 and len(fake_models["gapfill"]) == 1
    assert "speech covered 0.2 -> 0.2 of 1.0 min" in capsys.readouterr().out
    assert fake_models["transcribe"][0]["num_speakers"] is None


def test_transcribe_no_gapfill_skips_it_and_passes_the_speaker_bounds(tmp_path: Path, fake_models: dict) -> None:
    from local_transcribe.cli import main

    out = tmp_path / "t.json"
    assert main(["transcribe", "a.wav", "--no-gapfill", "--speakers", "2", "--max-speakers", "3", "-o", str(out)]) == 0
    assert "gapfill" not in json.loads(out.read_text(encoding="utf-8")) and fake_models["gapfill"] == []
    call = fake_models["transcribe"][0]
    assert (call["num_speakers"], call["min_speakers"], call["max_speakers"]) == (2, None, 3)


def test_mixed_reuse_applies_the_listing_on_disk_without_a_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from local_transcribe.cli import main

    transcript = tmp_path / "x.transcript.json"
    transcript.write_text(json.dumps({"language": "ru", "segments": [seg(0, 5, "раз"), seg(5, 9, "два")]}), encoding="utf-8")
    write_candidates([candidate(1, 1, 5.0, 9.0, 0.6, "two, as edited by hand")], tmp_path / "x.mixed.jsonl")
    assert main(["mixed", str(transcript), "--other", "en", "--reuse"]) == 2  # nothing to do without --apply
    assert main(["mixed", str(transcript), "--other", "en", "--reuse", "--apply", "0"]) == 0
    out = json.loads((tmp_path / "x.mixed.transcript.json").read_text(encoding="utf-8"))
    assert [s["text"] for s in out["segments"]] == ["раз", "two, as edited by hand"]
    assert "1 windows replaced" in capsys.readouterr().out


def test_render_fix_and_compare_commands_write_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from local_transcribe.cli import main

    a, b = tmp_path / "a.transcript.json", tmp_path / "b.transcript.json"
    a.write_text(json.dumps({"segments": [seg(0, 5, "we have 20 days")]}), encoding="utf-8")
    b.write_text(json.dumps({"segments": [seg(0, 5, "we have 28 days")]}), encoding="utf-8")
    assert main(["render", str(a), "--name", "SPEAKER_00=Alice"]) == 0
    text = tmp_path / "a.txt"
    assert text.read_text(encoding="utf-8") == "[0:00:00] Alice: we have 20 days\n"
    assert "without a speaker: 0%" in capsys.readouterr().out

    fixes = tmp_path / "fixes.json"
    fixes.write_text(json.dumps([{"old": "20 days", "new": "28 days"}, {"old": "nowhere", "new": "x"}]), encoding="utf-8")
    assert main(["fix", str(text), str(fixes)]) == 1
    assert "20 days" in text.read_text(encoding="utf-8")  # one miss: nothing written
    assert "fix 2: 'nowhere' occurs 0 times" in capsys.readouterr().err
    fixes.write_text(json.dumps([{"old": "20 days", "new": "28 days"}]), encoding="utf-8")
    assert main(["fix", str(text), str(fixes)]) == 0
    assert text.read_text(encoding="utf-8") == "[0:00:00] Alice: we have 28 days\n"

    report_path = tmp_path / "report.txt"
    assert main(["compare", str(a), str(b), "-o", str(report_path)]) == 0
    assert "⟦20 ⇒ 28⟧" in report_path.read_text(encoding="utf-8")
    assert "disagreement" in capsys.readouterr().out
