"""Command line: ``local-transcribe <command>``.

    local-transcribe langs AUDIO                      which language is spoken where
    local-transcribe transcribe AUDIO --lang ru       -> transcript.json (speech, words, speakers, gaps filled)
    local-transcribe gapfill TRANSCRIPT               re-transcribe the holes of a transcript made earlier
    local-transcribe clean TRANSCRIPT                 drop loops, stock phrases and copies from one made earlier
    local-transcribe mixed TRANSCRIPT --other en      stretches in a second language, with both readings
    local-transcribe redecode AUDIO --at 0:53:50      a second opinion on a doubtful spot
    local-transcribe render TRANSCRIPT                -> text by speaker turn
    local-transcribe fix TEXT FIXES                   corrections that cannot miss silently
    local-transcribe compare A B                      how much two transcripts disagree, and where

Heavy imports happen inside the commands, so ``--help`` is instant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: Path | str) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _err(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _say(message: str) -> None:
    """Print, whatever the console encoding is (a transcript is rarely ASCII)."""
    try:
        print(message)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((message + "\n").encode("utf-8", errors="replace"))


# --- commands ---------------------------------------------------------------


def cmd_transcribe(args: argparse.Namespace) -> int:
    from local_transcribe.transcribe import load_hf_token, transcribe

    token = None if args.no_diarize else load_hf_token()
    result = transcribe(
        args.audio,
        language=args.lang,
        model_name=args.model,
        batch_size=args.batch_size,
        diarize=not args.no_diarize,
        hf_token=token,
        device=args.device,
        vad=args.vad,
        num_speakers=args.speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    filled = ""
    if not args.no_gapfill:
        from local_transcribe.gapfill import coverage, gapfill

        before = coverage(result["segments"])
        result = gapfill(result, audio=args.audio, model_name=args.model, device=args.device)
        filled = (
            f"; gapfill added {result['gapfill']['added']}, speech covered "
            f"{before / 60:.1f} -> {coverage(result['segments']) / 60:.1f} of {result['duration'] / 60:.1f} min"
        )
    _write_json(result, args.output)
    mode = "with speaker labels" if result["diarized"] else "without speaker labels"
    _say(f"wrote {args.output}: {len(result['segments'])} segments, {result['duration'] / 60:.1f} min, {mode}{filled}")
    return 0


def cmd_gapfill(args: argparse.Namespace) -> int:
    from local_transcribe.gapfill import coverage, default_output, gapfill

    transcript = _read_json(args.transcript)
    before = coverage(transcript["segments"])
    result = gapfill(
        transcript,
        audio=args.audio,
        model_name=args.model,
        min_gap=args.min_gap,
        max_window=args.max_window,
        device=args.device,
    )
    output = args.output or default_output(args.transcript)
    _write_json(result, output)
    _say(
        f"wrote {output}: {result['gapfill']['added']} filled segments over {result['gapfill']['windows']} windows, "
        f"speech covered {before / 60:.1f} -> {coverage(result['segments']) / 60:.1f} of {result['duration'] / 60:.1f} min"
    )
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    from local_transcribe.clean import clean, default_output

    transcript = _read_json(args.transcript)
    result = clean(transcript)
    output = args.output or default_output(args.transcript)
    _write_json(result, output)
    dropped = ", ".join(f"{k} {v}" for k, v in result["clean"].items())
    words = sum(len(s["text"].split()) for s in transcript["segments"])
    words_left = sum(len(s["text"].split()) for s in result["segments"])
    _say(
        f"wrote {output}: {len(transcript['segments'])} -> {len(result['segments'])} segments, "
        f"{words} -> {words_left} words; dropped: {dropped}"
    )
    return 0


def cmd_langs(args: argparse.Namespace) -> int:
    from local_transcribe.probe import format_time, language_profile, summarize_languages

    rows = language_profile(args.audio, step=args.step, model_name=args.model, device=args.device)
    for r in rows:
        scores = " ".join(f"{k}:{v:.2f}" for k, v in r["probabilities"].items()) or "silent"
        _say(f"{format_time(r['start'])}  {scores}")
    shares = summarize_languages(rows)
    _say("windows won: " + (", ".join(f"{k} {v:.0%}" for k, v in shares.items()) or "none (all silent)"))
    return 0


def cmd_mixed(args: argparse.Namespace) -> int:
    from local_transcribe.mixed import (
        apply_candidates,
        default_candidates_path,
        default_output,
        find_candidates,
        read_candidates,
        write_candidates,
    )

    transcript = _read_json(args.transcript)
    listing = Path(args.candidates or default_candidates_path(args.transcript))
    if args.reuse:
        if args.apply is None:
            _err("--reuse applies a listing made earlier: give --apply P as well")
            return 2
        candidates = read_candidates(listing)
        _say(f"read {listing}: {len(candidates)} windows")
    else:
        candidates = find_candidates(
            transcript, other=args.other, threshold=args.threshold, audio=args.audio, model_name=args.model, device=args.device
        )
        write_candidates(candidates, listing)
        _say(f"wrote {listing}: {len(candidates)} windows where {args.other} scored {args.threshold} or more")
    if args.apply is not None:
        result = apply_candidates(transcript, candidates, min_prob=args.apply)
        output = args.output or default_output(args.transcript)
        _write_json(result, output)
        _say(f"wrote {output}: {result['mixed']['replaced']} windows replaced (score {args.apply} or more)")
    return 0


def cmd_redecode(args: argparse.Namespace) -> int:
    from local_transcribe.probe import format_redecode, parse_time, redecode

    results = redecode(
        args.audio,
        [parse_time(t) for t in args.at],
        languages=[lang.strip() for lang in args.langs.split(",") if lang.strip()],
        prompt=args.prompt,
        before=args.before,
        length=args.length,
        model_name=args.model,
        device=args.device,
    )
    text = format_redecode(results)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        _say(f"wrote {args.output}: {len(results)} spots")
    else:
        _say(text.rstrip("\n"))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from local_transcribe.render import default_output, parse_names, speaker_stats, to_text, unlabelled_share

    transcript = _read_json(args.transcript)
    output = args.output or default_output(args.transcript)
    Path(output).write_text(to_text(transcript, parse_names(args.name), args.max_paragraph), encoding="utf-8")
    stats = ", ".join(f"{k} {v['minutes']} min" for k, v in speaker_stats(transcript).items())
    _say(f"wrote {output}; speech by label: {stats}; without a speaker: {unlabelled_share(transcript):.0%}")
    return 0


def cmd_fix(args: argparse.Namespace) -> int:
    from local_transcribe.fixes import apply_fixes

    text = Path(args.text).read_text(encoding="utf-8")
    fixes = json.loads(Path(args.fixes).read_text(encoding="utf-8"))
    fixed, misses = apply_fixes(text, fixes)
    if misses:
        for miss in misses:
            _err(miss)
        _err(f"{len(misses)} of {len(fixes)} fixes did not match as stated: nothing written")
        return 1
    output = args.output or args.text
    Path(output).write_text(fixed, encoding="utf-8")
    _say(f"wrote {output}: {len(fixes)} fixes applied")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from local_transcribe.compare import report

    text = report(_read_json(args.a), _read_json(args.b))
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        _say(f"wrote {args.output}")
        _say("\n".join(text.splitlines()[:4]))
    else:
        _say(text.rstrip("\n"))
    return 0


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from local_transcribe.gapfill import DEFAULT_MAX_WINDOW, DEFAULT_MIN_GAP
    from local_transcribe.mixed import DEFAULT_THRESHOLD
    from local_transcribe.probe import DEFAULT_BEFORE, DEFAULT_LENGTH, DEFAULT_STEP
    from local_transcribe.render import DEFAULT_MAX_PARAGRAPH
    from local_transcribe.transcribe import DEFAULT_BATCH_SIZE, DEFAULT_MODEL, DEFAULT_VAD, VAD_METHODS

    parser = argparse.ArgumentParser(
        prog="local-transcribe", description="Local speech-to-text with speaker labels for long, messy recordings."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    device_help = "cuda or cpu (default: cuda if an NVIDIA GPU is visible, else cpu)"

    p = sub.add_parser("transcribe", help="transcribe audio with WhisperX (word-aligned, diarized if HF_TOKEN is set), then fill the gaps")
    p.add_argument("audio")
    p.add_argument("-o", "--output", default="transcript.json")
    p.add_argument("--lang", default="en", help="the language that carries the recording (default: en); see `langs`")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--no-diarize", action="store_true", help="skip speaker diarization")
    p.add_argument("--speakers", type=int, default=None, help="exact number of speakers, if known (default: found by the diarizer)")
    p.add_argument("--min-speakers", type=int, default=None, help="lower bound on the number of speakers")
    p.add_argument("--max-speakers", type=int, default=None, help="upper bound on the number of speakers")
    p.add_argument("--device", default=None, help=device_help)
    p.add_argument("--vad", choices=VAD_METHODS, default=DEFAULT_VAD, help=f"voice activity detector (default: {DEFAULT_VAD})")
    p.add_argument("--no-gapfill", action="store_true", help="skip re-transcribing the gaps the detector left (on by default)")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("gapfill", help="re-transcribe the stretches the batched transcription dropped, window by window")
    p.add_argument("transcript")
    p.add_argument("-o", "--output", default=None, help="default: X.gapfill.transcript.json next to the input")
    p.add_argument("--audio", default=None, help="the audio, if it moved since transcribe (default: the path in the transcript)")
    p.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP, help="seconds without a segment that count as a hole")
    p.add_argument("--max-window", type=float, default=DEFAULT_MAX_WINDOW, help="longest stretch transcribed at once")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None, help=device_help)
    p.set_defaults(func=cmd_gapfill)

    p = sub.add_parser("clean", help="drop loops, empty segments, stock phrases and copies of a neighbour; never rewrites text")
    p.add_argument("transcript")
    p.add_argument("-o", "--output", default=None, help="default: X.clean.transcript.json next to the input")
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("langs", help="the language Whisper hears in a 30 s window every STEP seconds")
    p.add_argument("audio")
    p.add_argument("--step", type=float, default=DEFAULT_STEP, help=f"seconds between windows (default: {DEFAULT_STEP:.0f})")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None, help=device_help)
    p.set_defaults(func=cmd_langs)

    p = sub.add_parser("mixed", help="find stretches spoken in a second language and transcribe them in it")
    p.add_argument("transcript")
    p.add_argument("--other", required=True, help="the second language, e.g. en")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"score from which a window is listed (default: {DEFAULT_THRESHOLD})")
    p.add_argument("--apply", type=float, default=None, metavar="P", help="also write a transcript with windows scoring P or more replaced (suggested: 0.9)")
    p.add_argument("--reuse", action="store_true", help="with --apply: use the listing already on disk (edited or not) instead of looking again")
    p.add_argument("--candidates", default=None, help="the listing; default: X.mixed.jsonl next to the input")
    p.add_argument("-o", "--output", default=None, help="with --apply; default: X.mixed.transcript.json next to the input")
    p.add_argument("--audio", default=None, help="the audio, if it moved since transcribe")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None, help=device_help)
    p.set_defaults(func=cmd_mixed)

    p = sub.add_parser("redecode", help="transcribe a short window around a timestamp again, per language and with a prompt")
    p.add_argument("audio")
    p.add_argument("--at", action="append", required=True, metavar="TIME", help="seconds, m:ss or h:mm:ss; repeat for more spots")
    p.add_argument("--langs", default="en", help="comma-separated languages to read the window in (default: en)")
    p.add_argument("--prompt", default=None, help="vocabulary to bias one extra reading in the first language")
    p.add_argument("--before", type=float, default=DEFAULT_BEFORE, help=f"seconds ahead of the spot the window starts (default: {DEFAULT_BEFORE:.0f})")
    p.add_argument("--length", type=float, default=DEFAULT_LENGTH, help=f"window length in seconds (default: {DEFAULT_LENGTH:.0f})")
    p.add_argument("-o", "--output", default=None, help="write the readings to a file instead of printing them")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None, help=device_help)
    p.set_defaults(func=cmd_redecode)

    p = sub.add_parser("render", help="transcript.json -> text with a timestamp on every paragraph")
    p.add_argument("transcript")
    p.add_argument("-o", "--output", default=None, help="default: X.txt next to the input")
    p.add_argument("--name", action="append", metavar="LABEL=Name", help="replace a speaker label, e.g. SPEAKER_00=Alice; repeatable")
    p.add_argument(
        "--max-paragraph",
        type=float,
        default=DEFAULT_MAX_PARAGRAPH,
        metavar="SECONDS",
        help=f"start a new paragraph this long into a speaker's turn (default: {DEFAULT_MAX_PARAGRAPH:.0f}; 0: one paragraph per turn)",
    )
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("fix", help="apply corrections to a text; every correction states how often it must match")
    p.add_argument("text")
    p.add_argument("fixes", help='JSON list of {"old": ..., "new": ..., "count": 1}')
    p.add_argument("-o", "--output", default=None, help="default: the text file itself is rewritten")
    p.set_defaults(func=cmd_fix)

    p = sub.add_parser("compare", help="how much two transcripts of the same audio disagree, and where")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("-o", "--output", default=None, help="write the report to a file instead of printing it")
    p.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
