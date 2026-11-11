---
name: local-transcribe
description: Use when a long recording, audio or video (meeting, lecture, interview, call, live or pre-recorded), has to be transcribed on the local machine with speaker labels, when a Whisper or WhisperX transcript has holes, repeated lines, invented phrases or text in the wrong language, when the speech mixes two languages, or when a transcription run on a machine without an NVIDIA GPU looks hung or far too slow.
---

# Local transcription of long, messy recordings

## Overview

This folder is working code plus what running it taught. Run the code as it is first, and
change it only against a measurement.

**A transcript is evidence only where it was checked. The tool's job is to show where to check.**

Run commands from any folder as `uv run --project <this folder> local-transcribe <command>`.
`--help` on a command lists its options.

## Setup, once

- `uv sync` in this folder; `ffmpeg` on PATH.
- Speaker labels need a HuggingFace token: `HF_TOKEN` in a `.env` in this folder (or in the
  folder you run from), from an account that accepted the three gated models `.env.example`
  names. A person has to do that part. No token is a mode, not an error: the run goes on
  without labels.
- "GPU" here means NVIDIA (CUDA). Anywhere else, Apple Silicon included, the run uses the
  CPU with int8 weights. The first log line says which: `device: cpu (int8)`.
- The first run downloads the models: 2.9 GB for large-v3, an alignment model per language
  (2.4 GB for Russian, 0.4 GB for English), under 0.1 GB for the speaker models. After that
  `HF_HUB_OFFLINE=1` keeps the libraries off the network. The audio and the text never leave
  the machine; pyannote's usage reporting is switched off by this package.

## The run

1. `ffmpeg -i IN -vn -ac 1 -ar 16000 -c:a pcm_s16le X.wav` (IN is audio or video).
2. `langs X.wav`: `--lang` is the language that wins most windows.
3. A slice first: `ffmpeg -ss 600 -t 300 -i X.wav slice.wav`, then `transcribe slice.wav`.
   It times the machine and proves the token. A run saves nothing until it ends, and
   speakers come after recognition, so a bad token costs the whole run.
4. `transcribe X.wav --lang ru -o X.transcript.json` in the background, output to a log
   file, laptop kept awake (`caffeinate -i` on macOS). `--speakers 2` or `--max-speakers 3`
   if you know: it stops one voice being split across labels.
5. The last line of stdout is `wrote X: ... speech covered A -> B of C min`: A before gap
   filling, B after, C the whole recording with its silences. Hand that line over.
6. Two languages: `mixed X.transcript.json --other en` writes `X.mixed.jsonl`. Read it. A
   replacement swaps a whole window (up to 28 s), so delete the lines for windows that hold
   both languages, then `mixed ... --reuse --apply 0.9`. Use `X.mixed.transcript.json` from
   here on.
7. `render` it: a timestamp on every paragraph, minutes per speaker, the share without a
   speaker. Correct that text, not the JSON.
8. Doubtful spots: `redecode X.wav --at 0:53:50 --langs ru,en --prompt "names, terms"`.
9. Corrections: `fix X.txt fixes.json`. Every fix states how many times it must match, so
   none is lost silently and none lands twice.

## Video recordings

A recorded lecture or a screen recording of a call goes through the same run, and usually
needs less repair than a room. Use the picture as the check for names, numbers and terms.
No slide deck? Cut the slides out of the video yourself:
`ffmpeg -i IN -vf fps=1/10 frames/%04d.png` (frame N is about (N-1)*10 s in), then look at
the frames. Two identical neighbours are not proof of one slide: check the speech between
them, because a slide shown briefly between two samples is lost that way.

## How long (one 114-minute recording, large-v3)

| Machine | End to end |
|---|---|
| RTX 4090 | about 5 min |
| i9-13900K (24 cores), no GPU | 68.5 min, about 0.6 of the recording's length |
| Apple Silicon | not measured: scale the slice, minus the model downloads |

An hour on a CPU is not a hang. Recognition prints nothing until it is done; log lines carry
the clock. To tell slow from stuck, watch the process's CPU time grow
(`ps -o etime,time -p PID`), and do not restart on silence alone.

## What goes wrong (each of these happened)

| Symptom | Cause | What to do |
|---|---|---|
| 20-35 s of speech missing mid-sentence, as loud as the rest | the voice detector dropped it | already handled: silero by default, then gapfill. Never pass `--no-gapfill`; report the coverage line |
| "Thank you for watching.", "Субтитры сделал DimaTorzok", "Продолжение следует..." | Whisper reading silence | dropped when it is a whole filled segment; inside a longer segment it survives, so search the rendered text |
| Fluent text in the main language where the speaker used another | Whisper translated it | `mixed` |
| A line that repeats the end of the line before it | a gap window re-read its neighbour | `clean` drops these; it runs inside `transcribe` and `gapfill` |
| `FILLED` and `UNKNOWN` paragraphs; 10-22 % of room speech has no speaker | gap filling has no speakers; the diarizer skips some | attribute from context while correcting; state the share `render` prints |
| Names, product names, numbers wrong, the same in every pass | the model | check against documents or people; never against a second pass |
| Low-confidence text about something unrelated | gapfill caught speech from the next room | ask, then drop |

A second full pass with the same model repeats the same mistakes: agreement between two
passes confirms nothing. `redecode` (other window edges, other language, a prompt) is the
independent opinion, and a prompted reading is a hint, not a verdict.

## What does not help on a CPU (measured, do not retry without a reason)

- float32 instead of int8: slower (1.5x realtime against 2.5x).
- More CTranslate2 threads: 2.1x at 4, 2.5x at 24, and the batched pipeline died twice at 16.
- Beam 1: slower than beam 5.

What did help: silero-vad sets torch to one thread when it is imported, so alignment and
diarization ran on one core. `transcribe` gives the threads back (`restore_torch_threads`).
Any pipeline built around silero needs the same.

## Handing over

Mark, do not smooth: `[?]` for a guess, `[inaudible]`, `[heard: "..."]`. The reader must be
able to tell what was heard from what was inferred. Hand over with the coverage line, the
share without a speaker, what was checked against what, and what was not checked.

## Apple Silicon

Not run there yet. The dependencies resolve for arm64 and the CPU path should work as on the
desktop above. Faster routes are unmeasured ideas, and each needs a code change: `--device`
sets one device for the whole run and CTranslate2 has no `mps`. Candidates: `mps` for
alignment and diarization only; mlx-whisper or whisper.cpp for recognition, keeping gapfill,
clean, mixed and compare, which work on the transcript and on windows of audio. Keep a
change only if a stage gets faster and `compare` shows the text moved no more than the
noise floor (4.7 % of words between a GPU and a CPU pass of the same model).
