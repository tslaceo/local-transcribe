# local-transcribe

Speech-to-text with speaker labels for long, messy recordings, on your own machine. Built
for phone-on-the-table audio: one or two hours, a room, people who switch language in the
middle of a sentence.

It is WhisperX (large-v3, word alignment, pyannote diarization) with the parts that turned
out to be missing once the transcripts were actually read: filling the holes the voice
detector leaves, dropping the phrases Whisper invents, putting translated stretches back
into the language they were spoken in, and tools to see how far a transcript can be trusted.

**Status: as is.** This was extracted from a private tool and is not maintained. It has run
on Windows with an NVIDIA GPU and on a CPU alone. It has not been run on macOS or Linux; the
dependencies resolve there.

**What leaves the machine.** The audio and the text: never. The libraries download the
models on the first run, and run offline after that (`HF_HUB_OFFLINE=1`). One thing would
leave by default and is switched off here: pyannote-audio 4.x reports usage to its maker
(pipeline name, library version, a random session id, the length of the audio, the number of
speakers asked for). Importing this package sets `PYANNOTE_METRICS_ENABLED=false` unless you
set it yourself.

## Install

```
uv sync
```

Python 3.12 (uv fetches it) and `ffmpeg` on PATH. On Windows and Linux torch comes from the
PyTorch CUDA 12.8 index; the same build runs on a machine without a GPU. On macOS torch comes
from PyPI. "GPU" in this project means NVIDIA: on Apple Silicon the run uses the CPU.

Speaker labels need a HuggingFace token with access to three gated pyannote models: copy
`.env.example` to `.env` and follow the notes in it. `.env` is read from the folder you run
in (or above it) and from this folder. Without a token everything works, only without
speaker labels.

The first run downloads 2.9 GB for large-v3, an alignment model for the language (2.4 GB
for Russian, 0.4 GB for English) and under 0.1 GB of speaker models.

## Use

```
ffmpeg -i recording.m4a -ac 1 -ar 16000 -c:a pcm_s16le rec.wav
uv run local-transcribe langs rec.wav                         # which language carries the recording
ffmpeg -ss 600 -t 300 -i rec.wav slice.wav                    # five minutes first:
uv run local-transcribe transcribe slice.wav --lang ru        #   times the machine, proves the token
uv run local-transcribe transcribe rec.wav --lang ru -o rec.transcript.json
uv run local-transcribe mixed rec.transcript.json --other en  # -> rec.mixed.jsonl: read it, delete what is wrong
uv run local-transcribe mixed rec.transcript.json --other en --reuse --apply 0.9
uv run local-transcribe render rec.mixed.transcript.json      # -> rec.mixed.txt
uv run local-transcribe fix rec.mixed.txt fixes.json          # corrections, each with its expected count
```

From another folder: `uv run --project <this folder> local-transcribe ...`.

| Command | What it does |
|---|---|
| `langs AUDIO` | The language Whisper hears in a 30 s window every two minutes. |
| `transcribe AUDIO` | Speech detection (silero), recognition, word alignment, speakers, then gap filling and junk filtering. Writes `transcript.json`. `--speakers N` or `--max-speakers N` if the number is known. |
| `gapfill TRANSCRIPT` | Re-transcribes the holes of a transcript made earlier. |
| `clean TRANSCRIPT` | Drops loops, empty segments, stock phrases and copies of a neighbour. Never rewrites text. |
| `mixed TRANSCRIPT --other en` | Lists windows where a second language is likely, with both readings. `--apply 0.9` replaces the confident ones; `--reuse` applies the listing on disk, as edited. |
| `redecode AUDIO --at 0:53:50` | Transcribes a short window around a timestamp again, per language and with a vocabulary prompt. |
| `render TRANSCRIPT` | Text with a timestamp on every paragraph (speaker change, or 60 s into a turn), minutes per speaker and the share without one. `--name SPEAKER_00=Alice` renames labels. |
| `fix TEXT FIXES` | Applies a JSON list of `{"old", "new", "count"}` to a text. A fix that does not match exactly `count` times stops everything: nothing is written. |
| `compare A B` | How much two transcripts of the same audio disagree, and every place they do. |

`transcript.json` holds `segments` with `start`, `end`, `text`, `speaker`; segments added by
gap filling carry `filled: true` and no speaker. The last line `transcribe` prints is
`wrote X: N segments, C min, with speaker labels; gapfill added K, speech covered A -> B of
C min`: A before gap filling, B after, C the whole recording, silences included.

## Why each part exists

**silero instead of WhisperX's default detector.** On room recordings the default (pyannote)
dropped whole 20-35 s stretches mid-sentence, at full loudness: in one case 39 of 55 minutes
were kept. Silero against pyannote, words in the transcript: +28 % and +27 % on two room
recordings, +4 % on a clean one, +0 % on a screen recording of a call. `--vad pyannote`
restores the old behaviour.

**gapfill.** Silero still loses quiet and distant voices: room recordings stayed at 50-60 %
coverage. Every stretch of at least 1.5 s with no segment is cut into windows of at most
28 s and transcribed alone, with no detector. One recording went from 51 % to 80 %. About a
minute per 100 windows on an RTX 4090.

**clean.** Both passes leave junk. Four rules, found on 42 recordings: a segment with no
word in it; four or more identical segments in a row (one sentence was written 19 times
running); a filled segment that is nothing but a stock phrase ("Thank you for watching."
was 73 of them, and in Russian the model says "Субтитры сделал DimaTorzok" and "Продолжение
следует..."); a filled segment whose words mostly repeat its neighbour, because the gap
window re-read the edge. Segments are dropped, never rewritten.

**mixed.** Whisper decodes a run in one language. A whole sentence in another language is
translated, and the transcript reads fluently and is wrong. On the recording this was
written for, a threshold of 0.15 flagged 11 windows; the four above 0.95 were real
translations, one at 0.83 held both languages at once, the rest were fine as they were. A
replacement swaps a whole window, up to 28 s, so the listing is meant to be read and edited
before it is applied.

**fix.** A long transcript takes hundreds of corrections over several passes. A correction
whose target no longer matches does nothing and says nothing; one that matches twice changes
a place nobody looked at. Stating the count turns both into errors.

**redecode and compare.** A second full pass with the same model repeats most of its own
mistakes. Two passes of one 114-minute recording, on a GPU (float16) and on a CPU (int8),
disagreed in 4.7 % of the words, in both directions and mostly where both were wrong. Of
163 numbers, eight places differed and none had two different values. Two runs with the
very same settings on the same CPU still differed in 1.3 % of the words of a 4-minute
slice. So: a few percent is
the noise floor, and agreement between passes confirms nothing in a hard spot. A short
window cut at other places, another language, or a vocabulary prompt is the independent
reading.

## How long it takes

One 114-minute recording, large-v3, speakers on:

| Machine | End to end | By stage |
|---|---|---|
| RTX 4090 | about 5 min | detection, recognition, alignment 172 s; diarization and gapfill 91 s |
| i9-13900K, no GPU | 68.5 min | recognition 36, alignment 6, diarization 20, gapfill 6.5 |

On a CPU that is about 0.6 of the recording's length.

One fix made the CPU figure possible. silero-vad calls `torch.set_num_threads(1)` when it is
imported, for the whole process, so alignment and diarization ran on one core: 198 s and
600 s for a 10-minute slice, 41 s and 114 s once torch has its 24 threads back. `transcribe`
restores them after recognition, on the CPU only.

What did not help on the CPU: float32 (1.5x realtime against 2.5x for int8); more
CTranslate2 threads (2.1x at 4, 2.5x at 24, and the batched pipeline died twice at 16 with
no traceback); beam 1 (slower than beam 5).

Not measured: whether the GPU path is the most a card can give. Batch size 16 was never
tuned, and the one-thread problem may slow the CPU side of a GPU run as well.

## Limits

- Names, product names and numbers are unreliable, and they are wrong the same way in every
  pass. Check them against documents or people.
- 10-22 % of the speech in a room recording gets no speaker label; filled segments never do.
- Stock phrases inside a longer segment survive the filter.
- Gap filling can pick up speech from the next room.
- `transcribe` writes nothing until the whole run ends. A failure in the speaker stage, for
  example a token without access, loses the recognition before it. Run a slice first.
- `mixed --apply` replaces whole windows. A window with both languages in it has to be
  merged by hand.
- One device for the whole run: there is no way yet to put only some stages on a GPU.
- The synthesised-speech fixture for the slow tests has a macOS branch that was never run.

## As an agent skill

`SKILL.md` is written for a coding agent: the order of the run, what goes wrong and what was
already tried. Clone the repository into the agent's skills folder (for Claude Code,
`~/.claude/skills/local-transcribe`) and it is picked up as a skill with the code next to it.

## Tests

```
uv run pytest                 # fast: no model
uv run pytest --run-slow      # also runs the real models on a few seconds of synthesised speech
```
