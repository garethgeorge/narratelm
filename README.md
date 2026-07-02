# narratelm

Narrate epub books into chapterized `.m4b` audiobooks, locally, with
[VibeVoice](https://github.com/vibevoice-community/VibeVoice) (Microsoft's
long-form TTS model). Designed for Apple Silicon (target: M4, 24 GB unified
memory); also runs on CUDA or (slowly) CPU.

- epub → editable `book.json` → per-chapter `.m4a` parts → single `.m4b`
  with chapter marks and cover art
- voice cloning from any 30–60 s reference clip; four bundled narrators
- resumable, cached, Ctrl-C-safe generation with hallucination detection
  and retry
- consistency-first design: one pinned seed + fixed voice anchor for every
  window (see below)

## Requirements

[Nix](https://nixos.org) with flakes. Everything else (Python 3.12, uv,
ffmpeg) comes from the dev shell; ML packages install into a local `.venv`
from PyPI.

[Git LFS](https://git-lfs.com) is required to check out the repo: the bundled
reference voices in `src/narratelm/data/voices/` are stored as LFS objects.
Without it those `.wav` files clone as small text pointers and voice loading
fails. (`git-lfs` is provided by the Nix dev shell.)

## Setup

```sh
git lfs install && git lfs pull      # fetch the bundled voice .wav files (once per clone)
nix develop                          # python, uv, ffmpeg, just, git-lfs
just setup                          # uv sync — torch + pinned VibeVoice fork (several GB)
narratelm setup                     # environment report (ffmpeg / torch / device)
narratelm setup --download-model    # optional: prefetch the default 1.5B model (~6 GB)
```

The default model is the 1.5B VibeVoice; `--model` switches to the 7B (including
low-VRAM quantized) variants — see [Models](#models).

On a Linux dev box you can skip the ML stack (`just setup-lite`); extract,
combine, and the entire test suite run without torch.

## Workflow

```sh
# 1. extract the epub into an editable JSON next to it
narratelm extract book.epub
#    → prints a section table; flip "skip" flags / fix titles in book.json

# 2. (once per book+voice) audition seeds on a real excerpt — seed choice is
#    the single biggest quality variable
narratelm probe book.json --voice en-val-grimm
#    → listen to book/probe/*.wav, pick the best-sounding seed

# 3. render — one .m4a per chapter, big chapters split into balanced .ptNN parts
narratelm generate book.json --voice en-val-grimm --seed <winner>

# 4. merge into one chapterized audiobook
narratelm combine book.json          # → book.m4b
```

`narratelm run book.epub` chains extract → generate → combine with defaults.

### Flags worth knowing

| flag | what it does |
|---|---|
| `--model 7b-4bit` | pick the TTS model — alias, HF repo id, or local path (see [Models](#models)) |
| `--sections "7,9-12,interlude2"` | render a subset (spine indexes and/or section ids) |
| `--limit-words 400` | smoke test: only the first N words of each selected section |
| `--voice /path/to/ref.wav` | clone a voice from 30–60 s of clean speech (no background music — it measurably increases hallucinations) |
| `--ddpm-steps 10` | ~2× faster than the quality default of 20 (`6` = fast draft, audibly flatter) |
| `--force` | re-render, ignoring caches |
| `narratelm voices list` | bundled narrators (LibriTTS-R, CC BY 4.0): en-ashwath-ganesan, en-mark-welch (m); en-val-grimm (f) |

Generation is resumable: re-running skips finished parts and cached windows;
editing `book.json` re-renders only the audio whose text changed; Ctrl-C is
safe to press at any time.

## Models

`--model` selects the TTS model. It accepts a friendly alias, any Hugging Face
repo id, or a local path (`--model /path/to/model`):

| alias | repo | footprint | notes |
|---|---|---|---|
| `1.5b` *(default)* | `microsoft/VibeVoice-1.5B` | ~6 GB | runs on MPS / CUDA / CPU |
| `7b` | `aoi-ot/VibeVoice-Large` | ~18 GB VRAM | full-precision 7B; needs a large GPU |
| `7b-4bit` | `DevParker/VibeVoice7b-low-vram` `4bit/` | ~6.6 GB VRAM | nf4-quantized 7B — **CUDA only** |
| `7b-8bit` | `DevParker/VibeVoice7b-low-vram` `8bit/` | ~10.6 GB VRAM | int8-quantized 7B — **CUDA only** |

Recommended on an NVIDIA GPU: the `7b-4bit` model — 7B quality that fits in
~6.6 GB of VRAM. Pass `--model` to whichever command you run.

```sh
# end to end (extract → generate → combine) in one command
narratelm setup --download-model --model 7b-4bit     # once: fetch just the 4bit/ variant
narratelm run book.epub --model 7b-4bit --device cuda
#   → book.m4b next to book.epub
```

```sh
# or drive the stages individually, for control between steps
narratelm extract book.epub                          # → book.json; edit skip flags / titles
narratelm probe   book.json --model 7b-4bit --device cuda   # audition seeds by ear
narratelm generate book.json --model 7b-4bit --device cuda --seed <winner>
narratelm combine book.json                          # merge parts → book.m4b
```

The `7b-4bit` / `7b-8bit` variants are pre-quantized
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) weights
and load **only on CUDA** — bitsandbytes has no macOS/MPS kernels. On Apple
Silicon use `1.5b` (or full `7b` if you have the unified memory to spare).
`bitsandbytes` is installed automatically with the ML deps on Linux; the
embedded quantization config is detected and applied at load — nothing extra to
pass.

The DevParker repo ships both variants side by side, so the alias also sets
`--model-subfolder` (`4bit` / `8bit`). To point a raw repo id at a subfolder,
pass `--model-subfolder` yourself (`--model DevParker/VibeVoice7b-low-vram
--model-subfolder 4bit`). Because the weights change the audio, model +
subfolder are part of every window's cache key: switching models re-renders
rather than mixing outputs within a book.

Weights download to the standard Hugging Face cache (`$HF_HOME`, default
`~/.cache/huggingface`); override the location per run with `--cache-dir`, or use
a local path as `--model` for fully offline use.

## How consistency is kept

Noticeable mid-story changes in narration performance are the failure mode
this tool is designed around:

- Text is rendered in **windows** of ~2000 words (one TTS call each, ≈13 min
  of audio). VibeVoice is a long-form model, so windows are kept large for
  natural cross-sentence prosody and few joins; lower `--window-words` if you
  hear consistency drift within a single long generation.
- Every window is conditioned on the **same reference voice** and the
  **same pinned seed** — the community-established recipe for drift-free
  single-speaker output. Retries (after a QA rejection) derive a new seed
  deterministically; nothing else ever varies within a book.
- QA checks every window (duration ratio vs. word count, silence scan,
  overrun) to catch VibeVoice's music/garble hallucinations; failures retry
  up to 3× and are otherwise flagged `suspect` in the manifest for
  spot-listening.
- Window boundaries fall only at sentence ends, preferring paragraph and
  scene breaks; joins are RMS-matched and land inside inserted pauses; every
  part gets a two-pass ffmpeg `loudnorm` (−19 LUFS).
- `--rolling-context` (experimental) additionally conditions each window on
  the last ~15 s of the previous window's audio. No community tool has
  shipped this pattern — A/B it against the default before trusting it with
  a full book.

## Development

```sh
just setup-lite   # venv without torch — everything below works on any box
just test         # 143 tests; ffmpeg-dependent ones auto-skip if ffmpeg is missing
just lint
```

The test suite runs the real pipeline end to end against the bundled test
epub using a deterministic fake TTS engine (`--engine fake` also works on the
CLI for dry runs). The VibeVoice fork is pinned by exact SHA in
`pyproject.toml` / `uv.lock`; all fork-API contact lives in
`src/narratelm/tts/vibevoice.py`.

## Troubleshooting

- **"MPS requested but unavailable"** — you're not on Apple Silicon or torch
  lacks MPS; generation falls back to CPU with a warning (very slow).
- **Model download** goes to the standard Hugging Face cache
  (`~/.cache/huggingface`); `--model` accepts a local path for offline use.
- **NixOS**: manylinux wheels need `libstdc++` on the loader path — the dev
  shell exports `LD_LIBRARY_PATH` for you; run everything via `nix develop`.
- **An audible AI disclaimer** at the start of generated audio has been
  reported on some VibeVoice inference paths. Spot-check your first render
  (`--limit-words 100`); if present, open an issue — the assembly layer has
  a trim hook (`trim_offset`) ready to apply a fixed lead-in cut.
- VibeVoice embeds an **inaudible provenance watermark**; narratelm leaves it
  intact. English (and Chinese) text only. Model weights are MIT with a
  research-use advisory from Microsoft.
