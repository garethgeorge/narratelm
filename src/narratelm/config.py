"""Defaults and constants shared across the pipeline.

Consistency-critical generation settings (voice, seed, cfg_scale, ddpm_steps)
are locked per book run and recorded in the manifest — mixing them within one
book is an audible bug, so everything that affects output audio flows through
GenerationSettings and into cache keys.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

SAMPLE_RATE = 24_000  # VibeVoice output rate (mono)

# Words-per-minute used to estimate audio duration from text (refined by `probe`).
DEFAULT_WPM = 150

# Window = one model.generate() call. 2000 words ≈ 13 min audio. VibeVoice is a
# long-form model (the 7B/Large Qwen2 decoder has a 32k-token context), so we
# favour large windows: fewer window joins and more natural cross-sentence
# prosody. Kept under a full 20-min part so a QA-rejected window is cheap to
# re-render. Lower it (--window-words) if you hear long-generation drift.
DEFAULT_WINDOW_WORDS = 2000

# Part = one output .m4a. 20 min ≈ 3,000 words at 150 wpm.
DEFAULT_MAX_PART_MINUTES = 20.0

DEFAULT_MODEL = "microsoft/VibeVoice-1.5B"


@dataclass(frozen=True)
class ModelSpec:
    """A selectable TTS model: an HF repo id (or local path) + optional subfolder.

    `subfolder` is for repos that ship several variants side by side (the 7B
    quantized weights live under `4bit/` and `8bit/`). Quantized weights carry
    their own bitsandbytes `quantization_config` and are auto-detected at load;
    there is nothing to configure here beyond pointing at the right subfolder.
    """

    repo: str
    subfolder: str | None = None
    vram: str = ""  # rough footprint, shown in docs/help
    note: str = ""


# Friendly aliases for `--model`. Anything not listed is treated as a raw HF
# repo id or a local path. The 7B quantized variants are bitsandbytes weights
# (nf4 / int8) and load only on CUDA — bitsandbytes has no MPS/CPU kernels.
MODEL_ALIASES: dict[str, ModelSpec] = {
    "1.5b": ModelSpec(
        "microsoft/VibeVoice-1.5B", vram="~6 GB", note="default; runs on MPS/CUDA/CPU"
    ),
    "7b": ModelSpec(
        "aoi-ot/VibeVoice-Large", vram="~18 GB", note="full-precision 7B; needs a big GPU"
    ),
    "7b-4bit": ModelSpec(
        "DevParker/VibeVoice7b-low-vram", "4bit",
        vram="~6.6 GB", note="nf4-quantized 7B; CUDA only (needs bitsandbytes)",
    ),
    "7b-8bit": ModelSpec(
        "DevParker/VibeVoice7b-low-vram", "8bit",
        vram="~10.6 GB", note="int8-quantized 7B; CUDA only (needs bitsandbytes)",
    ),
}


def resolve_model(model: str, subfolder: str | None = None) -> ModelSpec:
    """Expand a `--model` value (alias | HF repo id | local path) to a ModelSpec.

    An explicit `--model-subfolder` always wins over an alias's default subfolder.
    """
    spec = MODEL_ALIASES.get(model.strip().lower()) or ModelSpec(model)
    if subfolder is not None:
        spec = replace(spec, subfolder=subfolder)
    return spec


DEFAULT_VOICE = "en-mark-welch"
# Every narratelm render is voice cloning (reference-conditioned), and cloned
# voices want more diffusion steps: 20 per upstream guidance. 10 (the fork
# demo default) is audibly flatter; 6 is draft quality. Speed is opt-in via
# --ddpm-steps.
DEFAULT_CFG_SCALE = 1.5  # community "balanced" expressiveness; 1.3 reads flat
DEFAULT_DDPM_STEPS = 20
DEFAULT_SEED = 42

# Silence inserted when concatenating windows into a part (seconds).
PAUSE_BETWEEN_WINDOWS_S = 0.35
PAUSE_BETWEEN_PARAGRAPHS_S = 0.65
PAUSE_SCENE_BREAK_S = 1.2
PAUSE_BETWEEN_PARTS_S = 1.5  # inserted by `combine` between files

# Join smoothing: equal-power crossfade applied at window joins (seconds).
JOIN_CROSSFADE_S = 0.05

# Loudness targets (mono speech): EBU R128-ish audiobook levels.
LOUDNORM_I = -19.0
LOUDNORM_TP = -2.0
LOUDNORM_LRA = 11.0

AAC_BITRATE = "64k"

# QA thresholds (hallucination detection) — ratios of actual/expected duration.
QA_DURATION_RATIO_MIN = 0.55
QA_DURATION_RATIO_MAX = 1.7
QA_MAX_INTERNAL_SILENCE_S = 3.5
QA_MAX_TRAILING_OVERRUN_S = 8.0
QA_MAX_RETRIES = 3


@dataclass(frozen=True)
class GenerationSettings:
    """Everything that affects generated audio. Feeds cache keys and the manifest."""

    model: str = DEFAULT_MODEL
    model_subfolder: str | None = None  # e.g. "4bit"/"8bit" for the quantized 7B repo
    voice: str = DEFAULT_VOICE
    device: str = "auto"
    cfg_scale: float = DEFAULT_CFG_SCALE
    ddpm_steps: int = DEFAULT_DDPM_STEPS
    seed: int = DEFAULT_SEED
    window_words: int = DEFAULT_WINDOW_WORDS
    max_part_minutes: float = DEFAULT_MAX_PART_MINUTES
    wpm: int = DEFAULT_WPM
    rolling_context: bool = False


def output_dir_for(epub_or_json: Path) -> Path:
    """BOOK.epub or BOOK.json -> sibling directory BOOK/ that holds the audio files."""
    return epub_or_json.parent / epub_or_json.stem


def json_path_for(epub_path: Path) -> Path:
    """BOOK.epub -> sibling BOOK.json."""
    return epub_path.parent / (epub_path.stem + ".json")
