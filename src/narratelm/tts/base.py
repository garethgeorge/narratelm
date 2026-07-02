"""Engine-agnostic TTS contract.

Single narrator today; the request shape already carries multiple speakers and
per-speaker voice references, so multi-speaker later means building richer
script lines (a future casting step) — not touching engines or the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NARRATOR = "narrator"


@dataclass(frozen=True)
class ScriptLine:
    speaker: str  # logical speaker key, e.g. "narrator"
    text: str


@dataclass(frozen=True)
class VoiceRef:
    """Reference audio conditioning one speaker.

    `wav_path` is the base reference (bundled voice or user-provided).
    `context_audio`, when set, is prior generated audio (24 kHz mono float32)
    appended to the reference — the experimental rolling-context hook.
    """

    name: str
    wav_path: Path
    context_audio: np.ndarray | None = None


@dataclass(frozen=True)
class SynthesisRequest:
    lines: tuple[ScriptLine, ...]
    voices: dict[str, VoiceRef]  # speaker key -> voice
    seed: int
    cfg_scale: float
    ddpm_steps: int


@dataclass
class SynthesisResult:
    audio: np.ndarray  # float32 mono at sample_rate
    sample_rate: int
    stats: dict = field(default_factory=dict)  # rtf, tokens, timing — informational


class TTSEngine(ABC):
    """One loaded model. load() once, synthesize() per window."""

    sample_rate: int = 24_000

    @abstractmethod
    def load(self) -> None:
        """Heavyweight init (model download/load). Idempotent."""

    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> SynthesisResult: ...

    @abstractmethod
    def fingerprint(self) -> dict:
        """Everything identity-relevant for cache keys: model id/rev, engine name/version."""
