"""Deterministic no-torch engine for pipeline tests.

Produces a quiet sine whose duration matches the words/wpm estimate, so QA
checks pass by default. Tests can inject misbehaviour per call to exercise
retry and failure paths.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import DEFAULT_WPM, SAMPLE_RATE
from .base import SynthesisRequest, SynthesisResult, TTSEngine


class FakeEngine(TTSEngine):
    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        *,
        wpm: int = DEFAULT_WPM,
        misbehave: Callable[[SynthesisRequest, int], np.ndarray | None] | None = None,
    ):
        """misbehave(request, call_index) may return replacement audio (or None
        to fall through to normal synthesis)."""
        self.wpm = wpm
        self.misbehave = misbehave
        self.calls: list[SynthesisRequest] = []
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.calls.append(request)
        if self.misbehave is not None:
            override = self.misbehave(request, len(self.calls) - 1)
            if override is not None:
                return SynthesisResult(audio=override, sample_rate=self.sample_rate)

        words = sum(len(line.text.split()) for line in request.lines)
        duration_s = max(0.1, words * 60.0 / self.wpm)
        n = int(duration_s * self.sample_rate)
        # frequency varies with seed so retries produce measurably different audio
        freq = 200.0 + (request.seed % 97) * 5.0
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        audio = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            stats={"audio_s": duration_s, "fake": True},
        )

    def fingerprint(self) -> dict:
        return {"engine": "fake", "wpm": self.wpm}
