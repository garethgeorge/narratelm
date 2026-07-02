"""Cheap numpy checks that catch VibeVoice's known failure modes.

Hallucinations manifest as runaway duration (music/garble continuing past the
text), dead output, or long internal silences. All checks are heuristics —
failing windows get retried with a derived seed; repeatedly failing windows are
accepted-best-effort and flagged `suspect` for spot-listening.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import (
    QA_DURATION_RATIO_MAX,
    QA_DURATION_RATIO_MIN,
    QA_MAX_INTERNAL_SILENCE_S,
    QA_MAX_TRAILING_OVERRUN_S,
)

_FRAME_S = 0.05
_SILENCE_RMS = 10 ** (-50 / 20)  # -50 dBFS


@dataclass
class QAResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Lower is better; used to pick the least-bad attempt after retries."""
        ratio = self.metrics.get("duration_ratio", 10.0)
        silence = self.metrics.get("max_internal_silence_s", 0.0)
        return abs(np.log(max(ratio, 1e-3))) + silence / 10.0


def frame_rms(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(1, int(_FRAME_S * sample_rate))
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    trimmed = audio[: n * frame].reshape(n, frame)
    return np.sqrt((trimmed.astype(np.float64) ** 2).mean(axis=1))


def check_window(
    audio: np.ndarray,
    sample_rate: int,
    word_count: int,
    wpm: int,
) -> QAResult:
    expected_s = max(1.0, word_count * 60.0 / wpm)
    actual_s = len(audio) / sample_rate
    ratio = actual_s / expected_s

    rms = frame_rms(audio, sample_rate)
    metrics: dict = {
        "expected_s": round(expected_s, 2),
        "actual_s": round(actual_s, 2),
        "duration_ratio": round(ratio, 3),
    }
    reasons: list[str] = []

    if len(rms) == 0 or float(rms.max()) < _SILENCE_RMS:
        return QAResult(False, ["audio is empty or silent"], metrics)

    if ratio < QA_DURATION_RATIO_MIN:
        reasons.append(f"too short: {actual_s:.1f}s vs ~{expected_s:.1f}s expected")
    if ratio > QA_DURATION_RATIO_MAX:
        reasons.append(f"too long: {actual_s:.1f}s vs ~{expected_s:.1f}s expected (music/garble?)")
    if actual_s - expected_s > QA_MAX_TRAILING_OVERRUN_S:
        reason = f"runs {actual_s - expected_s:.1f}s past expected end"
        if reason not in reasons:
            reasons.append(reason)

    voiced = rms >= _SILENCE_RMS
    if voiced.any():
        first, last = int(np.argmax(voiced)), len(voiced) - 1 - int(np.argmax(voiced[::-1]))
        internal = voiced[first : last + 1]
        # longest run of silent frames strictly inside the voiced span
        max_run = 0
        run = 0
        for v in internal:
            run = 0 if v else run + 1
            max_run = max(max_run, run)
        max_sil = max_run * _FRAME_S
        trailing_sil = (len(voiced) - 1 - last) * _FRAME_S
        metrics["max_internal_silence_s"] = round(max_sil, 2)
        metrics["trailing_silence_s"] = round(trailing_sil, 2)
        if max_sil > QA_MAX_INTERNAL_SILENCE_S:
            reasons.append(f"internal silence of {max_sil:.1f}s")

    return QAResult(ok=not reasons, reasons=reasons, metrics=metrics)
