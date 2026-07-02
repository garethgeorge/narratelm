"""Device/dtype/attention selection for VibeVoice inference.

Follows the pinned fork's demo exactly: MPS wants float32 + sdpa (bf16 and
flash-attention are unavailable on Metal; the fork's own demo loads float32
there), CUDA wants bfloat16 + flash_attention_2 (with an sdpa fallback handled
at load time), CPU is float32 + sdpa.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DevicePlan:
    device: str  # "mps" | "cuda" | "cpu"
    dtype_name: str  # "float32" | "bfloat16" — resolved to a torch dtype lazily
    attn_implementation: str  # "sdpa" | "flash_attention_2"
    warning: str | None = None


def pick_device(requested: str = "auto") -> DevicePlan:
    """Resolve the requested device against what torch actually offers.

    Imports torch lazily so that extract/combine and the test suite never
    need ML dependencies installed.
    """
    import torch

    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"

    if requested == "mps":
        if not torch.backends.mps.is_available():
            return DevicePlan("cpu", "float32", "sdpa", warning="MPS requested but unavailable; falling back to CPU (very slow)")
        return DevicePlan("mps", "float32", "sdpa")
    if requested == "cuda":
        if not torch.cuda.is_available():
            return DevicePlan("cpu", "float32", "sdpa", warning="CUDA requested but unavailable; falling back to CPU (very slow)")
        return DevicePlan("cuda", "bfloat16", "flash_attention_2")
    if requested == "cpu":
        return DevicePlan("cpu", "float32", "sdpa", warning="CPU inference is many times slower than realtime")
    raise ValueError(f"unknown device {requested!r} (expected auto|mps|cuda|cpu)")


def torch_dtype(plan: DevicePlan):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[plan.dtype_name]
