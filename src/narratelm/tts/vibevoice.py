"""VibeVoice engine — the only module that touches the community fork's API.

Verified against vibevoice-community/VibeVoice @ 07cb79fe
(demo/inference_from_file.py is the canonical usage):

  processor = VibeVoiceProcessor.from_pretrained(model_id)
  inputs = processor(text=[script], voice_samples=[[wav_path, ...]],
                     padding=True, return_tensors="pt", return_attention_mask=True)
  outputs = model.generate(**inputs, max_new_tokens=None, cfg_scale=...,
                           tokenizer=processor.tokenizer,
                           generation_config={"do_sample": False},
                           is_prefill=True)          # is_prefill=True == voice cloning
  audio = outputs.speech_outputs[0]                   # 24 kHz mono

Script format: newline-joined "Speaker N: <text>" lines, N in 1..4.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from ..config import SAMPLE_RATE
from .base import SynthesisRequest, SynthesisResult, TTSEngine, VoiceRef
from .devices import DevicePlan, pick_device, torch_dtype

FORK_REV = "07cb79feadd2d3fd7f47530d4c964a12857936a0"


def _is_quantized(model_path: str) -> bool:
    """True when the model config declares a `quantization_config` (bitsandbytes)."""
    local = os.path.join(model_path, "config.json")
    try:
        if os.path.exists(local):
            with open(local) as f:
                cfg = json.load(f)
        else:
            from transformers.utils import cached_file

            with open(cached_file(model_path, "config.json")) as f:
                cfg = json.load(f)
    except Exception:
        return False
    return "quantization_config" in cfg


def _require_quantization_support(model_id: str, plan: DevicePlan) -> None:
    """bitsandbytes weights need CUDA + the bitsandbytes package installed."""
    import importlib.util

    if plan.device != "cuda":
        raise RuntimeError(
            f"{model_id!r} is a quantized (bitsandbytes) model, which runs only on "
            f"CUDA — bitsandbytes has no {plan.device} kernels. Use --device cuda on an "
            f"NVIDIA GPU, or pick a non-quantized model (e.g. --model 1.5b)."
        )
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            f"{model_id!r} needs bitsandbytes to load its quantized weights, but it "
            f"isn't installed. Run `uv sync` inside the nix dev shell on a CUDA box "
            f"(bitsandbytes is Linux-only)."
        )


def _quiet_load_warnings() -> None:
    """Silence known-benign chatter emitted while importing/loading the fork.

    All three are expected on our setup and documented as harmless:
      - "audio_utils not available" (the fork's ASR processor falls back to
        soundfile, which is exactly what we want);
      - "APEX FusedRMSNorm not available" (native RMSNorm is fine on MPS/CPU);
      - the Qwen2Tokenizer-vs-VibeVoiceTextTokenizerFast class-mismatch notice
        (the fork loads its tokenizer this way on purpose — see module docstring).
    """
    import warnings

    from transformers.utils import logging as hf_logging

    warnings.filterwarnings("ignore", message="audio_utils not available.*")
    hf_logging.set_verbosity_error()


class VibeVoiceEngine(TTSEngine):
    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        model_id: str,
        *,
        subfolder: str | None = None,
        device: str = "auto",
        ddpm_steps: int = 10,
        cache_dir: str | None = None,
    ):
        self.model_id = model_id
        self.subfolder = subfolder
        self.requested_device = device
        self.ddpm_steps = ddpm_steps
        self.cache_dir = cache_dir
        self.quantized = False  # set at load time from the model config
        self.plan: DevicePlan | None = None
        self._model = None
        self._processor = None
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        _quiet_load_warnings()
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        plan = pick_device(self.requested_device)
        self.plan = plan
        dtype = torch_dtype(plan)

        # A subfolder repo (the quantized 7B ships variants under 4bit/, 8bit/)
        # can't be loaded via a `subfolder=` kwarg — VibeVoiceProcessor forwards
        # that kwarg to the Qwen tokenizer fetch, which then 404s. So materialise
        # just that subfolder locally and load everything from the plain path.
        model_path = self._materialize(self.model_id, self.subfolder, self.cache_dir)

        self.quantized = _is_quantized(model_path)
        if self.quantized:
            _require_quantization_support(self.model_id, plan)

        self._processor = VibeVoiceProcessor.from_pretrained(model_path, cache_dir=self.cache_dir)

        def _load(attn: str):
            kwargs = dict(torch_dtype=dtype, attn_implementation=attn, cache_dir=self.cache_dir)
            if self.quantized:
                # bitsandbytes weights: device_map places them; never call .to().
                # The embedded quantization_config is applied automatically.
                return VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path, device_map=plan.device, **kwargs
                )
            if plan.device == "mps":
                m = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path, device_map=None, **kwargs
                )
                m.to("mps")
                return m
            return VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path, device_map=plan.device, **kwargs
            )

        try:
            self._model = _load(plan.attn_implementation)
        except Exception:
            if plan.attn_implementation != "flash_attention_2":
                raise
            # flash-attn missing on this CUDA box; the fork itself falls back to sdpa.
            self.plan = DevicePlan(plan.device, plan.dtype_name, "sdpa", plan.warning)
            self._model = _load("sdpa")

        self._model.eval()
        self._model.set_ddpm_inference_steps(num_steps=self.ddpm_steps)

    def _materialize(self, model_id: str, subfolder: str | None, cache_dir: str | None) -> str:
        """Return a local path to load from. Downloads only `subfolder` when set."""
        if not subfolder:
            return model_id
        local = os.path.join(model_id, subfolder)
        if os.path.isdir(local):  # model_id is already a local checkout
            return local
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            model_id, allow_patterns=[f"{subfolder}/*"], cache_dir=cache_dir
        )
        return os.path.join(root, subfolder)

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.load()
        import torch

        assert self._model is not None and self._processor is not None and self.plan is not None

        script, voice_paths = self._build_script(request)

        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)

        inputs = self._processor(
            text=[script],
            voice_samples=[voice_paths],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.plan.device)

        start = time.monotonic()
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=request.cfg_scale,
            tokenizer=self._processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            is_prefill=True,
        )
        if self.plan.device == "mps":
            torch.mps.synchronize()
        elapsed = time.monotonic() - start

        speech = outputs.speech_outputs[0]
        if speech is None:
            raise RuntimeError("VibeVoice produced no audio for this window")
        audio = speech.detach().to(torch.float32).cpu().numpy().reshape(-1)

        duration = len(audio) / self.sample_rate
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            stats={
                "generation_s": round(elapsed, 2),
                "audio_s": round(duration, 2),
                "rtf": round(elapsed / duration, 3) if duration > 0 else None,
            },
        )

    def fingerprint(self) -> dict:
        return {
            "engine": "vibevoice",
            "fork_rev": FORK_REV,
            "model": self.model_id,
            "subfolder": self.subfolder,  # distinct weights (e.g. 4bit vs 8bit)
            "ddpm_steps": self.ddpm_steps,
            # device/dtype/attn deliberately EXCLUDED: they change performance,
            # not identity enough to force a full re-render when moving machines.
        }

    def _build_script(self, request: SynthesisRequest) -> tuple[str, list[str]]:
        """Map logical speakers to 'Speaker N' slots in order of first appearance."""
        order: list[str] = []
        for line in request.lines:
            if line.speaker not in order:
                order.append(line.speaker)
        if len(order) > 4:
            raise ValueError(f"VibeVoice supports at most 4 speakers, got {len(order)}")

        slot = {speaker: i + 1 for i, speaker in enumerate(order)}
        script_lines = []
        for line in request.lines:
            text = " ".join(line.text.split())
            script_lines.append(f"Speaker {slot[line.speaker]}: {text}")
        script = "\n".join(script_lines).replace("’", "'")

        voice_paths = [self._voice_path(request.voices[speaker]) for speaker in order]
        return script, voice_paths

    def _voice_path(self, voice: VoiceRef) -> str:
        """Base reference path, or a composite temp WAV when rolling context is on."""
        if voice.context_audio is None:
            return str(voice.wav_path)

        import soundfile as sf

        base, rate = sf.read(voice.wav_path, dtype="float32", always_2d=False)
        if base.ndim > 1:
            base = base.mean(axis=1)
        if rate != self.sample_rate:
            # cheap linear resample; reference conditioning is tolerant
            n = int(len(base) * self.sample_rate / rate)
            base = np.interp(
                np.linspace(0, len(base) - 1, n), np.arange(len(base)), base
            ).astype(np.float32)
        composite = np.concatenate([base, voice.context_audio.astype(np.float32)])

        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="narratelm-ctx-")
        out = Path(self._tmpdir.name) / f"{voice.name}-ctx.wav"
        sf.write(out, composite, self.sample_rate)
        return str(out)
