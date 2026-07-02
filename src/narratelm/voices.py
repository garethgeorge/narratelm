"""Bundled voice registry and --voice resolution."""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .config import SAMPLE_RATE

MIN_REFERENCE_S = 10.0
MAX_REFERENCE_S = 90.0


@dataclass(frozen=True)
class BundledVoice:
    name: str
    path: Path
    gender: str
    duration_s: float
    notes: str | None = None


def _voices_dir() -> Path:
    return Path(str(resources.files("narratelm") / "data" / "voices"))


def bundled_voices() -> list[BundledVoice]:
    vdir = _voices_dir()
    manifest = json.loads((vdir / "voices.json").read_text(encoding="utf-8"))
    out = []
    for v in manifest["voices"]:
        out.append(
            BundledVoice(
                name=v["name"],
                path=vdir / v["file"],
                gender=v.get("gender", "?"),
                duration_s=float(v.get("duration_s", 0.0)),
                notes=v.get("notes"),
            )
        )
    return out


def resolve_voice(arg: str, *, cache_dir: Path | None = None) -> tuple[str, Path]:
    """Resolve --voice into (name, wav_path).

    Accepts a bundled voice name (case-insensitive) or a path to reference
    audio. Non-WAV / non-24kHz / non-mono files are converted via ffmpeg into
    cache_dir. Raises ValueError with the list of bundled names on miss.
    """
    for v in bundled_voices():
        if v.name.lower() == arg.lower():
            return v.name, v.path

    src = Path(arg).expanduser()
    if not src.exists():
        names = ", ".join(v.name for v in bundled_voices())
        raise ValueError(f"--voice {arg!r} is neither a bundled voice ({names}) nor an existing file")

    info = _probe_audio(src)
    if info is not None:
        duration, rate, channels = info
        if duration < MIN_REFERENCE_S:
            warnings.warn(
                f"reference audio {src.name} is only {duration:.1f}s; 30-60s of clean speech clones far better"
            )
        elif duration > MAX_REFERENCE_S:
            warnings.warn(
                f"reference audio {src.name} is {duration:.0f}s; only the model's prompt window benefits — consider trimming to 30-60s"
            )
        if src.suffix.lower() == ".wav" and rate == SAMPLE_RATE and channels == 1:
            return src.stem, src

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "narratelm" / "voices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    converted = cache_dir / (src.stem + f".{SAMPLE_RATE}mono.wav")
    if not converted.exists():
        if shutil.which("ffmpeg") is None:
            raise ValueError(f"{src} needs conversion to {SAMPLE_RATE} Hz mono WAV but ffmpeg is not on PATH")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-y", "-i", str(src),
             "-ar", str(SAMPLE_RATE), "-ac", "1", str(converted)],
            check=True, capture_output=True,
        )
    return src.stem, converted


def _probe_audio(path: Path) -> tuple[float, int, int] | None:
    """(duration_s, sample_rate, channels) via ffprobe, or None if unprobeable."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels:format=duration",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        )
        data = json.loads(res.stdout)
        stream = data["streams"][0]
        return (
            float(data["format"]["duration"]),
            int(stream["sample_rate"]),
            int(stream["channels"]),
        )
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError, json.JSONDecodeError):
        return None
