"""Audio assembly layer: concatenate TTS windows → parts → chapterized .m4b."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from narratelm.config import (
    AAC_BITRATE,
    JOIN_CROSSFADE_S,
    LOUDNORM_I,
    LOUDNORM_LRA,
    LOUDNORM_TP,
    PAUSE_BETWEEN_PARTS_S,
    SAMPLE_RATE,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> str:
    """Run a subprocess, return combined stderr; raise RuntimeError on failure."""
    result = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stderr = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        tail = "\n".join(stderr.splitlines()[-30:])
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n{' '.join(cmd)}\n\n{tail}"
        )
    return stderr


def _ffmpeg(*args: str) -> str:
    return _run(["ffmpeg", "-nostdin", "-hide_banner", "-y", *args])


def _ffprobe(*args: str) -> str:
    result = subprocess.run(
        ["ffprobe", "-hide_banner", *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        tail = "\n".join(stderr.splitlines()[-30:])
        raise RuntimeError(f"ffprobe failed:\n{tail}")
    return stdout


# ---------------------------------------------------------------------------
# Pure audio helpers
# ---------------------------------------------------------------------------

def _rms(audio: np.ndarray) -> float:
    """RMS over voiced samples (abs > 1e-4); returns 0.0 if none."""
    voiced = audio[np.abs(audio) > 1e-4]
    if voiced.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(voiced.astype(np.float64) ** 2)))


def _fade_ramp(n: int, fade_in: bool) -> np.ndarray:
    """5ms-style linear ramp of length n samples."""
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return ramp if fade_in else ramp[::-1]


def _equal_power_crossfade(
    tail: np.ndarray, head: np.ndarray, n: int
) -> np.ndarray:
    """Equal-power crossfade between tail[-n:] and head[:n], returns blended region."""
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    return tail[-n:] * fade_out + head[:n] * fade_in


def assemble_windows(
    chunks: list[np.ndarray],
    pauses_after_s: list[float],
    *,
    sample_rate: int = SAMPLE_RATE,
    crossfade_s: float = JOIN_CROSSFADE_S,
    rms_match: bool = True,
) -> np.ndarray:
    """Concatenate window audio into one part-level array.

    - float32 mono in/out; len(pauses_after_s) == len(chunks).
    - rms_match: gain-scale each chunk toward the median RMS, clamped to [0.5, 2.0].
    - Insert pauses_after_s[i] seconds of silence after chunk i.
    - Equal-power crossfade at direct joins (pause < 0.05 s); plain concat with
      5 ms fade ramps at silence edges otherwise.
    """
    if len(chunks) != len(pauses_after_s):
        raise ValueError("len(chunks) must equal len(pauses_after_s)")
    if not chunks:
        return np.zeros(0, dtype=np.float32)

    # --- RMS matching ---
    scaled: list[np.ndarray] = []
    if rms_match:
        rms_vals = [_rms(c) for c in chunks]
        voiced_rms = [r for r in rms_vals if r > 0.0]
        median_rms = float(np.median(voiced_rms)) if voiced_rms else 0.0
        for chunk, r in zip(chunks, rms_vals):
            if median_rms > 0.0 and r > 0.0:
                gain = np.clip(median_rms / r, 0.5, 2.0)
                scaled.append((chunk * gain).astype(np.float32))
            else:
                scaled.append(chunk.astype(np.float32))
    else:
        scaled = [c.astype(np.float32) for c in chunks]

    crossfade_n = int(crossfade_s * sample_rate)
    DIRECT_JOIN_THRESH = 0.05  # seconds
    FADE_MS = int(0.005 * sample_rate)  # 5 ms click-prevention ramp

    parts: list[np.ndarray] = []

    for i, (chunk, pause_s) in enumerate(zip(scaled, pauses_after_s)):
        is_last = i == len(scaled) - 1
        next_chunk = scaled[i + 1] if not is_last else None

        direct_join = (
            not is_last
            and pause_s < DIRECT_JOIN_THRESH
            and next_chunk is not None
            and crossfade_n > 0
            and len(chunk) >= crossfade_n
            and len(next_chunk) >= crossfade_n
        )

        if direct_join:
            # Emit chunk body (without tail that will be crossfaded)
            parts.append(chunk[:-crossfade_n])
            # Crossfaded region replaces the join
            blend = _equal_power_crossfade(chunk, next_chunk, crossfade_n)
            parts.append(blend)
            # next chunk head already consumed; advance by mutating scaled[i+1]
            scaled[i + 1] = next_chunk[crossfade_n:]
            # No silence inserted for direct joins
        else:
            parts.append(chunk)
            if pause_s > 0.0:
                silence_n = int(pause_s * sample_rate)
                if silence_n > 0:
                    # 5 ms fade-out ramp at end of chunk (click prevention)
                    if len(parts) and len(parts[-1]) >= FADE_MS:
                        tail = parts[-1].copy()
                        tail[-FADE_MS:] *= _fade_ramp(FADE_MS, fade_in=False)
                        parts[-1] = tail
                    silence = np.zeros(silence_n, dtype=np.float32)
                    parts.append(silence)

    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def trim_leading_silence(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    threshold_db: float = -45.0,
    keep_s: float = 0.2,
) -> np.ndarray:
    """Remove leading silence below threshold_db, keeping keep_s seconds."""
    threshold_linear = 10 ** (threshold_db / 20.0)
    voiced = np.where(np.abs(audio) > threshold_linear)[0]
    if voiced.size == 0:
        return audio
    first_voiced = voiced[0]
    keep_samples = int(keep_s * sample_rate)
    start = max(0, first_voiced - keep_samples)
    return audio[start:]


def trim_offset(
    audio: np.ndarray,
    offset_s: float,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Drop the first offset_s seconds (disclaimer trim)."""
    n = int(offset_s * sample_rate)
    return audio[n:]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_part_m4a(
    audio: np.ndarray,
    out_path: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    bitrate: str = AAC_BITRATE,
    title: str | None = None,
    track: int | None = None,
    loudnorm: bool = True,
) -> None:
    """Write temp WAV → ffmpeg AAC-LC mono m4a (atomic write).

    Two-pass loudnorm when loudnorm=True; falls back to single-pass with a
    warnings.warn on parse failure.
    """
    tmp_m4a = out_path.with_suffix(".m4a.tmp")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        sf.write(wav_path, audio, sample_rate, subtype="FLOAT")

        meta_args: list[str] = []
        if title is not None:
            meta_args += ["-metadata", f"title={title}"]
        if track is not None:
            meta_args += ["-metadata", f"track={track}"]

        common_out = [
            "-ac", "1",
            "-ar", str(sample_rate),
            "-b:a", bitrate,
            "-c:a", "aac",
        ]

        # m4a/ipod container; explicit -f ipod so ffmpeg doesn't choke on .tmp extension
        fmt_args = ["-f", "ipod"]

        if loudnorm:
            loudnorm_filter = (
                f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
                ":print_format=json"
            )
            # Pass 1: measure
            measured: dict[str, Any] | None = None
            try:
                stderr1 = _ffmpeg(
                    "-i", wav_path,
                    "-af", loudnorm_filter,
                    "-f", "null", "-",
                )
                # JSON block is embedded in stderr
                measured = _parse_loudnorm_json(stderr1)
            except Exception:
                measured = None

            if measured is not None and not _loudnorm_values_sane(measured):
                # e.g. all-silent audio measures input_i = -inf, which pass 2
                # rejects; there is nothing to normalize — encode as-is.
                measured = None
                loudnorm = False

            if measured is not None:
                # Pass 2: apply with measured values
                lp2_filter = (
                    f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
                    f":measured_I={measured['input_i']}"
                    f":measured_TP={measured['input_tp']}"
                    f":measured_LRA={measured['input_lra']}"
                    f":measured_thresh={measured['input_thresh']}"
                    f":offset={measured['target_offset']}"
                    ":linear=true:print_format=summary"
                )
                _ffmpeg(
                    "-i", wav_path,
                    "-af", lp2_filter,
                    *common_out,
                    *meta_args,
                    *fmt_args,
                    str(tmp_m4a),
                )
            elif loudnorm:
                warnings.warn(
                    "loudnorm pass-1 JSON parse failed; falling back to single-pass loudnorm",
                    stacklevel=2,
                )
                _ffmpeg(
                    "-i", wav_path,
                    "-af", f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}",
                    *common_out,
                    *meta_args,
                    *fmt_args,
                    str(tmp_m4a),
                )

        if not loudnorm:
            _ffmpeg(
                "-i", wav_path,
                *common_out,
                *meta_args,
                *fmt_args,
                str(tmp_m4a),
            )

        os.replace(tmp_m4a, out_path)
    finally:
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass
        # Clean up tmp_m4a if encode failed before os.replace
        try:
            if tmp_m4a.exists():
                tmp_m4a.unlink()
        except Exception:
            pass


def _loudnorm_values_sane(measured: dict[str, Any]) -> bool:
    """Pass-2 rejects out-of-range measured values (silence measures -inf)."""
    import math

    try:
        values = {k: float(measured[k]) for k in
                  ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    except (KeyError, ValueError):
        return False
    if not all(math.isfinite(v) for v in values.values()):
        return False
    return -99.0 <= values["input_i"] <= 0.0


def _parse_loudnorm_json(stderr: str) -> dict[str, Any]:
    """Extract the JSON block that ffmpeg loudnorm prints to stderr."""
    # Find the { ... } block after the loudnorm summary line
    start = stderr.rfind("{")
    end = stderr.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON block found in loudnorm stderr")
    return json.loads(stderr[start:end])


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def probe_duration_s(path: Path) -> float:
    """Return duration in seconds via ffprobe."""
    out = _ffprobe(
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    )
    data = json.loads(out)
    return float(data["format"]["duration"])


def probe_chapters(path: Path) -> list[dict]:
    """Return list of {title, start_s, end_s} dicts from ffprobe chapters."""
    out = _ffprobe(
        "-v", "quiet",
        "-show_chapters",
        "-of", "json",
        str(path),
    )
    data = json.loads(out)
    chapters = []
    for ch in data.get("chapters", []):
        tags = ch.get("tags", {})
        title = tags.get("title", "")
        time_base_str = ch.get("time_base", "1/1")
        # time_base is "num/den"
        num, den = (int(x) for x in time_base_str.split("/"))
        tb = num / den
        chapters.append(
            {
                "title": title,
                "start_s": int(ch["start"]) * tb,
                "end_s": int(ch["end"]) * tb,
            }
        )
    return chapters


# ---------------------------------------------------------------------------
# FFMETADATA
# ---------------------------------------------------------------------------

def _escape_ffmeta(value: str) -> str:
    """Escape ffmetadata special characters: = ; # \\ and newline."""
    value = value.replace("\\", "\\\\")
    value = value.replace("=", "\\=")
    value = value.replace(";", "\\;")
    value = value.replace("#", "\\#")
    value = value.replace("\n", "\\\n")
    return value


def build_ffmetadata(
    book_title: str,
    author: str,
    chapters: list[tuple[str, float]],
) -> str:
    """Build FFMETADATA1 text with global tags and CHAPTER blocks.

    chapters is a list of (title, duration_s) tuples; START/END are cumulative
    milliseconds with TIMEBASE=1/1000.
    """
    lines: list[str] = [
        ";FFMETADATA1",
        f"title={_escape_ffmeta(book_title)}",
        f"artist={_escape_ffmeta(author)}",
        f"album={_escape_ffmeta(book_title)}",
        "genre=Audiobook",
        "",
    ]

    cursor_ms = 0
    for title, duration_s in chapters:
        start_ms = cursor_ms
        end_ms = cursor_ms + int(round(duration_s * 1000))
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={_escape_ffmeta(title)}",
            "",
        ]
        cursor_ms = end_ms

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# combine_m4b
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChapterMark:
    """Associates a chapter title with the ordered part m4a files it comprises."""

    title: str
    files: tuple[Path, ...]


def _generate_silent_m4a(
    duration_s: float,
    out_path: Path,
    *,
    sample_rate: int,
    bitrate: str,
) -> None:
    """Generate a silent AAC m4a of the given duration."""
    _ffmpeg(
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", str(duration_s),
        "-c:a", "aac",
        "-b:a", bitrate,
        "-ac", "1",
        "-ar", str(sample_rate),
        str(out_path),
    )


def _escape_concat_path(p: Path) -> str:
    """Escape a path for use in ffmpeg concat demuxer file list."""
    return str(p).replace("'", "'\\''")


def combine_m4b(
    chapters: list[ChapterMark],
    out_path: Path,
    *,
    book_title: str,
    author: str,
    cover: Path | None = None,
    gap_s: float = PAUSE_BETWEEN_PARTS_S,
    sample_rate: int = SAMPLE_RATE,
    bitrate: str = AAC_BITRATE,
) -> None:
    """Combine chapter part files into a single chapterized .m4b.

    Gaps between files are real AAC silence segments interleaved in the concat
    list. Chapter durations account for inter-file gaps.
    Raises FileNotFoundError if any part file is missing (checked before encoding).
    Raises RuntimeError if output duration deviates more than 2% from expected.
    """
    # --- Validate all input files exist first ---
    missing = [
        str(f)
        for cm in chapters
        for f in cm.files
        if not f.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing part file(s): {', '.join(missing)}"
        )

    tmp_out = out_path.with_suffix(".m4b.tmp")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # --- Probe durations ---
        # flat list of (path, duration_s) in order, grouped per chapter
        chapter_file_durations: list[list[tuple[Path, float]]] = []
        for cm in chapters:
            file_durs: list[tuple[Path, float]] = []
            for f in cm.files:
                file_durs.append((f, probe_duration_s(f)))
            chapter_file_durations.append(file_durs)

        # --- Generate the single reusable gap file ---
        gap_file = tmp / "gap.m4a"
        _generate_silent_m4a(gap_s, gap_file, sample_rate=sample_rate, bitrate=bitrate)
        gap_duration = probe_duration_s(gap_file)

        # --- Build concat list and chapter duration metadata ---
        concat_entries: list[Path] = []
        chapter_meta: list[tuple[str, float]] = []  # (title, duration_s)

        for cm_idx, (cm, file_durs) in enumerate(
            zip(chapters, chapter_file_durations)
        ):
            ch_duration = 0.0
            for f_idx, (f, dur) in enumerate(file_durs):
                concat_entries.append(f)
                ch_duration += dur
                # Insert gap between files within a chapter AND between chapters
                is_last_file_of_last_chapter = (
                    cm_idx == len(chapters) - 1
                    and f_idx == len(file_durs) - 1
                )
                if not is_last_file_of_last_chapter:
                    concat_entries.append(gap_file)
                    ch_duration += gap_duration

            chapter_meta.append((cm.title, ch_duration))

        expected_duration = sum(d for _, d in chapter_meta)

        # --- Write concat list file ---
        concat_list = tmp / "concat.txt"
        lines = []
        for entry in concat_entries:
            escaped = _escape_concat_path(entry)
            lines.append(f"file '{escaped}'")
        concat_list.write_text("\n".join(lines) + "\n")

        # --- Write ffmetadata file ---
        meta_text = build_ffmetadata(book_title, author, chapter_meta)
        meta_file = tmp / "metadata.txt"
        meta_file.write_text(meta_text, encoding="utf-8")

        # --- Build ffmpeg command ---
        cmd: list[str] = [
            "ffmpeg", "-nostdin", "-hide_banner", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-i", str(meta_file),
        ]

        cover_input_idx: int | None = None
        if cover is not None and cover.exists():
            cmd += ["-i", str(cover)]
            cover_input_idx = 2

        cmd += ["-map", "0:a"]
        if cover_input_idx is not None:
            cmd += ["-map", f"{cover_input_idx}:v"]

        cmd += ["-map_metadata", "1", "-c:a", "copy"]

        if cover_input_idx is not None:
            # Detect cover format by magic bytes / suffix
            suffix = cover.suffix.lower()
            is_jpg = suffix in (".jpg", ".jpeg")
            if not is_jpg:
                # sniff magic bytes
                try:
                    magic = cover.read_bytes()[:4]
                    is_jpg = magic[:2] == b"\xff\xd8"
                except Exception:
                    is_jpg = False
            codec = "mjpeg" if is_jpg else "png"
            cmd += ["-c:v", codec, "-disposition:v", "attached_pic"]

        cmd += ["-movflags", "+faststart", "-f", "mp4", str(tmp_out)]

        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stderr = result.stderr.decode("utf-8", errors="replace")
        if result.returncode != 0:
            tail = "\n".join(stderr.splitlines()[-30:])
            raise RuntimeError(f"ffmpeg combine failed:\n{tail}")

        # --- Validate duration ---
        actual_duration = probe_duration_s(tmp_out)
        # 5% tolerance: AAC frame-boundary rounding and container overhead can add
        # ~2-4% on short clips; 5% still catches grossly wrong output (missing chapters, etc.)
        tolerance = 0.05
        if expected_duration > 0:
            ratio = actual_duration / expected_duration
            if not (1 - tolerance <= ratio <= 1 + tolerance):
                raise RuntimeError(
                    f"Output duration {actual_duration:.2f}s deviates from expected "
                    f"{expected_duration:.2f}s (ratio {ratio:.3f}, tolerance ±5%)"
                )

        os.replace(tmp_out, out_path)
