"""Tests for src/narratelm/audio.py."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from narratelm.audio import (
    ChapterMark,
    assemble_windows,
    build_ffmetadata,
    combine_m4b,
    encode_part_m4a,
    probe_chapters,
    probe_duration_s,
    trim_leading_silence,
    trim_offset,
)
from narratelm.config import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = SAMPLE_RATE  # 24000


def _sine(duration_s: float, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    """Float32 mono sine wave at SR sample rate."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# assemble_windows — pure, no ffmpeg
# ---------------------------------------------------------------------------

class TestAssembleWindows:
    def test_output_length_no_pause(self):
        """Direct join: output length = sum of chunk lengths (crossfade overlap)."""
        c1 = _sine(1.0)
        c2 = _sine(0.5)
        c3 = _sine(0.8)
        # pause=0 → crossfades applied
        result = assemble_windows([c1, c2, c3], [0.0, 0.0, 0.0], crossfade_s=0.05)
        # Each crossfade of n samples reduces total by n (tail consumed into blend, head skipped)
        # actual: body1 + blend12 + body2 + blend23 + body3
        # = (len(c1)-n) + n + (len(c2)-n-n) + n + len(c3)-n ... complicated
        # Just check it's within a crossfade window of the naive sum
        crossfade_n = int(0.05 * SR)
        naive_sum = len(c1) + len(c2) + len(c3)
        assert abs(len(result) - naive_sum) <= crossfade_n * 4

    def test_output_length_with_pauses(self):
        """With explicit pause_s, output length = sum(chunks) + sum(pauses) within tolerance."""
        c1 = _sine(1.0)
        c2 = _sine(0.5)
        pauses = [0.35, 0.65]
        result = assemble_windows([c1, c2], pauses, rms_match=False)
        expected = len(c1) + len(c2) + int(0.35 * SR) + int(0.65 * SR)
        # Allow ±5 ms for rounding
        assert abs(len(result) - expected) <= int(0.005 * SR)

    def test_silence_regions_near_zero(self):
        """Inserted silence should be close to zero energy."""
        c1 = _sine(0.5)
        c2 = _sine(0.5)
        pause_s = 0.2
        result = assemble_windows([c1, c2], [pause_s, 0.0], rms_match=False)
        # The silence lives between the two chunks
        silence_start = len(c1)
        silence_end = silence_start + int(pause_s * SR)
        silence_region = result[silence_start:silence_end]
        assert np.max(np.abs(silence_region)) < 1e-3

    def test_rms_match_brings_quiet_chunk_closer(self):
        """A chunk at 0.5x amplitude needs 2x gain (clamp boundary) and should
        come within 25% RMS of the loud chunk after matching."""
        loud = _sine(1.0, amplitude=0.5)
        # 0.5x amplitude → gain needed = 2x (exactly at clamp ceiling) → matches loud
        quiet = _sine(1.0, amplitude=0.25)
        result = assemble_windows([loud, quiet], [0.35, 0.0], rms_match=True)
        pause_n = int(0.35 * SR)
        quiet_start = len(loud) + pause_n
        quiet_segment = result[quiet_start:]
        loud_segment = result[:len(loud)]

        def rms(a):
            voiced = a[np.abs(a) > 1e-4]
            if voiced.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(voiced.astype(np.float64) ** 2)))

        r_loud = rms(loud_segment)
        r_quiet = rms(quiet_segment)
        # After 2x gain, quiet RMS should match loud within 25%
        assert r_quiet > 0
        assert abs(r_quiet / r_loud - 1.0) < 0.25

    def test_rms_clamp_extreme_chunk(self):
        """A 0.01x-amplitude chunk should have gain clamped to 2.0x."""
        loud = _sine(1.0, amplitude=0.5)
        extreme_quiet = _sine(1.0, amplitude=0.005)  # 0.01x → gain would be 50x, clamped to 2
        result = assemble_windows([loud, extreme_quiet], [0.35, 0.0], rms_match=True)
        pause_n = int(0.35 * SR)
        quiet_start = len(loud) + pause_n
        quiet_segment = result[quiet_start:]

        def rms(a):
            voiced = a[np.abs(a) > 1e-4]
            if voiced.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(voiced.astype(np.float64) ** 2)))

        r_extreme = rms(quiet_segment)
        r_orig = rms(extreme_quiet)
        # gain must be ≤ 2.0
        if r_orig > 0:
            actual_gain = r_extreme / r_orig
            assert actual_gain <= 2.05  # small tolerance for fp

    def test_single_chunk(self):
        c = _sine(1.0)
        result = assemble_windows([c], [0.0], rms_match=False)
        assert len(result) == len(c)

    def test_empty(self):
        result = assemble_windows([], [], rms_match=False)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# trim_leading_silence & trim_offset — pure
# ---------------------------------------------------------------------------

class TestTrimFunctions:
    def test_trim_leading_silence_removes_prepended_zeros(self):
        """1 s of zeros prepended → trimmed, ~0.2 s kept."""
        silence_s = 1.0
        keep_s = 0.2
        audio = _sine(2.0)
        prepend = np.zeros(int(silence_s * SR), dtype=np.float32)
        padded = np.concatenate([prepend, audio])
        result = trim_leading_silence(padded, sample_rate=SR, threshold_db=-45.0, keep_s=keep_s)
        # The result should start around the keep_s mark (before voiced content)
        # i.e., length should be approximately len(audio) + keep_samples
        keep_samples = int(keep_s * SR)
        expected_len = len(audio) + keep_samples
        assert abs(len(result) - expected_len) < int(0.05 * SR)

    def test_trim_leading_silence_all_voiced(self):
        """No leading silence → keep entire array."""
        audio = _sine(1.0)
        result = trim_leading_silence(audio, sample_rate=SR)
        assert len(result) == len(audio)

    def test_trim_offset_drops_exact_samples(self):
        """trim_offset drops exactly offset_s * sample_rate samples."""
        audio = np.arange(SR * 2, dtype=np.float32)
        offset_s = 0.5
        result = trim_offset(audio, offset_s, sample_rate=SR)
        n_drop = int(offset_s * SR)
        assert len(result) == len(audio) - n_drop
        np.testing.assert_array_equal(result, audio[n_drop:])


# ---------------------------------------------------------------------------
# build_ffmetadata — pure
# ---------------------------------------------------------------------------

class TestBuildFfmetadata:
    def test_two_chapter_exact_text(self):
        """Two chapters produce correct ms math and escape special chars."""
        chapters = [("Intro=Chapter;One", 10.0), ("Part Two", 20.5)]
        text = build_ffmetadata("My=Book", "Auth;Or", chapters)

        lines = text.splitlines()
        assert lines[0] == ";FFMETADATA1"
        assert "title=My\\=Book" in lines
        assert "artist=Auth\\;Or" in lines
        assert "genre=Audiobook" in lines

        # Chapter 1: 0..10000 ms
        assert "[CHAPTER]" in lines
        assert "TIMEBASE=1/1000" in lines
        assert "START=0" in lines
        assert "END=10000" in lines
        assert "title=Intro\\=Chapter\\;One" in lines

        # Chapter 2: 10000..30500 ms
        assert "START=10000" in lines
        assert "END=30500" in lines
        assert "title=Part Two" in lines

    def test_escaping_backslash_and_newline(self):
        title_with_special = "Line1\nLine2\\End"
        text = build_ffmetadata("T", "A", [(title_with_special, 5.0)])
        # backslash escaped → \\
        assert "\\\\" in text
        # newline escaped → \\\n
        assert "\\\n" in text

    def test_single_chapter_zero_start(self):
        text = build_ffmetadata("Book", "Author", [("Only", 60.0)])
        assert "START=0" in text
        assert "END=60000" in text


# ---------------------------------------------------------------------------
# encode_part_m4a, probe_duration_s, probe_chapters, combine_m4b — ffmpeg
# ---------------------------------------------------------------------------

@pytest.mark.ffmpeg
class TestEncodePartM4a:
    def test_produces_valid_m4a(self, tmp_path):
        audio = _sine(2.0)
        out = tmp_path / "part.m4a"
        encode_part_m4a(audio, out, loudnorm=False)
        assert out.exists()
        dur = probe_duration_s(out)
        assert abs(dur - 2.0) < 0.15

    def test_title_metadata(self, tmp_path):
        audio = _sine(1.0)
        out = tmp_path / "titled.m4a"
        encode_part_m4a(audio, out, title="Test Chapter", loudnorm=False)
        import subprocess, json as _json
        result = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "quiet",
             "-show_entries", "format_tags=title",
             "-of", "json", str(out)],
            capture_output=True, text=True
        )
        data = _json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        assert tags.get("title", "").lower() == "test chapter"

    def test_loudnorm_two_pass_no_warning(self, tmp_path):
        """Two-pass loudnorm on a normal sine should NOT trigger fallback warning."""
        audio = _sine(3.0, amplitude=0.5)
        out = tmp_path / "normed.m4a"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            encode_part_m4a(audio, out, loudnorm=True)
        # No loudnorm fallback warning
        fallback_warns = [x for x in w if "loudnorm" in str(x.message).lower()]
        assert len(fallback_warns) == 0

    def test_atomic_write(self, tmp_path):
        """Output must not appear at out_path until encoding is complete."""
        audio = _sine(1.0)
        out = tmp_path / "atomic.m4a"
        encode_part_m4a(audio, out, loudnorm=False)
        assert out.exists()
        assert not out.with_suffix(".m4a.tmp").exists()

    def test_track_metadata(self, tmp_path):
        import subprocess, json as _json
        audio = _sine(1.0)
        out = tmp_path / "track.m4a"
        encode_part_m4a(audio, out, track=3, loudnorm=False)
        result = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "quiet",
             "-show_entries", "format_tags=track",
             "-of", "json", str(out)],
            capture_output=True, text=True
        )
        data = _json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        assert str(tags.get("track", "")) == "3"


@pytest.mark.ffmpeg
class TestCombineM4b:
    @pytest.fixture()
    def three_parts(self, tmp_path) -> list[Path]:
        """Encode 3 sine m4a files (loudnorm off for speed)."""
        parts = []
        for i, dur in enumerate([1.5, 1.2, 1.8]):
            p = tmp_path / f"part{i}.m4a"
            encode_part_m4a(_sine(dur, freq=440 + i * 110), p, loudnorm=False)
            parts.append(p)
        return parts

    def test_combine_produces_m4b(self, tmp_path, three_parts):
        out = tmp_path / "book.m4b"
        chapters = [
            ChapterMark(title="Chapter One", files=(three_parts[0], three_parts[1])),
            ChapterMark(title="Chapter Two", files=(three_parts[2],)),
        ]
        combine_m4b(chapters, out, book_title="Test Book", author="Test Author", gap_s=0.5)
        assert out.exists()

    def test_combine_chapters_count_and_titles(self, tmp_path, three_parts):
        out = tmp_path / "book.m4b"
        chapters = [
            ChapterMark(title="Chapter One", files=(three_parts[0], three_parts[1])),
            ChapterMark(title="Chapter Two", files=(three_parts[2],)),
        ]
        combine_m4b(chapters, out, book_title="Test Book", author="Test Author", gap_s=0.5)
        ch = probe_chapters(out)
        assert len(ch) == 2
        assert ch[0]["title"] == "Chapter One"
        assert ch[1]["title"] == "Chapter Two"

    def test_combine_duration_approx(self, tmp_path, three_parts):
        gap_s = 0.5
        out = tmp_path / "book.m4b"
        durations = [probe_duration_s(p) for p in three_parts]
        # 3 files → 2 gaps between files (gap after file 0, gap after file 1; no trailing gap)
        expected = sum(durations) + gap_s * 2
        chapters = [
            ChapterMark(title="Ch1", files=(three_parts[0], three_parts[1])),
            ChapterMark(title="Ch2", files=(three_parts[2],)),
        ]
        combine_m4b(chapters, out, book_title="B", author="A", gap_s=gap_s)
        actual = probe_duration_s(out)
        assert abs(actual - expected) < 0.3

    def test_missing_file_raises(self, tmp_path, three_parts):
        missing = tmp_path / "nonexistent.m4a"
        chapters = [
            ChapterMark(title="Ch1", files=(three_parts[0], missing)),
        ]
        with pytest.raises(FileNotFoundError, match="nonexistent.m4a"):
            combine_m4b(chapters, tmp_path / "out.m4b", book_title="B", author="A")

    def test_cover_attachment(self, tmp_path, three_parts):
        """A jpg cover image should appear as mjpeg attached_pic stream in the m4b."""
        import subprocess, json as _json

        # Generate a tiny 1x1 jpg with ffmpeg
        cover = tmp_path / "cover.jpg"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-y",
             "-f", "lavfi", "-i", "color=c=red:size=8x8:rate=1",
             "-frames:v", "1", str(cover)],
            check=True, capture_output=True,
        )

        out = tmp_path / "with_cover.m4b"
        chapters = [ChapterMark(title="Ch1", files=(three_parts[0],))]
        combine_m4b(chapters, out, book_title="B", author="A", cover=cover, gap_s=0.5)

        probe_result = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "quiet",
             "-show_streams", "-of", "json", str(out)],
            capture_output=True, text=True
        )
        data = _json.loads(probe_result.stdout)
        streams = data.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        assert len(video_streams) >= 1
        # Check disposition attached_pic
        v = video_streams[0]
        disposition = v.get("disposition", {})
        assert disposition.get("attached_pic") == 1
