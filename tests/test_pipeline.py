from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from narratelm.bookjson import Book, Metadata, Section
from narratelm.config import GenerationSettings
from narratelm.pipeline import Manifest, derive_seed, render_book, window_key
from narratelm.tts.fake import FakeEngine

WORDS_PER_PARA = 60


def _para(i: int, n: int = WORDS_PER_PARA) -> str:
    words = [f"w{i}x{j}" for j in range(n - 1)]
    return " ".join(words) + f" end{i}."


def make_book(n_paras: int = 12) -> Book:
    paragraphs = [_para(i) for i in range(n_paras)]
    words = sum(len(p.split()) for p in paragraphs)
    return Book(
        source_epub="fake.epub",
        metadata=Metadata(title="Test Book", author="Nobody"),
        sections=[
            Section(
                id="chapter1", index=3, source_file="c1.xhtml", title="Chapter One",
                type="chapter", skip=False, skip_reason=None,
                word_count=words, est_minutes=words / 150, paragraphs=paragraphs,
            ),
            Section(
                id="glossary", index=4, source_file="g.xhtml", title="Glossary",
                type="glossary", skip=True, skip_reason="glossary", word_count=100,
                est_minutes=0.7, paragraphs=[_para(99)],
            ),
        ],
    )


@pytest.fixture
def settings() -> GenerationSettings:
    return GenerationSettings(window_words=100, max_part_minutes=2.0, wpm=150)


@pytest.fixture
def voice_wav(tmp_path: Path) -> Path:
    import soundfile as sf

    path = tmp_path / "voice.wav"
    t = np.arange(24000 * 2, dtype=np.float32) / 24000
    sf.write(path, (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), 24000)
    return path


@pytest.mark.ffmpeg
def test_render_resume_and_invalidate(tmp_path: Path, settings, voice_wav):
    book = make_book()
    out = tmp_path / "out"
    out.mkdir()

    engine = FakeEngine()
    report = render_book(book, out, engine, settings, voice_wav)
    assert report.parts_rendered, "should render at least one part"
    assert not report.parts_skipped
    assert not report.suspect_windows
    m4as = sorted(p.name for p in out.glob("*.m4a"))
    assert m4as == sorted(report.parts_rendered)
    # skipped section produced nothing
    assert not any("glossary" in name for name in m4as)
    first_calls = len(engine.calls)

    # resume: everything cached, zero synth calls
    engine2 = FakeEngine()
    report2 = render_book(book, out, engine2, settings, voice_wav)
    assert not report2.parts_rendered
    assert len(report2.parts_skipped) == len(report.parts_rendered)
    assert len(engine2.calls) == 0

    # editing ONE paragraph invalidates only the windows containing it
    book.sections[0].paragraphs[0] = _para(0).replace("end0.", "changed0.")
    engine3 = FakeEngine()
    report3 = render_book(book, out, engine3, settings, voice_wav)
    assert report3.parts_rendered  # affected part re-encoded
    assert 0 < len(engine3.calls) < first_calls, (
        f"edit should re-synthesize a subset, got {len(engine3.calls)} of {first_calls}"
    )


@pytest.mark.ffmpeg
def test_retry_and_suspect(tmp_path: Path, settings, voice_wav):
    book = make_book(4)
    out = tmp_path / "out"
    out.mkdir()

    # first call for a given text returns garbage (silence), retries succeed
    seen: dict[str, int] = {}

    def misbehave(request, call_index):
        text = request.lines[0].text
        seen[text] = seen.get(text, 0) + 1
        if seen[text] == 1:
            return np.zeros(24000, dtype=np.float32)  # silent → QA reject
        return None

    engine = FakeEngine(misbehave=misbehave)
    report = render_book(book, out, engine, settings, voice_wav)
    assert report.parts_rendered
    assert not report.suspect_windows, "retry should have recovered"
    # each unique window text was attempted at least twice
    assert all(count >= 2 for count in seen.values())

    # always-garbage engine → accepted best effort, flagged suspect
    out2 = tmp_path / "out2"
    out2.mkdir()
    engine_bad = FakeEngine(misbehave=lambda req, i: np.zeros(24000, dtype=np.float32))
    report_bad = render_book(book, out2, engine_bad, settings, voice_wav, max_retries=1)
    assert report_bad.suspect_windows
    manifest = Manifest(out2 / ".narratelm" / "manifest.json")
    assert any(w["qa"] == "suspect" for w in manifest.data["windows"].values())


@pytest.mark.ffmpeg
def test_interrupt_saves_progress(tmp_path: Path, settings, voice_wav):
    book = make_book()
    out = tmp_path / "out"
    out.mkdir()

    calls = {"n": 0}

    def misbehave(request, call_index):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return None

    engine = FakeEngine(misbehave=misbehave)
    report = render_book(book, out, engine, settings, voice_wav)
    assert report.interrupted
    # no partial tmp files left
    assert not list(out.rglob("*.tmp"))

    # resume completes and reuses the two accepted windows
    engine2 = FakeEngine()
    report2 = render_book(book, out, engine2, settings, voice_wav)
    assert not report2.interrupted
    assert report2.parts_rendered
    total_windows = calls["n"] - 1 + len(engine2.calls)
    assert len(engine2.calls) == total_windows - 2


def test_window_key_sensitivity(settings):
    fp = {"engine": "fake"}
    base = window_key("hello world", "abc", fp, settings, None)
    assert window_key("hello world", "abc", fp, settings, None) == base
    assert window_key("hello there", "abc", fp, settings, None) != base
    assert window_key("hello world", "def", fp, settings, None) != base
    assert window_key("hello world", "abc", {"engine": "other"}, settings, None) != base
    other = GenerationSettings(**{**settings.__dict__, "cfg_scale": 2.0})
    assert window_key("hello world", "abc", fp, other, None) != base
    # prev key only matters when rolling context is on
    assert window_key("hello world", "abc", fp, settings, "prev") == base
    rolling = GenerationSettings(**{**settings.__dict__, "rolling_context": True})
    assert window_key("hello world", "abc", fp, rolling, "p1") != window_key(
        "hello world", "abc", fp, rolling, "p2"
    )


def test_derive_seed():
    assert derive_seed(42, "k", 0) == 42
    a1 = derive_seed(42, "k", 1)
    assert a1 != 42
    assert derive_seed(42, "k", 1) == a1  # deterministic
    assert derive_seed(42, "k", 2) != a1
    assert derive_seed(42, "other", 1) != a1
