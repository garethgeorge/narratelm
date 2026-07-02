"""Generation orchestration: window synthesis with cache/resume/retry, part assembly.

Layout under the book's output directory:

    BOOK/
      <book-slug>-007-chapter-one.pt00.m4a     final parts
      .narratelm/
        manifest.json                          settings, window records, part records
        wav/<key16>.wav                        accepted window audio (cache)

Consistency rules enforced here (plan §5): one base seed for every window,
identical conditioning per window (same reference voice), retries alone derive
new seeds, and any change to text/voice/model/cfg/ddpm invalidates exactly the
affected windows via the cache key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import audio as audio_mod
from .bookjson import Book
from .chunking import SectionPlan, plan_book
from .config import QA_MAX_RETRIES, GenerationSettings
from .naming import book_slug, part_filename
from .qa import QAResult, check_window
from .tts.base import NARRATOR, ScriptLine, SynthesisRequest, TTSEngine, VoiceRef

ROLLING_CONTEXT_TAIL_S = 15.0

ProgressFn = Callable[[str], None]


def _fmt_dur(seconds: float) -> str:
    """Compact human duration: '48s', '1m48s'."""
    s = int(round(seconds))
    return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s"


class RenderProgress:
    """Formats and emits human-readable render progress through a plain sink.

    Owns the running window counter so every synthesis pass is announced as
    ``[ 12/87  13%]`` alongside the section/part/window it belongs to — the
    context that was missing while VibeVoice's own generation bar ticked away.
    """

    # continuation lines (✓, retry, …) hang under the counter column
    _INDENT = " " * 14

    def __init__(self, emit: ProgressFn, total_windows: int, wpm: int):
        self.emit = emit
        self.total_windows = total_windows
        self.wpm = max(1, wpm)
        self.done = 0

    def note(self, msg: str) -> None:
        self.emit(msg)

    def _counter(self) -> str:
        width = len(str(self.total_windows))
        pct = 100 * self.done // self.total_windows if self.total_windows else 0
        return f"[{self.done:>{width}}/{self.total_windows} {pct:>3d}%]"

    def _mins(self, words: int) -> str:
        return f"~{words / self.wpm:.1f} min"

    def plan_summary(
        self, *, sections: int, parts: int, words: int, settings: GenerationSettings
    ) -> None:
        self.emit(
            f"plan: {sections} section(s) · {parts} part(s) · {self.total_windows} window(s) "
            f"· {words:,} words ({self._mins(words)})"
        )
        self.emit(
            f"      voice {settings.voice} · seed {settings.seed} · cfg {settings.cfg_scale} "
            f"· {settings.ddpm_steps} ddpm steps · device {settings.device}"
            + (" · rolling-context" if settings.rolling_context else "")
        )

    def part_header(
        self, *, title: str, part_index: int, total_parts: int, words: int, windows: int
    ) -> None:
        part = f" · part {part_index + 1}/{total_parts}" if total_parts > 1 else ""
        self.emit(
            f"\n── {title}{part} · {words:,} words ({self._mins(words)}) · {windows} window(s)"
        )

    def skip_part(self, filename: str, windows: int) -> None:
        self.done += windows
        self.emit(f"{self._counter()} [{filename}] up to date, skipping ({windows} window(s))")

    def cached_window(self, label: str, words: int) -> None:
        self.done += 1
        self.emit(f"{self._counter()} {label} · {words} words · cached")

    def window_start(self, label: str, words: int, seed: int) -> None:
        self.done += 1
        self.emit(
            f"{self._counter()} {label} · {words} words ({self._mins(words)}) "
            f"· seed {seed} · generating…"
        )

    def window_done(self, stats: dict, attempts: int) -> None:
        audio_s = stats.get("audio_s")
        gen_s = stats.get("generation_s")
        rtf = stats.get("rtf")
        bits = []
        if audio_s is not None:
            bits.append(f"{audio_s / 60:.1f} min audio")
        if gen_s is not None:
            bits.append(_fmt_dur(gen_s))
        if rtf is not None:
            bits.append(f"rtf {rtf}")
        if attempts > 1:
            bits.append(f"{attempts} attempts")
        if bits:
            self.emit(f"{self._INDENT}✓ " + " · ".join(bits))

    def window_retry(self, attempt: int, max_retries: int, reasons: list[str], next_seed: int) -> None:
        self.emit(
            f"{self._INDENT}⟳ QA reject (attempt {attempt + 1}/{max_retries + 1}): "
            f"{'; '.join(reasons)} — retrying with seed {next_seed}"
        )

    def window_suspect(self, attempts: int, reasons: list[str]) -> None:
        self.emit(
            f"{self._INDENT}⚠ SUSPECT — accepted best of {attempts} attempts ({'; '.join(reasons)})"
        )


@dataclass
class WindowRecord:
    key: str
    wav: str
    seed_used: int
    attempts: int
    qa: str  # "ok" | "suspect"
    qa_reasons: list[str] = field(default_factory=list)


@dataclass
class RenderReport:
    parts_rendered: list[str] = field(default_factory=list)
    parts_skipped: list[str] = field(default_factory=list)
    suspect_windows: list[str] = field(default_factory=list)
    audio_seconds: float = 0.0
    interrupted: bool = False


class Manifest:
    """Small JSON sidecar; rewritten atomically after every accepted window."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"version": 1, "settings": {}, "windows": {}, "parts": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # corrupt manifest: cache wavs are still keyed by content, so
                # starting from an empty manifest only costs bookkeeping
                pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.path)


def _voice_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def window_key(
    text: str,
    voice_hash: str,
    fingerprint: dict,
    settings: GenerationSettings,
    prev_key: str | None,
) -> str:
    """Content-addressed cache key for one window's accepted audio.

    `prev_key` chains keys when rolling context is on (a window's audio then
    depends on its predecessor's audio).
    """
    payload = json.dumps(
        {
            "text": text,
            "voice": voice_hash,
            "engine": fingerprint,
            "cfg_scale": settings.cfg_scale,
            "seed": settings.seed,
            "rolling": settings.rolling_context,
            "prev": prev_key if settings.rolling_context else None,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_seed(base_seed: int, key: str, attempt: int) -> int:
    """attempt 0 == the book's pinned seed (the consistency anchor); retries derive."""
    if attempt == 0:
        return base_seed
    digest = hashlib.sha256(f"{base_seed}:{key}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def render_book(
    book: Book,
    out_dir: Path,
    engine: TTSEngine,
    settings: GenerationSettings,
    voice_path: Path,
    *,
    section_ids: set[str] | None = None,
    limit_words: int | None = None,
    force: bool = False,
    max_retries: int = QA_MAX_RETRIES,
    progress: ProgressFn = lambda msg: None,
) -> RenderReport:
    plans = plan_book(
        book,
        window_words=settings.window_words,
        max_part_minutes=settings.max_part_minutes,
        wpm=settings.wpm,
        section_filter=section_ids,
        limit_words=limit_words,
    )
    slug = book_slug(book.metadata.title or Path(book.source_epub).stem)

    cache_dir = out_dir / ".narratelm"
    wav_dir = cache_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(cache_dir / "manifest.json")
    manifest.data["settings"] = settings.__dict__ | {"voice_file": str(voice_path)}

    voice_hash = _voice_file_hash(voice_path)
    fingerprint = engine.fingerprint()
    report = RenderReport()

    total_windows = sum(len(part.windows) for plan in plans for part in plan.parts)
    total_parts = sum(len(plan.parts) for plan in plans)
    total_words = sum(plan.word_count for plan in plans)
    prog = RenderProgress(progress, total_windows, settings.wpm)
    prog.plan_summary(
        sections=len(plans), parts=total_parts, words=total_words, settings=settings
    )

    try:
        for plan in plans:
            _render_section(
                plan, book, out_dir, wav_dir, engine, settings, voice_path, voice_hash,
                fingerprint, manifest, slug, force, max_retries, report, prog,
            )
    except KeyboardInterrupt:
        manifest.save()
        report.interrupted = True
        prog.note("interrupted — progress saved; re-run the same command to resume")
    return report


def _render_section(
    plan: SectionPlan,
    book: Book,
    out_dir: Path,
    wav_dir: Path,
    engine: TTSEngine,
    settings: GenerationSettings,
    voice_path: Path,
    voice_hash: str,
    fingerprint: dict,
    manifest: Manifest,
    slug: str,
    force: bool,
    max_retries: int,
    report: RenderReport,
    prog: RenderProgress,
) -> None:
    total_parts = len(plan.parts)
    section_title = plan.title or plan.section_id
    prev_key: str | None = None
    context_tail: np.ndarray | None = None
    track_base = plan.spine_index * 100  # unique, ordered track numbers across the book

    for part in plan.parts:
        filename = part_filename(slug, plan.spine_index, section_title, part.index, total_parts)
        out_path = out_dir / filename

        # rolling context resets at part boundaries (natural prosody reset)
        if settings.rolling_context:
            context_tail = None

        keys = []
        k = prev_key
        for window in part.windows:
            k = window_key(window.text, voice_hash, fingerprint, settings, k)
            keys.append(k)
        prev_key = keys[-1] if keys else prev_key

        part_rec = manifest.data["parts"].get(filename)
        if (
            not force
            and out_path.exists()
            and part_rec
            and part_rec.get("window_keys") == keys
            and part_rec.get("complete")
        ):
            report.parts_skipped.append(filename)
            prog.skip_part(filename, len(part.windows))
            continue

        prog.part_header(
            title=section_title, part_index=part.index, total_parts=total_parts,
            words=part.word_count, windows=len(part.windows),
        )

        chunks: list[np.ndarray] = []
        pauses: list[float] = []
        for window, key in zip(part.windows, keys):
            label = f"{section_title} · pt{part.index + 1}/{total_parts} · w{window.index + 1:03d}"
            wav_path = wav_dir / f"{key[:16]}.wav"
            if wav_path.exists() and not force:
                chunk, _ = sf.read(wav_path, dtype="float32", always_2d=False)
                prog.cached_window(label, window.word_count)
            else:
                chunk = _synthesize_window(
                    window.text, window.word_count, key, engine, settings,
                    voice_path, context_tail, max_retries, manifest, wav_path,
                    label, report, prog,
                )
            chunks.append(chunk)
            pauses.append(window.pause_after_s)
            if settings.rolling_context:
                tail = int(ROLLING_CONTEXT_TAIL_S * engine.sample_rate)
                context_tail = chunk[-tail:] if len(chunk) > tail else chunk

        part_audio = audio_mod.assemble_windows(chunks, pauses, sample_rate=engine.sample_rate)
        title = section_title
        if total_parts > 1:
            title = f"{title} ({part.index + 1}/{total_parts})"
        prog.note(f"{prog._INDENT}encoding {filename} ({len(part_audio) / engine.sample_rate / 60:.1f} min)")
        audio_mod.encode_part_m4a(
            part_audio, out_path,
            sample_rate=engine.sample_rate,
            title=title,
            track=track_base + part.index,
        )
        manifest.data["parts"][filename] = {
            "section_id": plan.section_id,
            "spine_index": plan.spine_index,
            "title": plan.title or plan.section_id,
            "part_index": part.index,
            "total_parts": total_parts,
            "window_keys": keys,
            "complete": True,
        }
        manifest.save()
        report.parts_rendered.append(filename)
        report.audio_seconds += len(part_audio) / engine.sample_rate


def _synthesize_window(
    text: str,
    word_count: int,
    key: str,
    engine: TTSEngine,
    settings: GenerationSettings,
    voice_path: Path,
    context_tail: np.ndarray | None,
    max_retries: int,
    manifest: Manifest,
    wav_path: Path,
    label: str,
    report: RenderReport,
    prog: RenderProgress,
) -> np.ndarray:
    voice = VoiceRef(
        name=settings.voice,
        wav_path=voice_path,
        context_audio=context_tail if settings.rolling_context else None,
    )
    best: tuple[QAResult, np.ndarray, int] | None = None

    prog.window_start(label, word_count, settings.seed)
    for attempt in range(max_retries + 1):
        seed = derive_seed(settings.seed, key, attempt)
        request = SynthesisRequest(
            lines=(ScriptLine(NARRATOR, text),),
            voices={NARRATOR: voice},
            seed=seed,
            cfg_scale=settings.cfg_scale,
            ddpm_steps=settings.ddpm_steps,
        )
        result = engine.synthesize(request)
        qa = check_window(result.audio, engine.sample_rate, word_count, settings.wpm)
        if qa.ok:
            _accept(result.audio, engine.sample_rate, wav_path, manifest, key, seed, attempt, qa)
            prog.window_done(result.stats, attempts=attempt + 1)
            return result.audio
        if attempt < max_retries:
            prog.window_retry(attempt, max_retries, qa.reasons, derive_seed(settings.seed, key, attempt + 1))
        if best is None or qa.score < best[0].score:
            best = (qa, result.audio, seed)

    assert best is not None
    qa, audio, seed = best
    _accept(audio, engine.sample_rate, wav_path, manifest, key, seed, max_retries, qa, suspect=True)
    report.suspect_windows.append(label)
    prog.window_suspect(max_retries + 1, qa.reasons)
    return audio


def _accept(
    audio: np.ndarray,
    sample_rate: int,
    wav_path: Path,
    manifest: Manifest,
    key: str,
    seed: int,
    attempts: int,
    qa: QAResult,
    *,
    suspect: bool = False,
) -> None:
    tmp = wav_path.with_suffix(".wav.tmp")
    sf.write(tmp, audio, sample_rate, format="WAV")
    tmp.replace(wav_path)
    manifest.data["windows"][key] = WindowRecord(
        key=key,
        wav=wav_path.name,
        seed_used=seed,
        attempts=attempts + 1,
        qa="suspect" if suspect else "ok",
        qa_reasons=qa.reasons,
    ).__dict__
    manifest.save()
