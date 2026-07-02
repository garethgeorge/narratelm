"""`narratelm probe` — render the same real excerpt at candidate seeds/window
sizes so the user can pick a known-good seed by ear before committing a book.

Seed choice is the dominant consistency variable for VibeVoice (community
finding: some seeds are consistently good, others bad), so audition first,
then pin the winner via `generate --seed N`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from .bookjson import Book
from .chunking import plan_section
from .config import GenerationSettings
from .pipeline import ProgressFn
from .qa import check_window
from .tts.base import NARRATOR, ScriptLine, SynthesisRequest, TTSEngine, VoiceRef


@dataclass
class ProbeRow:
    seed: int
    window_words: int
    wav: Path
    ok: bool
    reasons: list[str]
    audio_s: float
    rtf: float | None


def run_probe(
    book: Book,
    out_dir: Path,
    engine: TTSEngine,
    settings: GenerationSettings,
    voice_path: Path,
    *,
    seeds: list[int],
    window_sizes: list[int],
    progress: ProgressFn = lambda msg: None,
) -> list[ProbeRow]:
    section = _pick_excerpt_section(book)
    probe_dir = out_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ProbeRow] = []
    for size in window_sizes:
        plan = plan_section(
            section,
            window_words=size,
            max_part_minutes=settings.max_part_minutes,
            wpm=settings.wpm,
            limit_words=size,
        )
        window = plan.parts[0].windows[0]
        for seed in seeds:
            label = f"seed{seed}-w{size}"
            progress(f"[probe {label}] generating ~{window.word_count} words from '{section.title or section.id}'")
            request = SynthesisRequest(
                lines=(ScriptLine(NARRATOR, window.text),),
                voices={NARRATOR: VoiceRef(name=settings.voice, wav_path=voice_path)},
                seed=seed,
                cfg_scale=settings.cfg_scale,
                ddpm_steps=settings.ddpm_steps,
            )
            result = engine.synthesize(request)
            qa = check_window(result.audio, engine.sample_rate, window.word_count, settings.wpm)
            wav = probe_dir / f"{label}.wav"
            sf.write(wav, result.audio, engine.sample_rate)
            rows.append(
                ProbeRow(
                    seed=seed,
                    window_words=size,
                    wav=wav,
                    ok=qa.ok,
                    reasons=qa.reasons,
                    audio_s=len(result.audio) / engine.sample_rate,
                    rtf=result.stats.get("rtf"),
                )
            )
    return rows


def _pick_excerpt_section(book: Book):
    for s in book.sections:
        if not s.skip and s.type == "chapter" and s.word_count > 800:
            return s
    for s in book.sections:
        if not s.skip and s.word_count > 100:
            return s
    raise ValueError("no narratable section found to probe")


def format_probe_table(rows: list[ProbeRow]) -> str:
    lines = [
        "seed    words  audio_s  rtf    qa       file",
        "-" * 78,
    ]
    for r in rows:
        rtf = f"{r.rtf:.2f}" if r.rtf is not None else "-"
        qa = "ok" if r.ok else "SUSPECT"
        lines.append(f"{r.seed:<7} {r.window_words:<6} {r.audio_s:<8.1f} {rtf:<6} {qa:<8} {r.wav}")
        for reason in r.reasons:
            lines.append(f"        ^ {reason}")
    lines.append("")
    lines.append("Listen to the wavs and pin the winner: narratelm generate ... --seed <best>")
    return "\n".join(lines)
