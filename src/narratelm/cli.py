"""narratelm command-line interface (thin; delegates to the library modules)."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import config
from .bookjson import Book, BookJsonError

# ---------------------------------------------------------------- helpers


def _load_or_extract(book_arg: Path, wpm: int) -> tuple[Book, Path]:
    """Accept BOOK.json or BOOK.epub; auto-extract when the JSON is missing."""
    if book_arg.suffix.lower() == ".json":
        return Book.load(book_arg), book_arg
    json_path = config.json_path_for(book_arg)
    if json_path.exists():
        return Book.load(json_path), json_path
    from .extract import extract_book

    click.echo(f"no {json_path.name} yet — extracting from {book_arg.name}")
    book = extract_book(book_arg, wpm=wpm)
    book.save(json_path, stamp_extractor_hash=True)
    return book, json_path


def _parse_sections(spec: str | None, book: Book) -> set[str] | None:
    """--sections "1,3-5,interlude2": ints are spine indexes, tokens are section ids."""
    if not spec:
        return None
    by_index = {s.index: s.id for s in book.sections}
    out: set[str] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token and all(p.strip().isdigit() for p in token.split("-", 1)):
            lo, hi = (int(p) for p in token.split("-", 1))
            for i in range(lo, hi + 1):
                if i in by_index:
                    out.add(by_index[i])
        elif token.isdigit():
            idx = int(token)
            if idx not in by_index:
                raise click.BadParameter(f"no section at spine index {idx}")
            out.add(by_index[idx])
        else:
            if book.section_by_id(token) is None:
                ids = ", ".join(s.id for s in book.sections[:12])
                raise click.BadParameter(f"unknown section id {token!r} (ids start: {ids} ...)")
            out.add(token)
    return out or None


def _make_engine(
    engine_name: str, spec: config.ModelSpec, device: str, ddpm_steps: int,
    cache_dir: Path | None,
):
    if engine_name == "fake":
        from .tts.fake import FakeEngine

        return FakeEngine()
    from .tts.vibevoice import VibeVoiceEngine

    return VibeVoiceEngine(
        spec.repo, subfolder=spec.subfolder, device=device, ddpm_steps=ddpm_steps,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def _generation_options(f):
    opts = [
        click.option("--voice", default=config.DEFAULT_VOICE, show_default=True,
                     help="bundled voice name or path to reference audio"),
        click.option("--model", default=config.DEFAULT_MODEL, show_default=True,
                     help="alias (1.5b, 7b, 7b-4bit, 7b-8bit), HF repo id, or local path"),
        click.option("--model-subfolder", default=None,
                     help="repo subfolder holding the weights (e.g. 4bit); overrides the alias default"),
        click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
                     help="where model weights are downloaded (default: $HF_HOME or ~/.cache/huggingface)"),
        click.option("--device", default="auto", show_default=True,
                     type=click.Choice(["auto", "mps", "cuda", "cpu"])),
        click.option("--cfg-scale", default=config.DEFAULT_CFG_SCALE, show_default=True),
        click.option("--ddpm-steps", default=config.DEFAULT_DDPM_STEPS, show_default=True),
        click.option("--seed", default=config.DEFAULT_SEED, show_default=True,
                     help="pinned for every window; audition candidates with `narratelm probe`"),
        click.option("--wpm", default=config.DEFAULT_WPM, show_default=True,
                     help="words/min estimate used for QA and part sizing"),
        click.option("--engine", default="vibevoice", type=click.Choice(["vibevoice", "fake"]),
                     hidden=True),
    ]
    for opt in reversed(opts):
        f = opt(f)
    return f


# ---------------------------------------------------------------- commands


@click.group()
@click.version_option(package_name="narratelm")
def main() -> None:
    """Narrate epub books into chapterized audiobooks locally with VibeVoice."""


@main.command()
@click.argument("epub", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--force", is_flag=True, help="overwrite an existing (even hand-edited) JSON")
@click.option("--wpm", default=config.DEFAULT_WPM, show_default=True)
def extract(epub: Path, force: bool, wpm: int) -> None:
    """Extract EPUB into an editable BOOK.json next to it."""
    from .extract import extract_book, format_section_table

    json_path = config.json_path_for(epub)
    if json_path.exists() and not force:
        try:
            existing = Book.load(json_path)
        except BookJsonError:
            existing = None
        if existing is not None and existing.is_user_edited():
            raise click.ClickException(
                f"{json_path.name} has hand edits; re-run with --force to overwrite them"
            )
    book = extract_book(epub, wpm=wpm)
    book.save(json_path, stamp_extractor_hash=True)
    click.echo(format_section_table(book))
    click.echo(f"\nwrote {json_path}")
    click.echo("review the table above; flip \"skip\" in the JSON to include/exclude sections")


@main.command()
@click.argument("book_path", metavar="BOOK.json|BOOK.epub",
                type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_generation_options
@click.option("--window-words", default=config.DEFAULT_WINDOW_WORDS, show_default=True,
              help="words per TTS call (raise only after a clean probe)")
@click.option("--max-part-minutes", default=config.DEFAULT_MAX_PART_MINUTES, show_default=True)
@click.option("--rolling-context", is_flag=True,
              help="EXPERIMENTAL: condition each window on the tail of the previous one")
@click.option("--sections", "sections_spec", default=None,
              help='e.g. "7,9-12,interlude2" (spine indexes and/or section ids)')
@click.option("--limit-words", type=int, default=None,
              help="smoke test: narrate only the first N words of each selected section")
@click.option("--force", is_flag=True, help="re-render even when cache/outputs are up to date")
def generate(
    book_path: Path,
    voice: str,
    model: str,
    model_subfolder: str | None,
    cache_dir: Path | None,
    device: str,
    cfg_scale: float,
    ddpm_steps: int,
    seed: int,
    wpm: int,
    engine: str,
    window_words: int,
    max_part_minutes: float,
    rolling_context: bool,
    sections_spec: str | None,
    limit_words: int | None,
    force: bool,
) -> None:
    """Render narration audio into the BOOK/ directory (one m4a per chapter part)."""
    from .pipeline import render_book
    from .voices import resolve_voice

    book, json_path = _load_or_extract(book_path, wpm)
    section_ids = _parse_sections(sections_spec, book)
    voice_name, voice_path = resolve_voice(voice)
    spec = config.resolve_model(model, model_subfolder)
    settings = config.GenerationSettings(
        model=spec.repo, model_subfolder=spec.subfolder, voice=voice_name, device=device,
        cfg_scale=cfg_scale, ddpm_steps=ddpm_steps, seed=seed, window_words=window_words,
        max_part_minutes=max_part_minutes, wpm=wpm, rolling_context=rolling_context,
    )
    out_dir = config.output_dir_for(json_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts_engine = _make_engine(engine, spec, device, ddpm_steps, cache_dir)
    report = render_book(
        book, out_dir, tts_engine, settings, voice_path,
        section_ids=section_ids, limit_words=limit_words, force=force,
        progress=lambda msg: click.echo(msg),
    )

    click.echo(
        f"\nrendered {len(report.parts_rendered)} part(s), skipped {len(report.parts_skipped)} up-to-date, "
        f"{report.audio_seconds / 60:.1f} min of new audio"
    )
    if report.suspect_windows:
        click.echo(f"⚠ {len(report.suspect_windows)} suspect window(s) — spot-listen these:")
        for label in report.suspect_windows:
            click.echo(f"  {label}")
    if report.interrupted:
        sys.exit(130)


@main.command()
@click.argument("book_path", metavar="BOOK.json|BOOK.epub",
                type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_generation_options
@click.option("--seeds", default="42,7,1234", show_default=True,
              help="comma-separated candidate seeds")
@click.option("--window-sizes", default="2000", show_default=True,
              help="comma-separated window sizes (words) to test stability at")
def probe(
    book_path: Path,
    voice: str,
    model: str,
    model_subfolder: str | None,
    cache_dir: Path | None,
    device: str,
    cfg_scale: float,
    ddpm_steps: int,
    seed: int,
    wpm: int,
    engine: str,
    seeds: str,
    window_sizes: str,
) -> None:
    """Audition seeds/window sizes on a real excerpt; pick the winner by ear."""
    from .probe import format_probe_table, run_probe
    from .voices import resolve_voice

    book, json_path = _load_or_extract(book_path, wpm)
    voice_name, voice_path = resolve_voice(voice)
    spec = config.resolve_model(model, model_subfolder)
    settings = config.GenerationSettings(
        model=spec.repo, model_subfolder=spec.subfolder, voice=voice_name, device=device,
        cfg_scale=cfg_scale, ddpm_steps=ddpm_steps, seed=seed, wpm=wpm,
    )
    out_dir = config.output_dir_for(json_path)
    rows = run_probe(
        book, out_dir, _make_engine(engine, spec, device, ddpm_steps, cache_dir), settings, voice_path,
        seeds=[int(s) for s in seeds.split(",") if s.strip()],
        window_sizes=[int(s) for s in window_sizes.split(",") if s.strip()],
        progress=lambda msg: click.echo(msg),
    )
    click.echo("\n" + format_probe_table(rows))


@main.command()
@click.argument("book_path", metavar="BOOK.json|BOOK.epub",
                type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None,
              help="output .m4b path (default: BOOK.m4b next to the epub)")
@click.option("--partial", is_flag=True, help="tolerate missing sections (smoke tests)")
def combine(book_path: Path, output: Path | None, partial: bool) -> None:
    """Merge rendered parts into one chapterized .m4b audiobook."""
    from .audio import ChapterMark, combine_m4b
    from .pipeline import Manifest

    book, json_path = _load_or_extract(book_path, config.DEFAULT_WPM)
    out_dir = config.output_dir_for(json_path)
    manifest = Manifest(out_dir / ".narratelm" / "manifest.json")
    parts: dict = manifest.data.get("parts", {})
    if not parts:
        raise click.ClickException(f"nothing rendered yet in {out_dir} — run `narratelm generate` first")

    # group manifest parts by section, in spine order then part order
    by_section: dict[str, list[tuple[int, str, dict]]] = {}
    for filename, rec in parts.items():
        by_section.setdefault(rec["section_id"], []).append((rec["part_index"], filename, rec))

    chapters: list[ChapterMark] = []
    missing_sections: list[str] = []
    for section in book.narrated_sections():
        recs = by_section.get(section.id)
        if not recs:
            missing_sections.append(section.id)
            continue
        recs.sort()
        files = tuple(out_dir / filename for _, filename, _ in recs)
        absent = [f for f in files if not f.exists()]
        if absent:
            missing_sections.append(section.id)
            continue
        chapters.append(ChapterMark(title=section.title or section.id, files=files))

    if missing_sections:
        msg = f"{len(missing_sections)} narrated section(s) have no rendered audio: " + ", ".join(missing_sections[:8])
        if not partial:
            raise click.ClickException(msg + "  (use --partial to combine anyway)")
        click.echo("warning: " + msg)
    if not chapters:
        raise click.ClickException("no complete sections to combine")

    cover = _extract_cover(book, json_path, out_dir)
    out_path = output or json_path.parent / (json_path.stem + ".m4b")
    combine_m4b(chapters, out_path, book_title=book.metadata.title,
                author=book.metadata.author, cover=cover)
    click.echo(f"wrote {out_path} ({len(chapters)} chapters)")


def _extract_cover(book: Book, json_path: Path, out_dir: Path) -> Path | None:
    if not book.metadata.cover_href:
        return None
    epub_path = json_path.parent / book.source_epub
    if not epub_path.exists():
        return None
    from .epub import Epub

    try:
        data = Epub.open(epub_path).read_bytes(book.metadata.cover_href)
    except Exception:  # cover is nice-to-have, never fatal
        return None
    suffix = Path(book.metadata.cover_href).suffix or ".jpg"
    cover = out_dir / (".narratelm/cover" + suffix)
    cover.parent.mkdir(parents=True, exist_ok=True)
    cover.write_bytes(data)
    return cover


@main.group()
def voices() -> None:
    """Voice management."""


@voices.command("list")
def voices_list() -> None:
    """List bundled narrator voices."""
    from .voices import bundled_voices

    click.echo(f"{'name':<12} {'gender':<8} {'length':<8} notes")
    click.echo("-" * 60)
    for v in bundled_voices():
        click.echo(f"{v.name:<12} {v.gender:<8} {v.duration_s:>5.1f}s  {v.notes or ''}")
    click.echo("\nuse --voice <name>, or --voice /path/to/30-60s-clean-speech.wav to clone")


@main.command()
@click.option("--download-model", is_flag=True, help="prefetch the model into the cache")
@click.option("--model", default=config.DEFAULT_MODEL, show_default=True,
              help="alias (1.5b, 7b, 7b-4bit, 7b-8bit), HF repo id, or local path")
@click.option("--model-subfolder", default=None,
              help="repo subfolder to fetch (e.g. 4bit); overrides the alias default")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="where model weights are downloaded (default: $HF_HOME or ~/.cache/huggingface)")
def setup(download_model: bool, model: str, model_subfolder: str | None,
          cache_dir: Path | None) -> None:
    """Check the environment (ffmpeg, torch, device) and optionally prefetch the model."""
    import shutil as _shutil

    ok = True
    for tool in ("ffmpeg", "ffprobe"):
        found = _shutil.which(tool)
        click.echo(f"{tool}: {found or 'MISSING — enter the nix devshell'}")
        ok = ok and bool(found)

    try:
        from .tts.devices import pick_device

        plan = pick_device("auto")
        click.echo(f"torch: ok — device {plan.device} ({plan.dtype_name}, {plan.attn_implementation})")
        if plan.warning:
            click.echo(f"  note: {plan.warning}")
    except ImportError as e:
        ok = False
        click.echo(f"torch: MISSING ({e}) — run `just setup` (uv sync) to install ML deps")

    try:
        import vibevoice  # noqa: F401

        click.echo("vibevoice: ok")
    except ImportError:
        ok = False
        click.echo("vibevoice: MISSING — run `just setup` (uv sync)")

    if download_model:
        from huggingface_hub import snapshot_download

        spec = config.resolve_model(model, model_subfolder)
        patterns = [f"{spec.subfolder}/*"] if spec.subfolder else None
        label = spec.repo + (f" [{spec.subfolder}]" if spec.subfolder else "")
        click.echo(f"downloading {label} ...")
        path = snapshot_download(
            spec.repo, allow_patterns=patterns,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        click.echo(f"model cached at {path}")

    if not ok:
        sys.exit(1)


@main.command()
@click.argument("epub", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
@_generation_options
def run(ctx: click.Context, epub: Path, **gen_kwargs) -> None:
    """extract (if needed) → generate → combine, end to end."""
    ctx.invoke(generate, book_path=epub, window_words=config.DEFAULT_WINDOW_WORDS,
               max_part_minutes=config.DEFAULT_MAX_PART_MINUTES, rolling_context=False,
               sections_spec=None, limit_words=None, force=False, **gen_kwargs)
    ctx.invoke(combine, book_path=epub, output=None, partial=False)


if __name__ == "__main__":
    main()
