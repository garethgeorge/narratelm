"""Drive the epub -> Book extraction: walk the spine, normalise, classify.

Produces the editable book JSON. Every linear spine item becomes a Section
(including skipped front/back matter) so a human can flip ``skip`` flags without
re-running extraction.
"""

from __future__ import annotations

from pathlib import Path

from narratelm.bookjson import Book, Section
from narratelm.classify import classify
from narratelm.config import DEFAULT_WPM
from narratelm.epub import Epub
from narratelm.textnorm import count_words, leading_title, normalize_document


def extract_book(epub_path: Path, *, wpm: int = DEFAULT_WPM) -> Book:
    """Extract *epub_path* into a fully-populated :class:`Book`."""
    epub = Epub.open(epub_path)
    try:
        sections: list[Section] = []
        for item in epub.spine:
            xhtml = epub.read_document(item.href)
            title_hint = epub.title_for(item.href)
            paragraphs = normalize_document(xhtml, title_hint=title_hint)
            words = count_words(paragraphs)
            display_title = title_hint or leading_title(xhtml) or item.id
            ctype, skip, reason = classify(
                item.id,
                item.href,
                display_title,
                paragraphs,
                xhtml,
                is_nav=item.href == epub.nav_href,
            )
            sections.append(
                Section(
                    id=item.id,
                    index=item.index,
                    source_file=item.href,
                    title=display_title,
                    type=ctype,
                    skip=skip,
                    skip_reason=reason,
                    word_count=words,
                    est_minutes=round(words / wpm, 1),
                    paragraphs=paragraphs,
                )
            )
    finally:
        epub.close()

    return Book(
        source_epub=epub_path.name,
        metadata=epub.metadata,
        sections=sections,
    )


def format_section_table(book: Book) -> str:
    """A fixed-width overview table (one row per section) for CLI output."""
    header = ("idx", "id", "type", "words", "est_min", "skip", "title")
    rows = [header]
    for s in book.sections:
        rows.append(
            (
                str(s.index),
                s.id,
                s.type,
                str(s.word_count),
                f"{s.est_minutes:.1f}",
                "skip" if s.skip else "",
                s.title,
            )
        )
    widths = [max(len(r[c]) for r in rows) for c in range(len(header))]
    lines = []
    for r in rows:
        lines.append("  ".join(cell.ljust(widths[c]) for c, cell in enumerate(r)).rstrip())
    return "\n".join(lines)
