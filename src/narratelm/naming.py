"""Filename generation for audiobook output files."""

from __future__ import annotations

import re
import unicodedata

from narratelm.chunking import SectionPlan


def slugify(text: str, max_len: int = 40) -> str:
    """Lowercase, ASCII-fold (strip diacritics), collapse non-alphanumeric to '-'.

    Trims leading/trailing dashes and truncates at max_len without a trailing dash.
    Empty or whitespace-only input → 'section'.
    """
    if not text or not text.strip():
        return "section"

    # NFKD normalization + strip combining characters (diacritics)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in normalized if not unicodedata.combining(c))

    # Lowercase
    ascii_text = ascii_text.lower()

    # Replace non-alphanumeric runs with a single dash
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text)

    # Trim leading/trailing dashes
    slug = slug.strip("-")

    if not slug:
        return "section"

    # Truncate at max_len without trailing dash
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")

    return slug or "section"


def book_slug(book_title: str) -> str:
    """Slugify a book title with max_len=60."""
    return slugify(book_title, max_len=60)


def part_filename(
    book_slug_str: str,
    spine_index: int,
    section_title_or_id: str,
    part_index: int,
    total_parts: int,
) -> str:
    """Return the .m4a filename for one part.

    Single-part sections: ``{book_slug}-{spine_index:03d}-{slug}.m4a``
    Multi-part sections:  ``{book_slug}-{spine_index:03d}-{slug}.pt{part_index:02d}.m4a``

    Raises ValueError if part_index > 99 (2-digit pt field only supports ≤100 parts).
    """
    if part_index > 99:
        raise ValueError(
            f"part_index {part_index} exceeds maximum of 99 (2-digit pt field)"
        )

    section_slug = slugify(section_title_or_id)
    base = f"{book_slug_str}-{spine_index:03d}-{section_slug}"

    if total_parts == 1:
        return f"{base}.m4a"
    return f"{base}.pt{part_index:02d}.m4a"


def section_filenames(plan: SectionPlan, book_slug_str: str) -> list[str]:
    """Return the ordered list of filenames for all parts of a section."""
    total = len(plan.parts)
    return [
        part_filename(
            book_slug_str,
            plan.spine_index,
            plan.title or plan.section_id,
            part.index,
            total,
        )
        for part in plan.parts
    ]
