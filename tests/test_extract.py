from __future__ import annotations

from pathlib import Path

import pytest

from narratelm.bookjson import Book
from narratelm.extract import extract_book, format_section_table


@pytest.fixture(scope="module")
def book(test_epub: Path) -> Book:
    return extract_book(test_epub)


def test_book_shape(book: Book) -> None:
    assert book.metadata.title == "The Wonderful Wizard of Oz"
    assert len(book.sections) == 28


def test_indexes_are_spine_positions(book: Book) -> None:
    indexes = [s.index for s in book.sections]
    assert indexes == sorted(indexes)
    assert indexes == list(range(len(book.sections)))  # strictly increasing, contiguous


def test_all_linear_spine_items_present(book: Book) -> None:
    ids = {s.id for s in book.sections}
    # Cover wrapper, PG header/footer boilerplate, and story chapters all become
    # sections (skipped or not) so a human can flip flags without re-extracting.
    for expected in ("coverpage-wrapper", "pg-header", "item4", "item6", "item28", "pg-footer"):
        assert expected in ids


def test_chapter_metrics(book: Book) -> None:
    ch2 = book.section_by_id("item6")  # "Chapter II The Council with the Munchkins"
    assert ch2 is not None
    assert ch2.word_count > 1500
    assert ch2.est_minutes == pytest.approx(round(ch2.word_count / 150, 1))
    assert 0 < ch2.est_minutes < 120


def test_roundtrip_and_edit_detection(book: Book, tmp_path: Path) -> None:
    path = tmp_path / "book.json"
    book.save(path, stamp_extractor_hash=True)

    loaded = Book.load(path)
    assert loaded.is_user_edited() is False

    target = next(s for s in loaded.sections if s.paragraphs)
    target.paragraphs[0] += " (edited)"
    assert loaded.is_user_edited() is True


def test_format_section_table(book: Book) -> None:
    table = format_section_table(book)
    lines = table.splitlines()
    assert lines[0].split()[:3] == ["idx", "id", "type"]
    assert len(lines) == len(book.sections) + 1  # header + one row per section
