"""Tests for src/narratelm/naming.py"""

from __future__ import annotations

import pytest

from narratelm.bookjson import Section
from narratelm.chunking import SectionPlan, plan_section
from narratelm.naming import book_slug, part_filename, section_filenames, slugify


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello-world"

    def test_unicode_diacritics(self):
        # é → e, ñ → n, ü → u
        assert slugify("Café") == "cafe"
        assert slugify("Über") == "uber"

    def test_em_dash_and_colon(self):
        result = slugify("Chapter — One: 'Test'")
        assert result == "chapter-one-test"

    def test_empty_input(self):
        assert slugify("") == "section"
        assert slugify("   ") == "section"

    def test_long_truncation(self):
        long = "a" * 50
        result = slugify(long, max_len=40)
        assert len(result) == 40
        assert not result.endswith("-")

    def test_truncation_no_trailing_dash(self):
        # "hello-world" at max_len=7 → "hello" (truncate removes trailing dash)
        result = slugify("hello world", max_len=7)
        assert not result.endswith("-")
        assert len(result) <= 7

    def test_numbers_preserved(self):
        assert slugify("Chapter 3") == "chapter-3"

    def test_all_special_chars(self):
        result = slugify("!!!###$$$")
        assert result == "section"

    def test_book_slug_max_60(self):
        title = "A " * 35  # 70 chars
        result = book_slug(title)
        assert len(result) <= 60
        assert not result.endswith("-")


# ---------------------------------------------------------------------------
# part_filename
# ---------------------------------------------------------------------------

class TestPartFilename:
    def test_single_part(self):
        name = part_filename("my-book", 7, "Chapter One", 0, 1)
        assert name == "my-book-007-chapter-one.m4a"

    def test_multi_part(self):
        name = part_filename("my-book", 7, "Chapter One", 0, 4)
        assert name == "my-book-007-chapter-one.pt00.m4a"

    def test_multi_part_last(self):
        name = part_filename("my-book", 7, "Chapter One", 3, 4)
        assert name == "my-book-007-chapter-one.pt03.m4a"

    def test_spine_index_zero_padded(self):
        name = part_filename("book", 1, "A", 0, 1)
        assert "-001-" in name

    def test_example_from_spec_multi(self):
        name = part_filename("a-memory-called-empire", 7, "Chapter One", 0, 3)
        assert name == "a-memory-called-empire-007-chapter-one.pt00.m4a"

    def test_example_from_spec_single(self):
        name = part_filename("a-memory-called-empire", 13, "Interlude Two", 0, 1)
        assert name == "a-memory-called-empire-013-interlude-two.m4a"

    def test_part_index_99_ok(self):
        name = part_filename("book", 1, "Chapter", 99, 100)
        assert name.endswith(".pt99.m4a")

    def test_part_index_over_99_raises(self):
        with pytest.raises(ValueError, match="99"):
            part_filename("book", 1, "Chapter", 100, 101)


# ---------------------------------------------------------------------------
# Lexicographic sort == reading order (KEY PROPERTY)
# ---------------------------------------------------------------------------

def _make_section_plan(
    section_id: str,
    spine_index: int,
    title: str,
    n_parts: int,
) -> SectionPlan:
    """Build a SectionPlan with exactly n_parts by controlling word count."""
    from narratelm.bookjson import Section

    # Each part should be ~3000 words (20min * 150wpm cap)
    # Use ~3000 * n_parts words total so we get n_parts parts
    words_per_para = 100
    paras_needed = max(1, (3000 * n_parts) // words_per_para)
    para = " ".join(["word"] * (words_per_para - 1)) + " done."
    paragraphs = [para] * paras_needed

    sec = Section(
        id=section_id,
        index=spine_index,
        source_file="x.xhtml",
        title=title,
        type="chapter",
        skip=False,
        skip_reason=None,
        word_count=words_per_para * paras_needed,
        est_minutes=0.0,
        paragraphs=paragraphs,
    )
    return plan_section(sec, window_words=400, max_part_minutes=20.0, wpm=150)


class TestLexicographicOrder:
    def test_sort_equals_reading_order(self):
        """Filenames sorted lexicographically must equal reading order."""
        bslug = "test-book"
        plans_data = [
            ("s-003", 3, "Prologue", 1),
            ("s-007", 7, "Chapter One", 4),
            ("s-013", 13, "Chapter Two", 1),
            ("s-020", 20, "Epilogue", 4),
        ]

        all_filenames_in_order: list[str] = []
        plans = []
        for sid, spine, title, nparts in plans_data:
            plan = _make_section_plan(sid, spine, title, nparts)
            plans.append(plan)
            fnames = section_filenames(plan, bslug)
            all_filenames_in_order.extend(fnames)

        # Verify reading order count
        assert len(all_filenames_in_order) > 0

        # THE KEY PROPERTY: sorted == reading order
        assert sorted(all_filenames_in_order) == all_filenames_in_order, (
            f"Lexicographic sort does not match reading order!\n"
            f"Sorted:  {sorted(all_filenames_in_order)}\n"
            f"InOrder: {all_filenames_in_order}"
        )

    def test_sort_equals_reading_order_many_parts(self):
        """A section with 12 parts must sort correctly."""
        bslug = "big-book"
        plans_data = [
            ("s-001", 1, "Part One", 12),
            ("s-002", 2, "Part Two", 1),
        ]

        all_filenames_in_order: list[str] = []
        for sid, spine, title, nparts in plans_data:
            plan = _make_section_plan(sid, spine, title, nparts)
            fnames = section_filenames(plan, bslug)
            all_filenames_in_order.extend(fnames)

        assert sorted(all_filenames_in_order) == all_filenames_in_order

    def test_two_digit_pt_boundary(self):
        """pt00..pt99 (100 parts) — check boundary behavior and ValueError at 100."""
        # pt99 is valid (part_index=99)
        name = part_filename("book", 1, "Chapter", 99, 100)
        assert "pt99" in name

        # part_index=100 must raise
        with pytest.raises(ValueError):
            part_filename("book", 1, "Chapter", 100, 101)

    def test_mixed_spine_indexes_sort(self):
        """Mixed single and multi-part at various spine indexes sort correctly."""
        bslug = "epic-tale"
        configs = [
            ("s-005", 5, "Intro", 1),
            ("s-010", 10, "Rising Action", 3),
            ("s-015", 15, "Climax", 1),
            ("s-025", 25, "Resolution", 2),
        ]

        all_in_order: list[str] = []
        for sid, spine, title, nparts in configs:
            plan = _make_section_plan(sid, spine, title, nparts)
            all_in_order.extend(section_filenames(plan, bslug))

        assert sorted(all_in_order) == all_in_order
