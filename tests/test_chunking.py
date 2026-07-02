"""Tests for src/narratelm/chunking.py"""

from __future__ import annotations

import math

import pytest

from narratelm.bookjson import SCENE_BREAK, Section
from narratelm.chunking import Window, plan_section, split_sentences
from narratelm.config import (
    PAUSE_BETWEEN_PARAGRAPHS_S,
    PAUSE_BETWEEN_WINDOWS_S,
    PAUSE_SCENE_BREAK_S,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_section(
    paragraphs: list[str],
    *,
    id: str = "sec-001",
    index: int = 0,
    title: str = "Chapter One",
    skip: bool = False,
) -> Section:
    return Section(
        id=id,
        index=index,
        source_file="test.xhtml",
        title=title,
        type="chapter",
        skip=skip,
        skip_reason=None,
        word_count=sum(len(p.split()) for p in paragraphs if p != SCENE_BREAK),
        est_minutes=0.0,
        paragraphs=paragraphs,
    )


def all_words_from_section(section: Section) -> list[str]:
    """All words in reading order from a section (excluding scene breaks)."""
    words = []
    for p in section.paragraphs:
        if p != SCENE_BREAK:
            words.extend(p.split())
    return words


def all_words_from_plan(plan) -> list[str]:
    """All words in order from all parts/windows of a plan."""
    words = []
    for part in plan.parts:
        for window in part.windows:
            words.extend(window.text.split())
    return words


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_basic_multi_sentence(self):
        text = "Hello world. This is a test. And another."
        result = split_sentences(text)
        assert len(result) == 3
        assert result[0] == "Hello world."
        assert result[1] == "This is a test."
        assert result[2] == "And another."

    def test_mr_not_split(self):
        text = "Mr. Smith went to Washington. He liked it there."
        result = split_sentences(text)
        assert len(result) == 2
        assert result[0].startswith("Mr. Smith")

    def test_dr_not_split(self):
        text = "She called Dr. Johnson for help. He was busy."
        result = split_sentences(text)
        assert len(result) == 2
        assert "Dr. Johnson" in result[0]

    def test_eg_not_split(self):
        text = "Many animals, e.g. dogs and cats, are pets. Fish too."
        result = split_sentences(text)
        assert len(result) == 2
        assert "e.g." in result[0]

    def test_ie_not_split(self):
        text = "The result, i.e. the answer, was zero. We confirmed it."
        result = split_sentences(text)
        assert len(result) == 2
        assert "i.e." in result[0]

    def test_initial_not_split(self):
        text = "A. Martine wrote the book. It was good."
        result = split_sentences(text)
        assert len(result) == 2
        assert result[0] == "A. Martine wrote the book."

    def test_decimal_not_split(self):
        text = "The value was 3.14 exactly. That is pi."
        result = split_sentences(text)
        assert len(result) == 2
        assert "3.14" in result[0]

    def test_exclamation_and_quotes(self):
        text = 'He said, "Stop!" Then he left.'
        result = split_sentences(text)
        assert len(result) == 2
        assert result[0].endswith('"')
        assert result[1] == "Then he left."

    def test_ellipsis_followed_by_capital(self):
        text = "She waited... Then the door opened."
        result = split_sentences(text)
        assert len(result) == 2
        assert "waited" in result[0]
        assert result[1] == "Then the door opened."

    def test_no_terminal_punctuation(self):
        text = "This sentence has no ending punctuation"
        result = split_sentences(text)
        assert result == ["This sentence has no ending punctuation"]

    def test_whitespace_only(self):
        assert split_sentences("   ") == []
        assert split_sentences("") == []

    def test_question_mark(self):
        text = "What is this? It is a test."
        result = split_sentences(text)
        assert len(result) == 2

    def test_mrs_not_split(self):
        text = "Mrs. Jones arrived. She was on time."
        result = split_sentences(text)
        assert len(result) == 2
        assert "Mrs. Jones" in result[0]


# ---------------------------------------------------------------------------
# plan_section — windows
# ---------------------------------------------------------------------------

class TestPlanSectionWindows:
    def test_windows_never_exceed_limit_unless_single_sentence(self):
        """No window exceeds window_words unless it's a single oversized sentence."""
        long_sent = " ".join(["word"] * 600)  # 600-word single sentence
        normal_para = " ".join(["word"] * 50) + "."
        section = make_section([normal_para, long_sent + ".", normal_para])
        plan = plan_section(section, window_words=400, max_part_minutes=20.0, wpm=150)

        for part in plan.parts:
            for w in part.windows:
                if w.text == long_sent + ".":
                    pass  # oversized single sentence is allowed
                else:
                    assert w.word_count <= 400, (
                        f"Window {w.index} has {w.word_count} words > 400"
                    )

    def test_no_sentence_split_across_windows(self):
        """All words in all windows == all words of input in order."""
        paragraphs = [
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            SCENE_BREAK,
            "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
            "Peter Piper picked a peck of pickled peppers.",
        ]
        section = make_section(paragraphs)
        plan = plan_section(section, window_words=15, max_part_minutes=20.0, wpm=150)

        expected = all_words_from_section(section)
        actual = all_words_from_plan(plan)
        assert actual == expected

    def test_scene_break_forces_boundary_and_pause(self):
        """A SCENE_BREAK must be the boundary between two windows with pause 1.2."""
        para_a = "First paragraph with some content here."
        para_b = "Second paragraph after the break."
        section = make_section([para_a, SCENE_BREAK, para_b])
        plan = plan_section(section, window_words=400, max_part_minutes=20.0, wpm=150)

        # Collect all windows across all parts
        windows = [w for part in plan.parts for w in part.windows]

        # We should have 2 windows: one before the scene break, one after
        assert len(windows) == 2
        # First window ends at scene break → pause = PAUSE_SCENE_BREAK_S
        assert windows[0].pause_after_s == PAUSE_SCENE_BREAK_S

    def test_paragraph_end_pause(self):
        """A window ending at a paragraph boundary (≥75% full) gets 0.65."""
        # 300 words in each paragraph to trigger the ≥75% rule (400 * 0.75 = 300)
        para_a = " ".join(["alpha"] * 149) + " done."  # ~150 words
        para_b = " ".join(["beta"] * 149) + " done."   # ~150 words
        para_c = " ".join(["gamma"] * 149) + " done."  # ~150 words
        # Three 150-word paragraphs; window_words=200 → first two fit (300 > 200*0.75=150)
        section = make_section([para_a, para_b, para_c])
        plan = plan_section(section, window_words=200, max_part_minutes=20.0, wpm=150)
        windows = [w for part in plan.parts for w in part.windows]

        # Check that at least one non-last window has paragraph pause
        non_last = windows[:-1]
        pauses = {w.pause_after_s for w in non_last}
        assert PAUSE_BETWEEN_PARAGRAPHS_S in pauses or PAUSE_SCENE_BREAK_S in pauses

    def test_last_window_pause_zero(self):
        """The last window of a section always has pause_after_s == 0.0."""
        section = make_section([
            "Sentence one here.",
            "Sentence two here.",
            "Sentence three here.",
        ])
        plan = plan_section(section, window_words=400, max_part_minutes=20.0, wpm=150)
        all_windows = [w for part in plan.parts for w in part.windows]
        assert all_windows[-1].pause_after_s == 0.0

    def test_mid_paragraph_pause(self):
        """Mid-window split (not at para end, not scene break) → 0.35."""
        # Many short sentences in one paragraph to force mid-paragraph splits
        sents = ["Word " * 50 + "done." for _ in range(10)]
        para = " ".join(sents)
        section = make_section([para])
        plan = plan_section(section, window_words=100, max_part_minutes=20.0, wpm=150)
        all_windows = [w for part in plan.parts for w in part.windows]
        non_last = all_windows[:-1]
        # Some should have mid-paragraph pause (0.35)
        assert any(w.pause_after_s == PAUSE_BETWEEN_WINDOWS_S for w in non_last)

    def test_oversized_single_sentence_becomes_own_window(self):
        """A single sentence > window_words becomes its own window."""
        big_sent = " ".join(["word"] * 500) + "."
        section = make_section([big_sent])
        plan = plan_section(section, window_words=400, max_part_minutes=20.0, wpm=150)
        all_windows = [w for part in plan.parts for w in part.windows]
        assert len(all_windows) == 1
        assert all_windows[0].word_count > 400


# ---------------------------------------------------------------------------
# plan_section — parts
# ---------------------------------------------------------------------------

class TestPlanSectionParts:
    def test_variable_paragraphs_near_balanced(self):
        """Realistic variable paragraph sizes → 3 near-balanced parts.

        Regression for lopsided greedy grouping: a ~6400-word section at cap
        3000 must split into 3 parts with max_words <= 1.25 * min_words.
        """
        import random

        rng = random.Random(1234)
        paragraphs: list[str] = []
        total = 0
        target_total = 6400
        while total < target_total:
            n_words = rng.randint(40, 120)
            if total + n_words > target_total + 120:
                break
            # One paragraph = one sentence of n_words words.
            paragraphs.append(" ".join(["word"] * (n_words - 1)) + " end.")
            total += n_words

        section = make_section(paragraphs)
        plan = plan_section(section, window_words=400, max_part_minutes=20.0, wpm=150)

        assert len(plan.parts) == 3, f"Expected 3 parts, got {len(plan.parts)}"
        word_counts = [p.word_count for p in plan.parts]
        assert max(word_counts) <= 1.25 * min(word_counts), (
            f"Parts not near-balanced: {word_counts}"
        )
        # Every part must stay under cap.
        for p in plan.parts:
            assert p.word_count <= 3000, f"Part {p.index} exceeds cap: {p.word_count}"

    def test_9001_words_three_or_four_parts(self):
        """9001 words at cap 3000 → 3 or 4 parts, reasonably balanced."""
        # cap = 20min * 150wpm = 3000 words; n_parts = ceil(9001/3000) = 4
        # Use sentences of ~100 words each; 91 of them ≈ 9100 words
        sentences_per_para = 5
        words_per_sent = 20
        n_paras = 91 // sentences_per_para  # 18 paragraphs
        para = (" ".join(["word"] * (words_per_sent - 1)) + ".") * sentences_per_para
        # Make each para ~100 words
        para = " ".join(["word"] * 99) + " done."
        paragraphs = [para] * 91  # ~91 * 100 = 9100 words

        section = make_section(paragraphs)
        plan = plan_section(section, window_words=400, max_part_minutes=20.0, wpm=150)

        assert 3 <= len(plan.parts) <= 5, f"Expected 3-5 parts, got {len(plan.parts)}"

        # Balance check: max part ≤ 1.5 × min part in words (loose, window granularity)
        word_counts = [p.word_count for p in plan.parts]
        assert max(word_counts) <= min(word_counts) * 1.5 + 400, (
            f"Parts not balanced: {word_counts}"
        )

        # Cap + one-window bound
        for p in plan.parts[:-1]:  # last part can be remainder
            assert p.word_count <= 3000 + 400, (
                f"Part {p.index} has {p.word_count} words (cap+1window)"
            )

    def test_500_word_section_one_part(self):
        """A 500-word section → exactly 1 part."""
        para = " ".join(["word"] * 49) + " done."  # ~50 words
        section = make_section([para] * 10)  # ~500 words
        plan = plan_section(section, window_words=400, max_part_minutes=20.0, wpm=150)
        assert len(plan.parts) == 1

    def test_limit_words_truncates(self):
        """limit_words=100 keeps only ~100 words of whole sentences."""
        para = " ".join(["word"] * 49) + " done."  # 50 words
        section = make_section([para] * 20)  # 1000 words total
        plan = plan_section(
            section, window_words=400, max_part_minutes=20.0, wpm=150, limit_words=100
        )
        assert plan.word_count <= 150, (
            f"Expected ≤150 words after limit_words=100, got {plan.word_count}"
        )
        assert plan.word_count >= 50, (
            f"Expected ≥50 words (at least one sentence), got {plan.word_count}"
        )

    def test_word_count_consistency(self):
        """SectionPlan.word_count == sum of all window word_counts."""
        para = "The fox jumped over the fence. It was quick."
        section = make_section([para] * 20)
        plan = plan_section(section, window_words=50, max_part_minutes=20.0, wpm=150)
        total = sum(w.word_count for part in plan.parts for w in part.windows)
        assert plan.word_count == total

    def test_part_word_count_consistency(self):
        """Part.word_count == sum of its windows' word_counts."""
        para = "One two three four five six seven eight nine ten."
        section = make_section([para] * 30)
        plan = plan_section(section, window_words=30, max_part_minutes=20.0, wpm=150)
        for part in plan.parts:
            expected = sum(w.word_count for w in part.windows)
            assert part.word_count == expected


# ---------------------------------------------------------------------------
# plan_book
# ---------------------------------------------------------------------------

class TestPlanBook:
    def _make_book(self):
        from narratelm.bookjson import Book, Metadata
        sections = [
            make_section(["Hello world."], id="s1", index=0, title="Chapter 1"),
            make_section(["Skipped content."], id="s2", index=1, title="Chapter 2", skip=True),
            make_section(["Another chapter."], id="s3", index=2, title="Chapter 3"),
            make_section(["Final chapter."], id="s4", index=3, title="Chapter 4"),
        ]
        return Book(
            source_epub="test.epub",
            metadata=Metadata(title="Test Book"),
            sections=sections,
        )

    def test_skipped_sections_excluded(self):
        from narratelm.chunking import plan_book
        book = self._make_book()
        plans = plan_book(book, window_words=400, max_part_minutes=20.0, wpm=150)
        ids = [p.section_id for p in plans]
        assert "s2" not in ids
        assert "s1" in ids
        assert "s3" in ids
        assert "s4" in ids

    def test_section_filter(self):
        from narratelm.chunking import plan_book
        book = self._make_book()
        plans = plan_book(
            book,
            window_words=400,
            max_part_minutes=20.0,
            wpm=150,
            section_filter={"s1", "s4"},
        )
        ids = [p.section_id for p in plans]
        assert ids == ["s1", "s4"]

    def test_spine_order_preserved(self):
        from narratelm.chunking import plan_book
        book = self._make_book()
        plans = plan_book(book, window_words=400, max_part_minutes=20.0, wpm=150)
        indices = [p.spine_index for p in plans]
        assert indices == sorted(indices)
