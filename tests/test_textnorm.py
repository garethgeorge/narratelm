from __future__ import annotations

from pathlib import Path

import pytest

from narratelm.bookjson import SCENE_BREAK
from narratelm.epub import Epub
from narratelm.textnorm import count_words, normalize_document


# "Chapter II The Council with the Munchkins" — OPF-relative href + its TOC title.
CH2_HREF = "1465615483244700307_55-h-3.htm.xhtml"
CH2_TITLE = "Chapter II The Council with the Munchkins"


@pytest.fixture(scope="module")
def chapter_raw(test_epub: Path) -> bytes:
    ep = Epub.open(test_epub)
    try:
        return ep.read_document(CH2_HREF)
    finally:
        ep.close()


@pytest.fixture(scope="module")
def chapter_paras(chapter_raw: bytes) -> list[str]:
    return normalize_document(chapter_raw, title_hint=CH2_TITLE)


def test_title_is_first_paragraph(chapter_paras: list[str]) -> None:
    assert chapter_paras[0] == f"{CH2_TITLE}."


def test_no_leaked_markup(chapter_paras: list[str]) -> None:
    for p in chapter_paras:
        assert "<" not in p and ">" not in p  # tags gone


def test_em_dash_spacing_real(chapter_paras: list[str]) -> None:
    # Em dashes in the source render as space-flanked em dashes in the output.
    assert any("gently — for a cyclone —" in p for p in chapter_paras)


def test_curly_quotes_preserved_real(chapter_paras: list[str]) -> None:
    # The Munchkins' dialogue keeps its curly double quotes.
    assert any("“" in p and "”" in p for p in chapter_paras)


# -- synthetic unit tests -------------------------------------------------

def test_hr_transition_scene_break() -> None:
    xhtml = b"<html><body><p>Before.</p><hr class='transition'/><p>After.</p></body></html>"
    assert normalize_document(xhtml) == ["Before.", SCENE_BREAK, "After."]


def test_angle_bracket_synthetic() -> None:
    xhtml = b"<html><body><p>He said &lt;come here&gt; softly.</p></body></html>"
    assert normalize_document(xhtml) == ["He said “come here” softly."]


def test_angle_bracket_noop_without_pairs() -> None:
    xhtml = b"<html><body><p>Plain text, no markers.</p></body></html>"
    assert normalize_document(xhtml) == ["Plain text, no markers."]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("word—word", "word — word"),          # em-dash spacing
        ("wait… what", "wait... what"),             # ellipsis
        ("it’s fine", "it's fine"),                 # curly apostrophe
        ("a b c", "a b c"),                    # nbsp / thin space
        ("sof­tly", "softly"),                      # soft hyphen stripped
        ("keep “quotes” here", "keep “quotes” here"),  # curly quotes kept
    ],
)
def test_typography_rules(raw: str, expected: str) -> None:
    xhtml = f"<html><body><p>{raw}</p></body></html>".encode("utf-8")
    assert normalize_document(xhtml) == [expected]


def test_count_words_excludes_sentinel() -> None:
    paras = ["Two words", SCENE_BREAK, "three more words"]
    assert count_words(paras) == 5
