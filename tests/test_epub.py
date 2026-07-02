from __future__ import annotations

from pathlib import Path

import pytest

from narratelm.epub import Epub, SpineItem


@pytest.fixture(scope="module")
def epub(test_epub: Path) -> Epub:
    ep = Epub.open(test_epub)
    yield ep
    ep.close()


# OPF-relative hrefs of a couple of Gutenberg spine documents, used below.
CH1_HREF = "1465615483244700307_55-h-2.htm.xhtml"
CH2_HREF = "1465615483244700307_55-h-3.htm.xhtml"
INTRO_HREF = "1465615483244700307_55-h-1.htm.xhtml"


def test_metadata(epub: Epub) -> None:
    md = epub.metadata
    assert md.title == "The Wonderful Wizard of Oz"
    assert md.author == "L. Frank Baum"
    assert md.language == "en"
    assert md.cover_href == "OEBPS/6780371351323423538_cover.jpg"


def test_cover_bytes_readable(epub: Epub) -> None:
    data = epub.read_bytes(epub.metadata.cover_href)
    assert data[:3] == b"\xff\xd8\xff"  # JPEG magic


def test_spine_ordered_and_linear_only(epub: Epub) -> None:
    spine = epub.spine
    assert all(isinstance(s, SpineItem) for s in spine)
    # Indexes are contiguous spine positions.
    assert [s.index for s in spine] == list(range(len(spine)))
    ids = [s.id for s in spine]
    # Project Gutenberg wraps the cover first and appends its license last.
    assert ids[0] == "coverpage-wrapper"
    assert ids[-1] == "pg-footer"
    # The story chapters sit between, in reading order.
    assert ids.index("item4") < ids.index("item28")


def test_toc_titles(epub: Epub) -> None:
    assert epub.toc_titles  # non-empty
    assert epub.title_for(CH2_HREF) == "Chapter II The Council with the Munchkins"
    assert epub.title_for(INTRO_HREF) == "Introduction"


def test_nav_href(epub: Epub) -> None:
    assert epub.nav_href == "toc.xhtml"


def test_read_document(epub: Epub) -> None:
    raw = epub.read_document(CH1_HREF)
    assert isinstance(raw, bytes)
    assert b"Dorothy" in raw
