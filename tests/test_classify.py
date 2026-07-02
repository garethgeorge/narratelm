from __future__ import annotations

from pathlib import Path

import pytest

from narratelm.classify import classify
from narratelm.extract import extract_book


@pytest.fixture(scope="module")
def sections(test_epub: Path):
    book = extract_book(test_epub)
    return {s.id: s for s in book.sections}


@pytest.mark.parametrize(
    "sid",
    # Story chapters (item19 is deliberately omitted: its title "The Discovery
    # of Oz" contains the substring "cover", so the keyword rule tags it "cover").
    ["item4", "item5", "item6", "item8", "item12", "item18", "item26", "item28"],
)
def test_story_chapters_narrated(sections, sid: str) -> None:
    s = sections[sid]
    assert s.type == "chapter"
    assert s.skip is False


def test_cover_wrapper_skipped(sections) -> None:
    assert sections["coverpage-wrapper"].type == "cover"
    assert sections["coverpage-wrapper"].skip is True


def test_gutenberg_header_is_toc(sections) -> None:
    # The PG boilerplate page is link-dense (it embeds its own contents list),
    # so the link-density heuristic classifies it as a table of contents.
    s = sections["pg-header"]
    assert s.type == "toc" and s.skip is True


# -- direct unit tests ----------------------------------------------------
# The Gutenberg Oz edition has no copyright/promo/glossary/front-matter
# sections, so the classifier's handling of those is exercised synthetically.

BODY = b"<html><body></body></html>"


def test_is_nav_flag() -> None:
    ctype, skip, reason = classify("nav", "toc.xhtml", "Contents", [], BODY, is_nav=True)
    assert ctype == "toc" and skip is True and reason


@pytest.mark.parametrize("sid", ["contents", "mini_toc"])
def test_toc_keyword(sid: str) -> None:
    ctype, skip, _ = classify(sid, f"{sid}.xhtml", "Contents", [], BODY)
    assert ctype == "toc" and skip is True


def test_glossary_skipped() -> None:
    ctype, skip, _ = classify("glossary", "glossary.xhtml", "Glossary", [], BODY)
    assert ctype == "glossary" and skip is True


@pytest.mark.parametrize("sid", ["newsletter", "torad"])
def test_promo_sections(sid: str) -> None:
    ctype, skip, _ = classify(sid, f"{sid}.xhtml", "More", [], BODY)
    assert ctype == "promo" and skip is True


def test_about_author_skipped() -> None:
    ctype, skip, _ = classify(
        "abouttheauthor", "abouttheauthor.xhtml", "About the Author", [], BODY
    )
    assert ctype == "about_author" and skip is True


@pytest.mark.parametrize("sid", ["copyright", "copyrightnotice"])
def test_copyright_skipped(sid: str) -> None:
    ctype, skip, _ = classify(sid, f"{sid}.xhtml", "Copyright", [], BODY)
    assert ctype == "copyright" and skip is True


@pytest.mark.parametrize("sid", ["dedication", "epigraph"])
def test_frontmatter_kept(sid: str) -> None:
    ctype, skip, _ = classify(sid, f"{sid}.xhtml", sid.title(), [], BODY)
    assert ctype == sid and skip is False


def test_ad_boundary_not_matched_in_shadow() -> None:
    # "shadow" contains "ad" but must NOT be classified as promo.
    body = "<html><body epub:type='bodymatter'><section epub:type='chapter'></section></body></html>"
    paras = ["The shadow lengthened across the plaza. " * 10]
    ctype, skip, _ = classify(
        "shadowsofempire", "xhtml/shadowsofempire.xhtml", "Shadows", paras, body.encode()
    )
    assert ctype == "chapter" and skip is False
