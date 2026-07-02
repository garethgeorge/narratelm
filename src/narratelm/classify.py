"""Assign each section a semantic type and a sensible default ``skip`` flag.

The type feeds the book JSON and drives which sections narrate by default. The
first matching rule wins; the returned reason string names the rule that fired
so a human editor can see *why* something was skipped.
"""

from __future__ import annotations

import posixpath
import re

from lxml import html as lxml_html

from narratelm.textnorm import count_words

_OPS_NS = "http://www.idpf.org/2007/ops"
# EPUB xhtml is UTF-8; force it so undeclared bytes aren't misread as latin-1.
_PARSER = lxml_html.HTMLParser(encoding="utf-8")

# Types whose default is to skip narration.
_SKIP_TYPES = {
    "toc", "cover", "titlepage", "copyright", "acknowledgments", "about_author",
    "promo", "glossary", "index", "notes", "image_page", "short",
}

# id / basename / title substring keywords -> type, in priority order.
_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"cover"), "cover"),
    (re.compile(r"title"), "titlepage"),
    (re.compile(r"copyright"), "copyright"),
    (re.compile(r"dedication"), "dedication"),
    (re.compile(r"epigraph"), "epigraph"),
    (re.compile(r"acknowledg"), "acknowledgments"),
    (re.compile(r"about[\s_-]?the[\s_-]?author|abouttheauthor"), "about_author"),
    (re.compile(r"newsletter|mailinglist"), "promo"),
    (re.compile(r"torad|backad|adcard|alsoby|praise|\bads?\b"), "promo"),
    (re.compile(r"glossary"), "glossary"),
    (re.compile(r"index"), "index"),
    (re.compile(r"footnotes|\bnotes\b"), "notes"),
]

# epub:type / role tokens -> type, tried in this order (specific before generic).
_ATTR_MAP: list[tuple[str, str]] = [
    ("cover", "cover"),
    ("titlepage", "titlepage"),
    ("copyright-page", "copyright"),
    ("dedication", "dedication"),
    ("epigraph", "epigraph"),
    ("toc", "toc"),
    ("glossary", "glossary"),
    ("appendix", "appendix"),
    ("acknowledgments", "acknowledgments"),
    ("bodymatter", "chapter"),
    ("chapter", "chapter"),
]

_TOC_RE = re.compile(r"toc|contents|mini_toc")
_TERM_DEF_RE = re.compile(r"^.{1,40}[:—-]\s")


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def default_skip(section_type: str) -> bool:
    """Whether a section of this type should skip narration by default."""
    return section_type in _SKIP_TYPES


def _haystacks(section_id: str, href: str, title: str) -> list[str]:
    base = posixpath.splitext(posixpath.basename(href))[0]
    return [section_id.lower(), base.lower(), title.lower()]


def _top_level_attr_tokens(body) -> set[str]:
    """epub:type + role tokens from the body and its top-level <section>s only.

    Deliberately ignores nested markup (e.g. a chapter's inline
    ``<blockquote epub:type="epigraph">``) which would otherwise mislabel it.
    """
    tokens: set[str] = set()

    def collect(el) -> None:
        etype = el.get("epub:type") or el.get(f"{{{_OPS_NS}}}type") or ""
        tokens.update(etype.split())
        role = el.get("role") or ""
        for r in role.split():
            tokens.add(r[4:] if r.startswith("doc-") else r)

    collect(body)
    for sec in body.iterdescendants():
        if _local(sec.tag) != "section":
            continue
        if not any(_local(a.tag) == "section" for a in sec.iterancestors()):
            collect(sec)
    return tokens


def classify(
    section_id: str,
    href: str,
    title: str,
    paragraphs: list[str],
    xhtml: bytes,
    *,
    is_nav: bool = False,
) -> tuple[str, bool, str | None]:
    hays = _haystacks(section_id, href, title)

    # 1. Navigation / table of contents.
    if is_nav:
        return "toc", True, "nav document"
    if any(_TOC_RE.search(h) for h in hays):
        return "toc", True, "toc/contents keyword"

    # 2. id / href / title keyword map.
    for pattern, ctype in _KEYWORDS:
        if any(pattern.search(h) for h in hays):
            return ctype, default_skip(ctype), f"keyword {pattern.pattern!r}"

    # 3. epub:type / role on the body/section.
    tree = lxml_html.fromstring(xhtml, parser=_PARSER)
    body = tree.find(".//body")
    if body is None:
        body = tree
    tokens = _top_level_attr_tokens(body)
    for token, ctype in _ATTR_MAP:
        if token in tokens:
            return ctype, default_skip(ctype), f"epub:type/role {token!r}"

    # 4. Structural heuristics.
    words = count_words(paragraphs)
    has_img = body.find(".//img") is not None
    if words < 50:
        ctype = "image_page" if has_img else "short"
        return ctype, True, f"{words} words, image={has_img}"

    all_text = "".join(body.itertext())
    total_chars = len("".join(all_text.split()))
    link_chars = sum(
        len("".join("".join(a.itertext()).split())) for a in body.findall(".//a")
    )
    if total_chars and link_chars / total_chars > 0.30:
        return "toc", True, f"link-dense ({link_chars}/{total_chars})"

    low = "\n".join(paragraphs).lower()
    if words < 600 and ("all rights reserved" in low or "isbn" in low):
        return "copyright", True, "copyright text scan"

    # 5. Glossary / definition-list heavy.
    spoken = [p for p in paragraphs if p]
    has_dl = body.find(".//dl") is not None
    termish = sum(1 for p in spoken if _TERM_DEF_RE.match(p))
    if len(spoken) > 20 and (has_dl or termish / len(spoken) > 0.5):
        return "glossary", True, "definition-list heavy"

    # 6. Default: narrated chapter.
    return "chapter", False, None
