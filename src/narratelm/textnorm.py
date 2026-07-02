"""XHTML -> narration-ready paragraphs.

Turns one spine document into a flat list of paragraph strings suitable for TTS:
Kobo span fragmentation is merged, page-break markers are dropped without eating
the surrounding words, scene breaks collapse to a ``SCENE_BREAK`` sentinel, and
typography is normalised the way the VibeVoice demo expects.
"""

from __future__ import annotations

import re
import unicodedata

from lxml import html as lxml_html

from narratelm.bookjson import SCENE_BREAK

_OPS_NS = "http://www.idpf.org/2007/ops"
# EPUB xhtml is UTF-8; force it so undeclared bytes aren't misread as latin-1.
_PARSER = lxml_html.HTMLParser(encoding="utf-8")
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_LEAF_BLOCK = {"p", "li", "dt", "dd"} | _HEADINGS
_REMOVE_TAGS = {"script", "style", "svg", "table"}

# Whitespace variants collapsed to a plain space; zero-width chars are dropped.
_SPACE_CHARS = "           "
_ZERO_WIDTH = "​‌‍﻿"
_SOFT_HYPHEN = "­"
_TRANSLATE = {ord(c): " " for c in _SPACE_CHARS}
_TRANSLATE.update({ord(c): None for c in _ZERO_WIDTH + _SOFT_HYPHEN})

# A paragraph made only of these (plus whitespace) is a decorative scene break.
_DECORATION_RE = re.compile(r"^[\s*⁂✱٭•·★☆❦✿]+$")
_ANGLE_RE = re.compile(r"<([^<>]{1,500})>")
_EMDASH_RE = re.compile(r"\s*—\s*")
_WS_RE = re.compile(r"\s+")


def _local(tag: object) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _epub_type(el) -> str:
    return el.get("epub:type") or el.get(f"{{{_OPS_NS}}}type") or ""


def _is_pagebreak(el) -> bool:
    return el.get("role") == "doc-pagebreak" or "pagebreak" in _epub_type(el).split()


def _should_remove(el) -> bool:
    tag = _local(el.tag)
    if tag in _REMOVE_TAGS:
        return True
    if _is_pagebreak(el):
        return True
    etype = _epub_type(el)
    if tag in {"sup", "a"} and "noteref" in etype.split():
        return True
    if tag == "aside" and "footnote" in etype.split():
        return True
    return False


def _normalize_text(text: str) -> str:
    """Apply NFKC + the demo's typography rules to one assembled paragraph."""
    s = unicodedata.normalize("NFKC", text)
    s = s.translate(_TRANSLATE)
    s = s.replace("…", "...")  # ellipsis
    s = s.replace("’", "'")  # curly apostrophe -> straight (curly quotes kept)
    s = _EMDASH_RE.sub(" — ", s)  # normalise em-dash spacing
    s = _WS_RE.sub(" ", s).strip()
    s = _ANGLE_RE.sub(lambda m: "“" + m.group(1) + "”", s)  # <telepathy> -> “ ”
    return s


def _format_title(text: str) -> str:
    """Title-case an ALL-CAPS heading and guarantee a spoken sentence terminator."""
    s = _WS_RE.sub(" ", text).strip()
    if not s:
        return s
    if s == s.upper() and s != s.lower():  # all-caps with letters
        s = s.title()
    if s[-1] not in ".!?":
        s += "."
    return s


def _is_decorative_break(el, norm: str) -> bool:
    if norm and _DECORATION_RE.match(norm):
        return True
    if not norm and el.find(".//img") is not None:
        return True
    return False


def _walk(el, out: list[tuple[str, str]]) -> None:
    """Depth-first block extraction into (kind, text) tuples in document order."""
    for child in el:
        tag = _local(child.tag)
        if tag == "hr":
            out.append(("scene", ""))
        elif tag in _LEAF_BLOCK:
            norm = _normalize_text("".join(child.itertext()))
            if _is_decorative_break(child, norm):
                out.append(("scene", ""))
            elif norm:
                out.append(("head" if tag in _HEADINGS else "para", norm))
        else:
            _walk(child, out)


def _document_items(xhtml: bytes) -> list[tuple[str, str]]:
    """Parse, strip noise, and walk the body into ordered (kind, text) blocks."""
    try:
        root = lxml_html.fromstring(xhtml, parser=_PARSER)
    except Exception:  # pragma: no cover - malformed fallback
        root = lxml_html.fragment_fromstring(xhtml, create_parent="body")

    for el in list(root.iter()):
        if el.getparent() is not None and _should_remove(el):
            el.drop_tree()

    body = root.find(".//body")
    if body is None:
        body = root

    items: list[tuple[str, str]] = []
    _walk(body, items)
    return items


def _leading_run(items: list[tuple[str, str]]) -> tuple[int, list[str]]:
    lead = 0
    while lead < len(items) and items[lead][0] == "head":
        lead += 1
    return lead, [t for _, t in items[:lead]]


def leading_title(xhtml: bytes) -> str | None:
    """The merged, formatted leading heading run (spoken title) or None."""
    _, run = _leading_run(_document_items(xhtml))
    if not run:
        return None
    return _format_title(" ".join(run)) or None


def normalize_document(xhtml: bytes, *, title_hint: str | None = None) -> list[str]:
    """Extract narration paragraphs from a spine document.

    The first paragraph is the spoken chapter title (from *title_hint* when given,
    else merged from the document's leading heading run); ``SCENE_BREAK`` marks
    scene breaks and is never emitted first, last, or consecutively.
    """
    items = _document_items(xhtml)
    lead, title_run = _leading_run(items)

    paragraphs: list[str] = []
    if title_run:
        base = title_hint if title_hint else " ".join(title_run)
        title = _format_title(base)
        if title:
            paragraphs.append(title)

    for kind, text in items[lead:]:
        if kind == "scene":
            if paragraphs and paragraphs[-1] != SCENE_BREAK:
                paragraphs.append(SCENE_BREAK)
        else:
            paragraphs.append(text)

    while paragraphs and paragraphs[-1] == SCENE_BREAK:
        paragraphs.pop()
    return paragraphs


def count_words(paragraphs: list[str]) -> int:
    """Total spoken words (scene-break sentinels excluded)."""
    return sum(len(p.split()) for p in paragraphs if p != SCENE_BREAK)
