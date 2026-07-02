"""Text-planning layer: split sections into windows and parts for TTS generation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from narratelm.bookjson import SCENE_BREAK, Book, Section
from narratelm.config import (
    PAUSE_BETWEEN_PARAGRAPHS_S,
    PAUSE_BETWEEN_WINDOWS_S,
    PAUSE_SCENE_BREAK_S,
)

# ---------------------------------------------------------------------------
# Abbreviations that do NOT end sentences
# ---------------------------------------------------------------------------
_ABBREVS = {
    "mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr",
    "vs", "etc", "approx", "dept",
}

# Sentence split regex.
# Matches after terminal punctuation (.!?...) + optional closing quotes/brackets,
# followed by whitespace, followed by a sentence-start character.
# Using \u escapes for curly quotes to avoid embedding non-ASCII in source.
_CLOSE_QUOTE_CHARS = (
    "\""       # ASCII double quote
    "'"        # ASCII single quote
    "“"   # left double quotation mark
    "”"   # right double quotation mark
    "‘"   # left single quotation mark
    "’"   # right single quotation mark
    "]"
    ")"
)
_CLOSE_Q_CLASS = "[" + re.escape("".join(_CLOSE_QUOTE_CHARS)) + "]*"

_START_CHARS = (
    "A-Z"
    "0-9"
    "\""
    "'"
    "“"
    "‘"
    "\\-"
    "—"   # em dash
)
_START_CLASS = "[" + _START_CHARS + "]"

# Terminal punctuation chars (for lookbehind)
_ELLIPSIS_CHAR = "…"

# Build the split pattern
_SPLIT_PATTERN = (
    r"(?<=[.!?" + _ELLIPSIS_CHAR + r"])"
    + _CLOSE_Q_CLASS
    + r"\s+"
    + r"(?=" + _START_CLASS + r")"
)
_SPLIT_RE = re.compile(_SPLIT_PATTERN)

# Token pattern: word chars (possibly with dots) before optional closing punct
_TOKEN_RE = re.compile(r"([A-Za-z0-9.]+)[" + re.escape("".join(_CLOSE_QUOTE_CHARS)) + r"\s]*$")


def split_sentences(text: str) -> list[str]:
    """Regex-based sentence splitter for narration text.

    Splits after period/bang/question/ellipsis optionally followed by closing
    punctuation, when followed by whitespace and an uppercase-ish char.
    Guards against common abbreviations, single-letter initials, and decimal
    numbers.  A rare bad split is harmless; never crashes.
    Whitespace-only input -> [].  No terminal punctuation -> single sentence.
    """
    text = text.strip()
    if not text:
        return []

    candidates = list(_SPLIT_RE.finditer(text))
    if not candidates:
        return [text]

    sentences: list[str] = []
    prev = 0
    for m in candidates:
        split_pos = m.start()
        before = text[:split_pos].rstrip()
        token_match = _TOKEN_RE.search(before)
        if token_match:
            raw_token = token_match.group(1)
            token = raw_token.rstrip(".")
            # Single-letter initial like "A. Martine" -- don't split
            if len(token) == 1 and token.isalpha():
                continue
            # Decimal number like "3.14" -- don't split
            if re.match(r"^\d+\.\d+$", raw_token):
                continue
            # Common abbreviations
            if token.lower() in _ABBREVS:
                continue
            # e.g. / i.e.
            ctx = before.lower().rstrip()
            if ctx.endswith("e.g") or ctx.endswith("i.e"):
                continue

        chunk = text[prev : m.end()].strip()
        if chunk:
            sentences.append(chunk)
        prev = m.end()

    tail = text[prev:].strip()
    if tail:
        sentences.append(tail)

    return sentences if sentences else [text]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Window:
    index: int           # 0-based within its section
    text: str            # narration text joined with " "
    word_count: int
    pause_after_s: float


@dataclass(frozen=True)
class Part:
    index: int
    windows: tuple[Window, ...]
    word_count: int
    est_minutes: float


@dataclass(frozen=True)
class SectionPlan:
    section_id: str
    spine_index: int
    title: str
    parts: tuple[Part, ...]
    word_count: int


# ---------------------------------------------------------------------------
# plan_section helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def _build_windows(
    paragraphs: list[str],
    window_words: int,
    limit_words: int | None,
) -> list[tuple[list[str], str]]:
    """Build windows from paragraphs.

    Returns list of (sentence_list, boundary_type) where boundary_type is
    one of 'scene_break', 'paragraph', 'mid'.
    """
    # Parse paragraphs into groups: (sentences, followed_by_scene_break)
    para_groups: list[tuple[list[str], bool]] = []
    total_limited_words = 0
    done = False

    i = 0
    while i < len(paragraphs) and not done:
        para = paragraphs[i]

        if para == SCENE_BREAK:
            # Mark previous group as followed by scene break
            if para_groups:
                sents, _ = para_groups[-1]
                para_groups[-1] = (sents, True)
            i += 1
            continue

        sents = split_sentences(para)
        if not sents:
            i += 1
            continue

        if limit_words is not None:
            kept: list[str] = []
            for s in sents:
                wc = _word_count(s)
                if total_limited_words + wc > limit_words:
                    done = True
                    break
                kept.append(s)
                total_limited_words += wc
            if kept:
                para_groups.append((kept, False))
            if done:
                break
        else:
            para_groups.append((sents, False))
        i += 1

    # Greedy window packing
    results: list[tuple[list[str], str]] = []
    current_sents: list[str] = []
    current_words = 0

    def flush(btype: str) -> None:
        nonlocal current_sents, current_words
        if not current_sents:
            return
        results.append((list(current_sents), btype))
        current_sents = []
        current_words = 0

    n_groups = len(para_groups)
    for gi, (sents, scene_after) in enumerate(para_groups):
        n_sents = len(sents)
        is_last_group = (gi == n_groups - 1)

        for si, sent in enumerate(sents):
            sw = _word_count(sent)
            is_last_sent_in_para = (si == n_sents - 1)

            # Flush before adding if window would overflow (but only if we have content)
            if current_words + sw > window_words and current_sents:
                flush("mid")

            current_sents.append(sent)
            current_words += sw

            # At paragraph end, decide whether to flush
            if is_last_sent_in_para:
                if scene_after:
                    flush("scene_break")
                elif is_last_group:
                    pass  # flush at end with "mid" (reclassified below)
                else:
                    # Paragraph boundary: flush if >= 75% full
                    if current_words >= window_words * 0.75:
                        flush("paragraph")
                    # else continue accumulating into next paragraph

    if current_sents:
        flush("mid")

    return results


def _group_parts(windows: list[Window], cap_words: int, wpm: int) -> list[Part]:
    """Group windows into near-balanced parts using ideal cut points.

    Algorithm:
      - n_parts = ceil(total / cap) (never more than the window count).
      - For each interior part boundary i (1..n_parts-1), the ideal cut in
        cumulative words is ``i * total / n_parts``.  We pick the window
        boundary whose cumulative word count is closest to that ideal; within
        +/-1 window of the closest boundary we prefer a boundary that falls on
        a paragraph/scene break (pause_after_s >= PAUSE_BETWEEN_PARAGRAPHS_S).
      - Cap guarantee: if a chosen cut would make the current part exceed cap,
        the cut is shifted back (earlier) until the part fits.  If enforcing
        the cap forces the final part over cap, n_parts is bumped by one and
        the layout recomputed (the ``+1`` allowance from the spec).
      - Parts are always non-empty.
    """
    n = len(windows)
    if n == 0:
        return []

    total = sum(w.word_count for w in windows)
    if total == 0:
        # No spoken words: keep everything in a single (silent) part.
        return [Part(index=0, windows=tuple(windows), word_count=0, est_minutes=0.0)]

    # Cumulative word counts: cum[k] = words in windows[0:k]; cum[n] == total.
    cum = [0] * (n + 1)
    for j in range(n):
        cum[j + 1] = cum[j] + windows[j].word_count

    def layout(n_parts: int) -> list[int]:
        """Return interior cut positions (window boundaries) for n_parts parts."""
        cuts: list[int] = []
        prev = 0
        for i in range(1, n_parts):
            ideal = i * total / n_parts
            # Valid range: strictly after prev, leaving >=1 window per remaining part.
            lo = prev + 1
            hi = n - (n_parts - i)
            if lo > hi:
                lo = hi = lo  # degenerate; forced single-window part
            # Closest boundary to the ideal cut (tie -> earlier boundary).
            best_k = min(range(lo, hi + 1), key=lambda k: (abs(cum[k] - ideal), k))
            # Within +/-1 window, prefer a paragraph/scene boundary if available.
            cands = [c for c in (best_k - 1, best_k, best_k + 1) if lo <= c <= hi]
            para = [
                c for c in cands
                if windows[c - 1].pause_after_s >= PAUSE_BETWEEN_PARAGRAPHS_S
            ]
            if para:
                best_k = min(para, key=lambda k: (abs(cum[k] - ideal), k))
            # Cap enforcement: shift the cut back until this part fits under cap.
            while best_k > lo and cum[best_k] - cum[prev] > cap_words:
                best_k -= 1
            cuts.append(best_k)
            prev = best_k
        return cuts

    n_parts = min(n, max(1, math.ceil(total / cap_words)))
    ceil_parts = n_parts
    cuts = layout(n_parts)

    # If the final part still exceeds cap (and could be split further), bump once.
    boundaries = [0, *cuts, n]
    last_words = cum[boundaries[-1]] - cum[boundaries[-2]]
    if last_words > cap_words and n_parts < n and n_parts == ceil_parts:
        n_parts += 1
        cuts = layout(n_parts)
        boundaries = [0, *cuts, n]

    parts: list[Part] = []
    for idx in range(len(boundaries) - 1):
        start, end = boundaries[idx], boundaries[idx + 1]
        part_windows = windows[start:end]
        pwc = cum[end] - cum[start]
        parts.append(Part(
            index=idx,
            windows=tuple(part_windows),
            word_count=pwc,
            est_minutes=round(pwc / wpm, 2),
        ))

    assert len(parts) <= ceil_parts + 1, (
        f"part count {len(parts)} exceeds ceil+1 {ceil_parts + 1}"
    )
    return parts


def plan_section(
    section: Section,
    *,
    window_words: int,
    max_part_minutes: float,
    wpm: int,
    limit_words: int | None = None,
) -> SectionPlan:
    """Plan TTS windows and parts for one section."""
    raw_windows = _build_windows(section.paragraphs, window_words, limit_words)

    windows: list[Window] = []
    for idx, (sents, btype) in enumerate(raw_windows):
        text = " ".join(sents)
        wc = _word_count(text)
        if btype == "scene_break":
            pause = PAUSE_SCENE_BREAK_S
        elif btype == "paragraph":
            pause = PAUSE_BETWEEN_PARAGRAPHS_S
        else:
            pause = PAUSE_BETWEEN_WINDOWS_S
        windows.append(Window(index=idx, text=text, word_count=wc, pause_after_s=pause))

    # Last window always gets 0.0
    if windows:
        last = windows[-1]
        windows[-1] = Window(
            index=last.index,
            text=last.text,
            word_count=last.word_count,
            pause_after_s=0.0,
        )

    total_words = sum(w.word_count for w in windows)

    cap_words = int(max_part_minutes * wpm)
    parts = _group_parts(windows, cap_words, wpm)

    return SectionPlan(
        section_id=section.id,
        spine_index=section.index,
        title=section.title,
        parts=tuple(parts),
        word_count=total_words,
    )


def plan_book(
    book: Book,
    *,
    window_words: int,
    max_part_minutes: float,
    wpm: int,
    section_filter: set[str] | None = None,
    limit_words: int | None = None,
) -> list[SectionPlan]:
    """Plan all narrated sections in spine order."""
    sections = book.narrated_sections()
    if section_filter is not None:
        sections = [s for s in sections if s.id in section_filter]
    return [
        plan_section(
            s,
            window_words=window_words,
            max_part_minutes=max_part_minutes,
            wpm=wpm,
            limit_words=limit_words,
        )
        for s in sections
    ]
