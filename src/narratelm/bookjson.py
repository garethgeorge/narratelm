"""The book JSON: the user-editable single source of truth between extract and generate.

Written by `narratelm extract` next to the epub, hand-editable (flip `skip`,
fix `title`, edit `paragraphs`), then consumed by `generate` and `combine`.
The epub itself is only re-read for cover image bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1

# Paragraph sentinel marking a scene break: never spoken, rendered as a pause,
# and treated as a hard window boundary by chunking.
SCENE_BREAK = "***"


class BookJsonError(ValueError):
    """Raised when a book JSON fails validation (clear message for hand-edit typos)."""


@dataclass
class Metadata:
    title: str = ""
    author: str = ""
    language: str = ""
    cover_href: str | None = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "cover_href": self.cover_href,
        }

    @classmethod
    def from_dict(cls, d: object) -> "Metadata":
        if not isinstance(d, dict):
            raise BookJsonError(f"metadata must be an object, got {type(d).__name__}")
        return cls(
            title=str(d.get("title", "")),
            author=str(d.get("author", "")),
            language=str(d.get("language", "")),
            cover_href=d.get("cover_href"),
        )


@dataclass
class Section:
    id: str
    index: int  # spine position; canonical ordering key
    source_file: str
    title: str
    type: str  # chapter|cover|titlepage|toc|copyright|dedication|epigraph|glossary|promo|...
    skip: bool
    skip_reason: str | None
    word_count: int
    est_minutes: float
    paragraphs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "index": self.index,
            "source_file": self.source_file,
            "title": self.title,
            "type": self.type,
            "skip": self.skip,
            "skip_reason": self.skip_reason,
            "word_count": self.word_count,
            "est_minutes": self.est_minutes,
            "paragraphs": self.paragraphs,
        }

    @classmethod
    def from_dict(cls, d: object, pos: int) -> "Section":
        if not isinstance(d, dict):
            raise BookJsonError(f"sections[{pos}] must be an object")
        try:
            sec = cls(
                id=str(d["id"]),
                index=int(d["index"]),
                source_file=str(d.get("source_file", "")),
                title=str(d.get("title", "")),
                type=str(d.get("type", "chapter")),
                skip=bool(d.get("skip", False)),
                skip_reason=d.get("skip_reason"),
                word_count=int(d.get("word_count", 0)),
                est_minutes=float(d.get("est_minutes", 0.0)),
                paragraphs=list(d.get("paragraphs", [])),
            )
        except KeyError as e:
            raise BookJsonError(f"sections[{pos}] missing required field {e}") from e
        if not all(isinstance(p, str) for p in sec.paragraphs):
            raise BookJsonError(f"sections[{pos}].paragraphs must be a list of strings")
        return sec

    @property
    def spoken_paragraphs(self) -> list[str]:
        """Paragraphs that produce speech (scene-break sentinels excluded)."""
        return [p for p in self.paragraphs if p != SCENE_BREAK]


@dataclass
class Book:
    source_epub: str
    metadata: Metadata
    sections: list[Section]
    extractor_hash: str = ""
    schema_version: int = SCHEMA_VERSION

    def content_hash(self) -> str:
        """Hash of the machine-editable content; used to detect user edits."""
        payload = json.dumps(
            {
                "metadata": self.metadata.to_dict(),
                "sections": [s.to_dict() for s in self.sections],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_user_edited(self) -> bool:
        return bool(self.extractor_hash) and self.extractor_hash != self.content_hash()

    def narrated_sections(self) -> list[Section]:
        return [s for s in self.sections if not s.skip]

    def section_by_id(self, section_id: str) -> Section | None:
        for s in self.sections:
            if s.id == section_id:
                return s
        return None

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_epub": self.source_epub,
            "extractor_hash": self.extractor_hash,
            "metadata": self.metadata.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
        }

    @classmethod
    def from_dict(cls, d: object) -> "Book":
        if not isinstance(d, dict):
            raise BookJsonError("book JSON root must be an object")
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise BookJsonError(
                f"unsupported schema_version {version!r} (this narratelm supports {SCHEMA_VERSION})"
            )
        sections_raw = d.get("sections")
        if not isinstance(sections_raw, list) or not sections_raw:
            raise BookJsonError("book JSON must contain a non-empty 'sections' list")
        book = cls(
            source_epub=str(d.get("source_epub", "")),
            metadata=Metadata.from_dict(d.get("metadata", {})),
            sections=[Section.from_dict(s, i) for i, s in enumerate(sections_raw)],
            extractor_hash=str(d.get("extractor_hash", "")),
        )
        seen_ids = set()
        for s in book.sections:
            if s.id in seen_ids:
                raise BookJsonError(f"duplicate section id {s.id!r}")
            seen_ids.add(s.id)
        return book

    def save(self, path: Path, *, stamp_extractor_hash: bool = False) -> None:
        if stamp_extractor_hash:
            self.extractor_hash = self.content_hash()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Book":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise BookJsonError(f"{path}: invalid JSON ({e})") from e
        return cls.from_dict(raw)
