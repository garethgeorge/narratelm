"""Minimal, dependency-light EPUB2/3 reader (zipfile + lxml.etree).

Deliberately not a general EPUB library: it extracts exactly what the pipeline
needs — metadata, the linear spine, human titles from the nav/ncx, and raw
document bytes — while staying tolerant of missing pieces (no ncx, no cover).
"""

from __future__ import annotations

import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from narratelm.bookjson import Metadata

# Namespaces used across container.xml, the OPF, nav docs and the ncx.
_NS = {
    "cnt": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "epub": "http://www.idpf.org/2007/ops",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
}

_XHTML_MEDIA = {"application/xhtml+xml"}


@dataclass
class SpineItem:
    """A linear reading-order document: manifest id, OPF-relative href, position."""

    id: str
    href: str
    index: int


def _local(tag: object) -> str:
    """Local (namespace-stripped) tag name, or '' for comments/PIs."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _norm_href(base_dir: str, href: str) -> str:
    """Resolve *href* (relative to *base_dir*) to a fragment-free zip path."""
    href = href.split("#", 1)[0]
    if not href:
        return ""
    return posixpath.normpath(posixpath.join(base_dir, href))


class Epub:
    """An opened EPUB. Construct via :meth:`open`."""

    def __init__(
        self,
        zf: zipfile.ZipFile,
        opf_dir: str,
        metadata: Metadata,
        spine: list[SpineItem],
        toc_titles: dict[str, str],
        nav_href: str | None,
    ) -> None:
        self._zf = zf
        self._opf_dir = opf_dir
        self._metadata = metadata
        self._spine = spine
        self._toc_titles = toc_titles
        self._nav_href = nav_href

    # -- construction ------------------------------------------------------

    @classmethod
    def open(cls, path: Path) -> "Epub":
        zf = zipfile.ZipFile(path)
        opf_path = cls._find_opf(zf)
        opf_dir = posixpath.dirname(opf_path)
        opf = etree.fromstring(zf.read(opf_path))

        manifest = cls._parse_manifest(opf)  # id -> {href, media, properties}
        metadata = cls._parse_metadata(opf, manifest, opf_dir)
        spine = cls._parse_spine(opf, manifest)

        nav_id = next(
            (i for i, it in manifest.items() if "nav" in it["properties"].split()),
            None,
        )
        nav_href = manifest[nav_id]["href"] if nav_id else None
        toc_titles = cls._parse_toc_titles(zf, opf_dir, manifest, opf, nav_id)

        return cls(zf, opf_dir, metadata, spine, toc_titles, nav_href)

    @staticmethod
    def _find_opf(zf: zipfile.ZipFile) -> str:
        root = etree.fromstring(zf.read("META-INF/container.xml"))
        el = root.find(".//cnt:rootfile", _NS)
        if el is None or not el.get("full-path"):
            raise ValueError("EPUB container.xml has no rootfile full-path")
        return el.get("full-path")

    @staticmethod
    def _parse_manifest(opf: etree._Element) -> dict[str, dict]:
        manifest: dict[str, dict] = {}
        for item in opf.findall(".//opf:manifest/opf:item", _NS):
            iid = item.get("id")
            if not iid:
                continue
            manifest[iid] = {
                "href": item.get("href", ""),
                "media": item.get("media-type", ""),
                "properties": item.get("properties", ""),
            }
        return manifest

    @classmethod
    def _parse_metadata(
        cls, opf: etree._Element, manifest: dict[str, dict], opf_dir: str
    ) -> Metadata:
        def text(tag: str) -> str:
            el = opf.find(f".//dc:{tag}", _NS)
            return (el.text or "").strip() if el is not None else ""

        cover_href = cls._find_cover(opf, manifest, opf_dir)
        return Metadata(
            title=text("title"),
            author=text("creator"),
            language=text("language"),
            cover_href=cover_href,
        )

    @staticmethod
    def _find_cover(
        opf: etree._Element, manifest: dict[str, dict], opf_dir: str
    ) -> str | None:
        cover_id: str | None = None
        # EPUB2 convention: <meta name="cover" content="ID"/>.
        for meta in opf.findall(".//opf:metadata/opf:meta", _NS):
            if meta.get("name") == "cover" and meta.get("content"):
                cover_id = meta.get("content")
                break
        # EPUB3 convention: manifest item with properties="cover-image".
        if cover_id is None or cover_id not in manifest:
            cover_id = next(
                (i for i, it in manifest.items() if "cover-image" in it["properties"].split()),
                cover_id,
            )
        if cover_id and cover_id in manifest:
            return _norm_href(opf_dir, manifest[cover_id]["href"]) or None
        return None

    @staticmethod
    def _parse_spine(opf: etree._Element, manifest: dict[str, dict]) -> list[SpineItem]:
        spine: list[SpineItem] = []
        for ref in opf.findall(".//opf:spine/opf:itemref", _NS):
            if ref.get("linear", "yes").lower() == "no":
                continue
            idref = ref.get("idref")
            item = manifest.get(idref or "")
            if item is None or item["media"] not in _XHTML_MEDIA:
                continue
            spine.append(SpineItem(id=idref, href=item["href"], index=len(spine)))
        return spine

    @classmethod
    def _parse_toc_titles(
        cls,
        zf: zipfile.ZipFile,
        opf_dir: str,
        manifest: dict[str, dict],
        opf: etree._Element,
        nav_id: str | None,
    ) -> dict[str, str]:
        # Prefer the EPUB3 nav document.
        if nav_id:
            nav_path = _norm_href(opf_dir, manifest[nav_id]["href"])
            try:
                titles = cls._titles_from_nav(zf.read(nav_path), posixpath.dirname(nav_path))
                if titles:
                    return titles
            except (KeyError, etree.XMLSyntaxError):
                pass
        # Fall back to the ncx.
        ncx_id = opf.find(".//opf:spine", _NS)
        ncx_id = ncx_id.get("toc") if ncx_id is not None else None
        if not ncx_id or ncx_id not in manifest:
            ncx_id = next(
                (i for i, it in manifest.items() if "dtbncx" in it["media"]), None
            )
        if ncx_id and ncx_id in manifest:
            ncx_path = _norm_href(opf_dir, manifest[ncx_id]["href"])
            try:
                return cls._titles_from_ncx(zf.read(ncx_path), posixpath.dirname(ncx_path))
            except (KeyError, etree.XMLSyntaxError):
                pass
        return {}

    @staticmethod
    def _titles_from_nav(data: bytes, base_dir: str) -> dict[str, str]:
        root = etree.fromstring(data)
        # Locate <nav epub:type="toc"> (fall back to the first <nav>).
        navs = root.findall(".//xhtml:nav", _NS)
        toc_nav = next(
            (n for n in navs if "toc" in (n.get(f"{{{_NS['epub']}}}type") or "").split()),
            navs[0] if navs else None,
        )
        if toc_nav is None:
            return {}
        titles: dict[str, str] = {}
        for a in toc_nav.findall(".//xhtml:a", _NS):
            href = a.get("href")
            if not href:
                continue
            key = _norm_href(base_dir, href)
            label = " ".join("".join(a.itertext()).split())
            if key and label and key not in titles:
                titles[key] = label
        return titles

    @staticmethod
    def _titles_from_ncx(data: bytes, base_dir: str) -> dict[str, str]:
        root = etree.fromstring(data)
        titles: dict[str, str] = {}
        for point in root.findall(".//ncx:navPoint", _NS):
            content = point.find("ncx:content", _NS)
            label = point.find("ncx:navLabel/ncx:text", _NS)
            if content is None or label is None or not content.get("src"):
                continue
            key = _norm_href(base_dir, content.get("src"))
            text = " ".join((label.text or "").split())
            if key and text and key not in titles:
                titles[key] = text
        return titles

    # -- accessors ---------------------------------------------------------

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def spine(self) -> list[SpineItem]:
        return list(self._spine)

    @property
    def toc_titles(self) -> dict[str, str]:
        return dict(self._toc_titles)

    @property
    def nav_href(self) -> str | None:
        return self._nav_href

    def title_for(self, href: str) -> str | None:
        """Human TOC title for a spine href (OPF-relative), or None."""
        return self._toc_titles.get(_norm_href(self._opf_dir, href))

    def read_document(self, href: str) -> bytes:
        """Raw bytes of a spine document (*href* is relative to the OPF dir)."""
        return self._zf.read(_norm_href(self._opf_dir, href))

    def read_bytes(self, zip_path: str) -> bytes:
        """Raw bytes of an entry by full zip path (e.g. a stored cover_href)."""
        try:
            return self._zf.read(zip_path)
        except KeyError:
            return self._zf.read(_norm_href(self._opf_dir, zip_path))

    def close(self) -> None:
        self._zf.close()
