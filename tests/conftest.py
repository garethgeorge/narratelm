from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_EPUB = REPO_ROOT / "testdata" / "The Wonderful Wizard of Oz - L. Frank Baum.epub"


@pytest.fixture(scope="session")
def test_epub() -> Path:
    assert TEST_EPUB.exists(), f"test epub missing: {TEST_EPUB}"
    return TEST_EPUB


@pytest.fixture(scope="session")
def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def pytest_collection_modifyitems(config, items):
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    skip = pytest.mark.skip(reason="ffmpeg/ffprobe not on PATH")
    for item in items:
        if "ffmpeg" in item.keywords:
            item.add_marker(skip)
