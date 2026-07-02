"""Shared fixtures. Ensures the FAISS/BM25 index exists before retrieval tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402


def _index_exists() -> bool:
    return config.FAISS_PATH.exists() and config.BM25_PATH.exists() and config.IDS_PATH.exists()


@pytest.fixture(scope="session", autouse=True)
def _ensure_index():
    """Build the index once per test session if it is missing."""
    if not _index_exists():
        from scripts.build_index import main as build_main

        build_main()
    assert _index_exists(), "Index build failed"


@pytest.fixture(scope="session")
def graph():
    from app.agent.graph import build_graph

    return build_graph()
