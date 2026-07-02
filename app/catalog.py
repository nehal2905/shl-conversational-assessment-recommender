"""Catalog model + loading + defensive URL verification (ARCHITECTURE.md §5, §7.1)."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel

from app import config


class CatalogEntry(BaseModel):
    id: str  # slug derived from the detail URL, stable & unique
    name: str
    url: str  # canonical catalog detail URL
    test_types: List[str]  # e.g. ["K"] or ["C", "P"]
    remote_testing: bool = False
    adaptive_irt: bool = False
    description: str = ""  # from the detail page; drives retrieval + compare
    job_levels: List[str] = []
    languages: List[str] = []
    length_minutes: Optional[int] = None


def load_catalog(path: str | None = None) -> List[CatalogEntry]:
    """Load `catalog.json` into a list of `CatalogEntry`."""
    p = path or str(config.CATALOG_PATH)
    with open(p, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [CatalogEntry(**row) for row in raw]


def id_index(entries: List[CatalogEntry]) -> dict[str, CatalogEntry]:
    """Map primary key `id` → entry."""
    return {e.id: e for e in entries}


def is_catalog_url(url: str, entries: List[CatalogEntry]) -> bool:
    """Defensive check: only URLs that exist in the catalog are ever emitted."""
    if not url:
        return False
    return any(e.url == url for e in entries)


@lru_cache(maxsize=1)
def get_catalog() -> List[CatalogEntry]:
    """Process-wide cached catalog."""
    return load_catalog()


@lru_cache(maxsize=1)
def get_id_index() -> dict[str, CatalogEntry]:
    return id_index(get_catalog())
