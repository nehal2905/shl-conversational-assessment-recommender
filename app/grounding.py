"""ids → Recommendation[], dedupe, clamp 1..10 (ARCHITECTURE.md §7.4, invariant 1)."""
from __future__ import annotations

from typing import Dict, List

from app import config
from app.catalog import CatalogEntry
from app.schemas import Recommendation


def ground(ids: List[str], catalog: Dict[str, CatalogEntry]) -> List[Recommendation]:
    """Map ids → entries, drop unknown ids, dedup, clamp to 1..10.

    ``test_type`` is the space-joined list of catalog codes. This is the only
    place assessment names/URLs enter the response, guaranteeing that every
    emitted item is a genuine catalog entry (invariant 1).
    """
    recs: List[Recommendation] = []
    seen: set[str] = set()
    for cid in ids:
        entry = catalog.get(cid)
        if entry is None or cid in seen:
            continue
        seen.add(cid)
        recs.append(
            Recommendation(
                name=entry.name,
                url=entry.url,
                test_type=" ".join(entry.test_types),
            )
        )
        if len(recs) >= config.MAX_RECS:
            break
    return recs
