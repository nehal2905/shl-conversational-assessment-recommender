"""Phase 5 — grounding & guardrails: fabricated ids never reach the response."""
from __future__ import annotations

from app.catalog import get_catalog, get_id_index, is_catalog_url
from app.grounding import ground


def test_drops_unknown_ids():
    catalog = get_id_index()
    real = get_catalog()[0].id
    ids = ["totally-made-up-id", real, "another-fake", "hallucinated-999"]
    recs = ground(ids, catalog)
    assert len(recs) == 1
    assert recs[0].name == get_catalog()[0].name


def test_dedup_and_clamp():
    catalog = get_id_index()
    all_ids = [e.id for e in get_catalog()]
    # Duplicates + more than 10 → clamp to 10, no repeats.
    ids = all_ids[:12] + all_ids[:5]
    recs = ground(ids, catalog)
    assert len(recs) <= 10
    urls = [r.url for r in recs]
    assert len(urls) == len(set(urls))


def test_every_url_is_catalog_url():
    catalog = get_id_index()
    entries = get_catalog()
    ids = [e.id for e in entries[:5]]
    recs = ground(ids, catalog)
    for r in recs:
        assert is_catalog_url(r.url, entries)


def test_fuzz_fabricated_ids_never_emitted():
    catalog = get_id_index()
    entries = get_catalog()
    fabricated = [f"fake-{i}" for i in range(200)]
    recs = ground(fabricated, catalog)
    assert recs == []
    # Mixed: only the single real id survives.
    mixed = fabricated + [entries[3].id]
    recs = ground(mixed, catalog)
    assert len(recs) == 1
    assert is_catalog_url(recs[0].url, entries)
