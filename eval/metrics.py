"""Recall@10 metric (ARCHITECTURE.md §6 Phase 6)."""
from __future__ import annotations

from typing import List


def recall_at_k(relevant_ids: List[str], returned_ids: List[str], k: int = 10) -> float:
    """Recall@k = |relevant ∩ top-k| / |relevant|."""
    if not relevant_ids:
        return 0.0
    top = set(returned_ids[:k])
    hits = sum(1 for r in set(relevant_ids) if r in top)
    return hits / len(set(relevant_ids))


def mean_recall_at_k(per_trace: List[float]) -> float:
    return sum(per_trace) / len(per_trace) if per_trace else 0.0
