"""Process-wide lazy singletons (retriever). Kept separate so nodes and the
FastAPI app share one warmed instance without import cycles."""
from __future__ import annotations

import threading
from typing import Optional

from app.retrieval import HybridRetriever, load_retriever

_lock = threading.Lock()
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Lazily load and cache the hybrid retriever (thread-safe)."""
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = load_retriever()
    return _retriever


def warm() -> None:
    """Force-load the retriever (used by the startup background task)."""
    get_retriever()
