"""Environment, paths, and tuning constants (ARCHITECTURE.md §11)."""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# .env loading (tiny, dependency-free — avoids requiring python-dotenv)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        # Do not clobber values already present in the real environment.
        os.environ.setdefault(key, val)


_load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
GROQ_API_KEY: str | None = os.environ.get("GROQ_API_KEY") or None
MODEL: str = os.environ.get("MODEL", "llama-3.3-70b-versatile")

# When no key is configured we fall back to a deterministic offline LLM so the
# app and its tests still run end-to-end. Real Groq is used whenever a key is set.
OFFLINE_LLM: bool = not bool(GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_DIR = DATA_DIR / "index"
FAISS_PATH = INDEX_DIR / "faiss.index"
BM25_PATH = INDEX_DIR / "bm25.pkl"
IDS_PATH = INDEX_DIR / "ids.json"

# ---------------------------------------------------------------------------
# Models / retrieval constants
# ---------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

RRF_K = 60  # retained for backward-compatible bonus scaling
DENSE_TOPN = 40
SPARSE_TOPN = 40
FUSION_TOPN = 20  # pool size before cosine rerank
KEYWORD_PROMOTE_SLOTS = 8  # extra strong keyword hits added to rerank pool
RERANK_TOPK = 15
BOOST_BONUS_RANK = 5

# Normalized hybrid fusion (dense + sparse)
DENSE_FUSION_WEIGHT = 0.5
SPARSE_FUSION_WEIGHT = 0.5

# Cosine rerank within the fusion top-N pool
RERANK_FUSION_WEIGHT = 0.35
RERANK_COSINE_WEIGHT = 0.65

# Keyword boosts from the job-description query
KEYWORD_NAME_BOOST = 0.14
KEYWORD_DESC_BOOST = 0.05
ROLE_KEYWORD_BOOST = 0.10

# Prefer specific technical (K/S/C) over broad aptitude/personality (A/P)
TECH_TYPE_BOOST = 0.08
BROAD_TYPE_PENALTY = 0.06
SOLUTION_BUNDLE_PENALTY = 0.04
REQUESTED_TYPE_BOOST = 0.35  # strong nudge when user asks for a test type (e.g. personality)
PERSONALITY_QUERY_BOOST = 0.12  # extra when query mentions personality/opq
APTITUDE_QUERY_BOOST = 0.10  # when query asks for cognitive/ability tests
APTITUDE_VERIFY_BOOST = 0.14  # named Verify / G+ products

MAX_CLARIFICATIONS = 2
MAX_RECS = 10
REFINE_CANDIDATE_TOPK = 40  # wider retrieval pool on refine so new test types can surface

DESC_TRUNC = 160  # chars of description shown to the LLM in rerank prompt

VALID_TEST_TYPES = {"A", "B", "C", "D", "E", "K", "P", "S"}
