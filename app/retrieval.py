"""Hybrid retriever: BM25 (sparse) + FAISS (dense) with normalized score fusion,
cosine reranking, keyword boosts, and technical-type preference."""
from __future__ import annotations

import pickle
import re
from typing import Dict, List, Optional, Set, Tuple

from app import config
from app.catalog import CatalogEntry

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "at", "by",
        "with", "from", "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "can", "need", "needs", "want", "wants", "use", "using",
        "what", "which", "who", "how", "when", "where", "why", "this", "that",
        "these", "those", "i", "we", "you", "your", "our", "their", "they",
        "he", "she", "it", "its", "my", "me", "us", "them", "hire", "hiring",
        "assessment", "assessments", "test", "tests", "recommend", "please",
        "yes", "remote", "level", "years", "year", "about", "around", "some",
        "kind", "also", "add", "actually", "too", "well", "not", "but", "any",
        "all", "both", "own", "owning", "across", "required", "strong", "keep",
        "drop", "here", "there", "can", "fill", "role", "job", "battery",
        "end", "delivery", "design", "deployment", "service", "relational",
        "databases", "senior", "full", "stack", "core", "will", "contribute",
        "decisions", "mentor", "mentors", "reports", "direct", "cloud", "native",
        "experience", "ci", "cd", "native", "own", "microservice",
    }
)

_ROLE_TERMS = frozenset(
    {
        "developer", "engineer", "programmer", "analyst", "manager", "designer",
        "administrator", "consultant", "accountant", "sales", "representative",
        "executive", "scientist", "architect", "technician", "specialist",
        "clerk", "officer", "nurse", "teacher", "agent", "supervisor", "operator",
    }
)

_TECH_TERMS = frozenset(
    {
        "java", "python", "javascript", "typescript", "rust", "c++", "c#", "csharp",
        "sql", "html", "css", "react", "angular", "node", "nodejs", "aws", "azure",
        "spring", "django", "excel", "salesforce", "sap", "linux", "docker",
        "kubernetes", "php", "ruby", "go", "golang", "net", ".net", "jsp", "xml",
        "jdbc", "microservice", "microservices", "ci/cd", "cicd", "networking",
        "network", "backend", "frontend", "fullstack", "full-stack", "rest", "api",
        "verify", "opq", "springboot", "hibernate", "postgres", "mysql", "oracle",
        "windows", "cisco", "safety", "financial", "accounting", "bookkeeping",
        "customer", "service", "contact", "centre", "center", "retail", "cashier",
        "office", "admin", "healthcare", "bilingual", "spanish", "graduate",
        "leadership", "management", "plant", "chemical", "petrochemical",
    }
)

_TECH_TYPES = frozenset({"K", "S", "C"})
_BROAD_TYPES = frozenset({"A", "P", "B", "D", "E"})
_PERSONALITY_QUERY_TERMS = frozenset({"personality", "opq", "behaviour", "behavior"})
_APTITUDE_QUERY_TERMS = frozenset(
    {
        "cognitive", "ability", "aptitude", "reasoning", "verify", "numerical",
        "verbal", "inductive", "gsa", "g+", "general ability",
    }
)


def _text_has_term(text: str, term: str) -> bool:
    # Short tokens like "java" must not match inside "javascript".
    if len(term) <= 4:
        return bool(re.search(rf"\b{re.escape(term)}\b", text))
    return bool(re.search(rf"\b{re.escape(term)}", text))


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer shared by index build and query time."""
    return re.findall(r"[a-z0-9\+\#\.]+", (text or "").lower())


def doc_text(entry: CatalogEntry) -> str:
    """Text used for BM25 (name + description)."""
    return f"{entry.name} {entry.description}"


def embed_text(entry: CatalogEntry) -> str:
    """Text used for dense embedding (name + '. ' + description)."""
    return f"{entry.name}. {entry.description}"


def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi <= lo:
        return {cid: 1.0 for cid in scores}
    span = hi - lo
    return {cid: (val - lo) / span for cid, val in scores.items()}


class HybridRetriever:
    def __init__(self, entries, faiss_index, bm25, ids: List[str], embedder):
        self.entries = entries
        self.by_id = {e.id: e for e in entries}
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.ids = ids  # row order → catalog id
        self.embedder = embedder

    # -- retrievers ---------------------------------------------------------
    def _dense_scored_ids(self, query: str, topn: int) -> Dict[str, float]:
        if self.faiss_index is None or self.embedder is None:
            return {}
        import numpy as np

        vec = next(iter(self.embedder.embed([query])))
        vec = np.asarray(vec, dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vec = vec.reshape(1, -1)
        k = min(topn, len(self.ids))
        scores, idxs = self.faiss_index.search(vec, k)
        out: Dict[str, float] = {}
        for i, score in zip(idxs[0], scores[0]):
            if 0 <= i < len(self.ids):
                out[self.ids[i]] = float(score)
        return out

    def _sparse_scored_ids(self, query: str, topn: int) -> Dict[str, float]:
        if self.bm25 is None:
            return {}
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: Dict[str, float] = {}
        for i in ranked[:topn]:
            if 0 <= i < len(self.ids):
                out[self.ids[i]] = float(scores[i])
        return out

    def _cosine_scores(self, query: str, ids: List[str]) -> Dict[str, float]:
        if not ids or self.embedder is None:
            return {}
        import numpy as np

        texts = [embed_text(self.by_id[cid]) for cid in ids if cid in self.by_id]
        if not texts:
            return {}
        vecs = list(self.embedder.embed([query] + texts))
        q = np.asarray(vecs[0], dtype="float32")
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        out: Dict[str, float] = {}
        for cid, doc_vec in zip(ids, vecs[1:]):
            d = np.asarray(doc_vec, dtype="float32")
            d_norm = np.linalg.norm(d)
            if d_norm > 0:
                d = d / d_norm
            out[cid] = float(np.dot(q, d))
        return out

    def _extract_keywords(self, query: str) -> Tuple[Set[str], Set[str]]:
        q = query.lower()
        tokens = tokenize(q)
        tech: Set[str] = set()
        role: Set[str] = set()

        for tok in tokens:
            if tok in _ROLE_TERMS:
                role.add(tok)
            if tok in _TECH_TERMS:
                tech.add(tok)
            elif len(tok) >= 4 and tok not in _STOPWORDS:
                tech.add(tok)

        for term in _ROLE_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", q):
                role.add(term)

        for term in _TECH_TERMS:
            if term in q:
                tech.add(term)

        # Pull technology tokens from quoted JD blocks (common in full job descriptions).
        for block in re.findall(r'"([^"]+)"', query):
            for tok in tokenize(block.lower()):
                if tok in _TECH_TERMS or (len(tok) >= 3 and tok not in _STOPWORDS):
                    tech.add(tok)

        for block in re.findall(r'"([^"]+)"', query):
            for tok in tokenize(block.lower()):
                if tok in _TECH_TERMS:
                    tech.add(tok)
                elif len(tok) >= 4 and tok not in _STOPWORDS:
                    tech.add(tok)

        return tech, role

    def _keyword_boost(
        self, entry: CatalogEntry, tech_kws: Set[str], role_kws: Set[str]
    ) -> float:
        name = entry.name.lower()
        desc = entry.description.lower()
        boost = 0.0

        for kw in tech_kws:
            if _text_has_term(name, kw):
                boost += config.KEYWORD_NAME_BOOST
            elif _text_has_term(desc, kw):
                boost += config.KEYWORD_DESC_BOOST

        for kw in role_kws:
            if _text_has_term(name, kw):
                boost += config.ROLE_KEYWORD_BOOST * 1.2
            elif _text_has_term(desc, kw):
                boost += config.ROLE_KEYWORD_BOOST

        return boost

    def _requested_type_boost(
        self,
        entry: CatalogEntry,
        boost_test_types: Optional[List[str]],
        query: str,
    ) -> float:
        if not boost_test_types:
            return 0.0
        types = set(entry.test_types)
        boost = set(boost_test_types)
        if not types.intersection(boost):
            return 0.0
        bonus = config.REQUESTED_TYPE_BOOST
        q = query.lower()
        if "P" in boost and "P" in types:
            if any(term in q for term in _PERSONALITY_QUERY_TERMS):
                bonus += config.PERSONALITY_QUERY_BOOST
        return bonus

    def _aptitude_boost(self, entry: CatalogEntry, query: str) -> float:
        q = query.lower()
        if not any(term in q for term in _APTITUDE_QUERY_TERMS):
            return 0.0
        if "A" not in entry.test_types:
            return 0.0
        name = entry.name.lower()
        if "verify" in name or "general ability" in name:
            return config.APTITUDE_VERIFY_BOOST
        return config.APTITUDE_QUERY_BOOST

    def _promote_keyword_matches(
        self,
        pool_ids: List[str],
        fused: Dict[str, float],
        tech_kws: Set[str],
        role_kws: Set[str],
    ) -> List[str]:
        """Ensure strong name keyword hits enter the rerank pool."""
        in_pool = set(pool_ids)
        promoted: List[Tuple[str, float]] = []
        for cid, entry in self.by_id.items():
            if cid in in_pool:
                continue
            kb = self._keyword_boost(entry, tech_kws, role_kws)
            if kb >= config.KEYWORD_NAME_BOOST:
                promoted.append((cid, fused.get(cid, 0.0) + kb))
        promoted.sort(key=lambda kv: kv[1], reverse=True)
        extra = [cid for cid, _ in promoted[: config.KEYWORD_PROMOTE_SLOTS]]
        return pool_ids + extra

    def _promote_sparse_hits(
        self, pool_ids: List[str], sparse: Dict[str, float], min_score: float = 8.0
    ) -> List[str]:
        """Keep strong BM25-only matches in the rerank pool."""
        in_pool = set(pool_ids)
        extras = [
            cid
            for cid, score in sorted(sparse.items(), key=lambda kv: kv[1], reverse=True)
            if cid not in in_pool and score >= min_score
        ]
        return pool_ids + extras[: config.KEYWORD_PROMOTE_SLOTS]

    def _promote_aptitude_hits(
        self, pool_ids: List[str], sparse: Dict[str, float], query: str
    ) -> List[str]:
        """When cognitive/ability is requested, keep Verify-style products in the pool."""
        q = query.lower()
        if not any(term in q for term in _APTITUDE_QUERY_TERMS):
            return pool_ids
        in_pool = set(pool_ids)
        extras: List[str] = []
        for cid, _score in sorted(sparse.items(), key=lambda kv: kv[1], reverse=True):
            if cid in in_pool:
                continue
            entry = self.by_id.get(cid)
            if entry is None or "A" not in entry.test_types:
                continue
            name = entry.name.lower()
            if "verify" in name or "general ability" in name:
                extras.append(cid)
        return pool_ids + extras[: config.KEYWORD_PROMOTE_SLOTS]

    def _type_adjustment(
        self,
        entry: CatalogEntry,
        boost_test_types: Optional[List[str]],
        has_technical_intent: bool,
        query: str,
    ) -> float:
        types = set(entry.test_types)
        adj = 0.0
        boost = set(boost_test_types or [])
        q = query.lower()
        personality_requested = bool(
            boost.intersection({"P"})
            and any(term in q for term in _PERSONALITY_QUERY_TERMS)
        )

        if not has_technical_intent or personality_requested:
            return adj

        if types.intersection(_TECH_TYPES):
            adj += config.TECH_TYPE_BOOST

        is_broad_only = bool(types) and not types.intersection(_TECH_TYPES)
        if is_broad_only and not (boost and types.intersection(boost)):
            adj -= config.BROAD_TYPE_PENALTY

        if entry.id.endswith("-solution") and not types.intersection({"K", "S", "C"}):
            adj -= config.SOLUTION_BUNDLE_PENALTY

        return adj

    def _has_technical_intent(
        self, query: str, tech_kws: Set[str], boost_test_types: Optional[List[str]]
    ) -> bool:
        q = query.lower()
        if any(term in q for term in _APTITUDE_QUERY_TERMS):
            return False
        if tech_kws:
            return True
        if boost_test_types and set(boost_test_types).intersection(_TECH_TYPES | {"K"}):
            return True
        return any(re.search(rf"\b{re.escape(term)}\b", q) for term in _TECH_TERMS)

    # -- fusion + rerank ----------------------------------------------------
    def search(
        self,
        query: str,
        k: int = config.RERANK_TOPK,
        boost_test_types: Optional[List[str]] = None,
    ) -> List[CatalogEntry]:
        dense = self._dense_scored_ids(query, config.DENSE_TOPN)
        sparse = self._sparse_scored_ids(query, config.SPARSE_TOPN)
        candidates = set(dense) | set(sparse)
        if not candidates:
            return []

        dense_norm = _minmax_normalize({cid: dense.get(cid, 0.0) for cid in candidates})
        sparse_norm = _minmax_normalize({cid: sparse.get(cid, 0.0) for cid in candidates})

        fused: Dict[str, float] = {}
        for cid in candidates:
            fused[cid] = (
                config.DENSE_FUSION_WEIGHT * dense_norm.get(cid, 0.0)
                + config.SPARSE_FUSION_WEIGHT * sparse_norm.get(cid, 0.0)
            )

        top_pool = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[
            : config.FUSION_TOPN
        ]
        pool_ids = [cid for cid, _ in top_pool]
        fused_pool = {cid: score for cid, score in top_pool}

        tech_kws, role_kws = self._extract_keywords(query)
        pool_ids = self._promote_keyword_matches(pool_ids, fused, tech_kws, role_kws)
        pool_ids = self._promote_sparse_hits(pool_ids, sparse)
        pool_ids = self._promote_aptitude_hits(pool_ids, sparse, query)
        for cid in pool_ids:
            if cid not in fused_pool:
                fused_pool[cid] = fused.get(cid, 0.0)

        cosine = self._cosine_scores(query, pool_ids)
        fused_norm = _minmax_normalize(fused_pool)
        cosine_norm = _minmax_normalize(cosine)

        has_technical = self._has_technical_intent(query, tech_kws, boost_test_types)

        final: Dict[str, float] = {}
        for cid in pool_ids:
            score = (
                config.RERANK_FUSION_WEIGHT * fused_norm.get(cid, 0.0)
                + config.RERANK_COSINE_WEIGHT * cosine_norm.get(cid, 0.0)
            )
            entry = self.by_id.get(cid)
            if entry is None:
                continue
            score += self._keyword_boost(entry, tech_kws, role_kws)
            score += self._requested_type_boost(entry, boost_test_types, query)
            score += self._aptitude_boost(entry, query)
            score += self._type_adjustment(
                entry, boost_test_types, has_technical, query
            )
            final[cid] = score

        ordered = sorted(final.items(), key=lambda kv: kv[1], reverse=True)
        out: List[CatalogEntry] = []
        seen: set[str] = set()
        for cid, _score in ordered:
            if cid in seen:
                continue
            entry = self.by_id.get(cid)
            if entry is None:
                continue
            seen.add(cid)
            out.append(entry)
            if len(out) >= k:
                break
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_retriever() -> HybridRetriever:
    """Load persisted FAISS + BM25 + ids and construct a retriever."""
    import json

    import faiss
    from fastembed import TextEmbedding

    from app.catalog import get_catalog

    entries = get_catalog()

    faiss_index = faiss.read_index(str(config.FAISS_PATH))
    with open(config.BM25_PATH, "rb") as fh:
        bm25 = pickle.load(fh)
    with open(config.IDS_PATH, "r", encoding="utf-8") as fh:
        ids = json.load(fh)

    embedder = TextEmbedding(model_name=config.EMBED_MODEL)
    return HybridRetriever(entries, faiss_index, bm25, ids, embedder)
