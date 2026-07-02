"""LangGraph nodes: analyze, clarify, recommend, compare, refuse, format
(ARCHITECTURE.md §7.6)."""
from __future__ import annotations

from typing import List

from rapidfuzz import fuzz, process

from app import config, grounding, llm, prompts
from app.agent.extraction import all_user_text, enrich_analysis
from app.agent.state import Analysis, GraphState
from app.catalog import CatalogEntry, get_catalog, get_id_index
from app.schemas import ChatResponse, Recommendation


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def format_messages(messages: List[dict]) -> str:
    lines = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = str(m.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _count_prior_assistant_turns(messages: List[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


def _requirement_summary(a: Analysis) -> str:
    parts: List[str] = []
    if a.role:
        parts.append(f"Role: {a.role}")
    if a.seniority:
        parts.append(f"Seniority: {a.seniority}")
    if a.skills:
        parts.append(f"Skills: {', '.join(a.skills)}")
    if a.test_types_wanted:
        parts.append(f"Wanted test types: {', '.join(a.test_types_wanted)}")
    if a.remote_required:
        parts.append("Remote testing required")
    if a.languages:
        parts.append(f"Languages: {', '.join(a.languages)}")
    if a.constraints:
        parts.append(f"Constraints: {', '.join(a.constraints)}")
    return "; ".join(parts) or "General assessment need"


def _search_query(a: Analysis, messages: List[dict]) -> str:
    user_text = all_user_text(messages)
    parts = [
        a.role or "",
        a.seniority or "",
        " ".join(a.skills),
        " ".join(a.constraints),
        user_text,
    ]
    return " ".join(p for p in parts if p).strip()


_PERSONALITY_QUERY_TERMS = frozenset(
    {"personality", "opq", "behaviour", "behavior", "motivation"}
)


def _types_in_recommendations(recs: List[Recommendation]) -> set[str]:
    found: set[str] = set()
    for rec in recs:
        found.update(rec.test_type.split())
    return found


def _score_type_match(entry: CatalogEntry, wanted: str, query: str) -> float:
    q = query.lower()
    hay = f"{entry.name} {entry.description}".lower()
    score = sum(1.0 for word in q.split() if len(word) > 2 and word in hay)
    if wanted == "P":
        if any(term in q for term in _PERSONALITY_QUERY_TERMS):
            score += 5.0
        if "personality" in hay or "opq" in hay:
            score += 3.0
    return score


def _pick_catalog_entry_for_type(
    wanted: str,
    candidates: List[CatalogEntry],
    exclude_ids: set[str],
    query: str,
    catalog: List[CatalogEntry],
) -> CatalogEntry | None:
    """Pick the best catalog entry for a requested test type (refine turns only)."""
    for entry in candidates:
        if wanted in entry.test_types and entry.id not in exclude_ids:
            return entry

    best: tuple[float, CatalogEntry] | None = None
    for entry in catalog:
        if wanted not in entry.test_types or entry.id in exclude_ids:
            continue
        score = _score_type_match(entry, wanted, query)
        if best is None or score > best[0]:
            best = (score, entry)
    if best is None:
        return None
    if best[0] > 0:
        return best[1]
    if wanted == "P" and any(term in query.lower() for term in _PERSONALITY_QUERY_TERMS):
        return best[1]
    return None


def _ensure_refine_type_coverage(
    recs: List[Recommendation],
    candidates: List[CatalogEntry],
    wanted_types: List[str],
    catalog_ids: dict[str, CatalogEntry],
    query: str,
) -> List[Recommendation]:
    """On refine, guarantee each newly requested test type appears in the shortlist."""
    if not wanted_types:
        return recs

    present = _types_in_recommendations(recs)
    missing = [t for t in wanted_types if t not in present]
    if not missing:
        return recs

    catalog = list(catalog_ids.values())
    used_urls = {r.url for r in recs}
    used_ids = {cid for cid, entry in catalog_ids.items() if entry.url in used_urls}

    injected: List[Recommendation] = []
    for wanted in missing:
        entry = _pick_catalog_entry_for_type(
            wanted, candidates, used_ids, query, catalog
        )
        if entry is None:
            continue
        used_ids.add(entry.id)
        injected.extend(grounding.ground([entry.id], catalog_ids))

    if not injected:
        return recs

    merged: List[Recommendation] = []
    seen_urls: set[str] = set()
    for rec in injected + recs:
        if rec.url in seen_urls:
            continue
        seen_urls.add(rec.url)
        merged.append(rec)
        if len(merged) >= config.MAX_RECS:
            break
    return merged


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
def _normalize_analysis_raw(raw: dict) -> dict:
    """Coerce common LLM JSON shape mismatches before Pydantic validation."""
    out = dict(raw)
    for key in ("skills", "test_types_wanted", "languages", "constraints", "compare_targets"):
        val = out.get(key)
        if val is None:
            out[key] = []
        elif isinstance(val, str):
            out[key] = [s.strip() for s in val.split(",") if s.strip()]
    for key in ("ready_to_recommend", "remote_required"):
        val = out.get(key)
        if isinstance(val, str):
            out[key] = val.strip().lower() in ("true", "1", "yes")
    for key in ("role", "seniority", "clarifying_question", "off_topic_reason"):
        val = out.get(key)
        if isinstance(val, str) and not val.strip():
            out[key] = None
    return out


def analyze(state: GraphState) -> GraphState:
    messages = state["messages"]
    formatted = format_messages(messages)
    raw = llm.chat_json(prompts.ANALYZE_SYSTEM, prompts.analyze_user(formatted))

    try:
        analysis = Analysis(**_normalize_analysis_raw(raw if isinstance(raw, dict) else {}))
    except Exception:
        # Defensive: if the LLM returns something off-shape, treat as vague.
        analysis = Analysis(
            intent="vague",
            clarifying_question="What role are you hiring for, and what seniority level?",
        )

    # Re-derive slots from the full conversation so earlier answers are not lost.
    analysis = enrich_analysis(analysis, messages)

    # Invariant 3: force-commit after 2 clarifications.
    prior = _count_prior_assistant_turns(messages)
    if prior >= config.MAX_CLARIFICATIONS and analysis.intent != "off_topic":
        analysis.ready_to_recommend = True
        if analysis.intent == "vague":
            analysis.intent = "recommend"

    state["analysis"] = analysis
    return state


def clarify(state: GraphState) -> GraphState:
    a: Analysis = state["analysis"]
    question = a.clarifying_question or (
        "Could you tell me the role and seniority you're hiring for?"
    )
    state["reply"] = question
    state["recommendations"] = []
    state["end_of_conversation"] = False
    return state


def recommend(state: GraphState) -> GraphState:
    from app.runtime import get_retriever

    a: Analysis = state["analysis"]
    messages = state["messages"]

    query = _search_query(a, messages)
    retriever = get_retriever()
    search_k = config.RERANK_TOPK
    if a.intent == "refine" and a.test_types_wanted:
        search_k = max(config.RERANK_TOPK, config.REFINE_CANDIDATE_TOPK)
    candidates = retriever.search(
        query, k=search_k, boost_test_types=a.test_types_wanted or None
    )
    state["candidates"] = candidates

    catalog_ids = get_id_index()
    recs: List[Recommendation] = []

    if candidates:
        raw = llm.chat_json(
            prompts.RERANK_SYSTEM,
            prompts.rerank_user(_requirement_summary(a), candidates),
        )
        ids = raw.get("ids", []) if isinstance(raw, dict) else []
        if not isinstance(ids, list):
            ids = []
        # Only ids present in the candidate set are trusted (defense in depth).
        candidate_ids = {c.id for c in candidates}
        ids = [i for i in ids if i in candidate_ids]
        if not ids:
            ids = [c.id for c in candidates]
        recs = grounding.ground(ids, catalog_ids)

    if a.intent == "refine" and a.test_types_wanted:
        recs = _ensure_refine_type_coverage(
            recs, candidates, a.test_types_wanted, catalog_ids, query
        )

    state["recommendations"] = recs
    if recs:
        verb = "refined shortlist" if a.intent == "refine" else "assessments"
        state["reply"] = (
            f"Here are {len(recs)} SHL {verb} that fit your requirement."
        )
    else:
        state["reply"] = (
            "I couldn't find matching assessments yet. Could you share the role or key skills?"
        )
    state["end_of_conversation"] = False
    return state


def compare(state: GraphState) -> GraphState:
    a: Analysis = state["analysis"]
    entries = _resolve_compare_targets(a.compare_targets)

    if len(entries) < 2:
        state["reply"] = (
            "I can only compare assessments in the SHL catalog. Please name two catalog "
            "assessments (for example, two you're considering) and I'll compare them."
        )
        state["recommendations"] = []
        state["end_of_conversation"] = False
        return state

    user_question = ""
    for m in reversed(state["messages"]):
        if m.get("role") == "user":
            user_question = str(m.get("content", ""))
            break

    text = llm.chat_text(prompts.COMPARE_SYSTEM, prompts.compare_user(user_question, entries))
    state["reply"] = text.strip() or "Here is a comparison based on the catalog descriptions."
    state["recommendations"] = []  # comparison is not a shortlist commitment
    state["end_of_conversation"] = False
    return state


def refuse(state: GraphState) -> GraphState:
    a: Analysis = state["analysis"]
    topic = a.off_topic_reason or "that"
    state["reply"] = prompts.refuse_reply(topic)
    state["recommendations"] = []
    state["end_of_conversation"] = False
    return state


def format_node(state: GraphState) -> GraphState:
    """Validate/coerce against ChatResponse; guarantees schema-valid output."""
    recs = state.get("recommendations") or []
    # Coerce any raw dicts and clamp to MAX_RECS.
    coerced: List[Recommendation] = []
    for r in recs[: config.MAX_RECS]:
        if isinstance(r, Recommendation):
            coerced.append(r)
        elif isinstance(r, dict):
            coerced.append(Recommendation(**r))
    resp = ChatResponse(
        reply=state.get("reply") or "",
        recommendations=coerced,
        end_of_conversation=bool(state.get("end_of_conversation", False)),
    )
    state["reply"] = resp.reply
    state["recommendations"] = resp.recommendations
    state["end_of_conversation"] = resp.end_of_conversation
    return state


# ---------------------------------------------------------------------------
# compare-target resolution via rapidfuzz
# ---------------------------------------------------------------------------
def _resolve_compare_targets(targets: List[str]) -> List[CatalogEntry]:
    catalog = get_catalog()
    names = [e.name for e in catalog]
    by_name = {e.name: e for e in catalog}
    resolved: List[CatalogEntry] = []
    seen: set[str] = set()
    for t in targets:
        if not t:
            continue
        match = process.extractOne(t, names, scorer=fuzz.WRatio)
        if match and match[1] >= 60:
            entry = by_name[match[0]]
            if entry.id not in seen:
                seen.add(entry.id)
                resolved.append(entry)
    return resolved
