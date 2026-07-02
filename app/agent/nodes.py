"""LangGraph nodes: analyze, clarify, recommend, compare, refuse, format
(ARCHITECTURE.md §7.6)."""
from __future__ import annotations

from typing import List

from rapidfuzz import fuzz, process

from app import config, grounding, llm, prompts
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
    free_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            free_text = str(m.get("content", ""))
            break
    parts = [
        a.role or "",
        a.seniority or "",
        " ".join(a.skills),
        " ".join(a.constraints),
        free_text,
    ]
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
def analyze(state: GraphState) -> GraphState:
    messages = state["messages"]
    formatted = format_messages(messages)
    raw = llm.chat_json(prompts.ANALYZE_SYSTEM, prompts.analyze_user(formatted))

    try:
        analysis = Analysis(**raw)
    except Exception:
        # Defensive: if the LLM returns something off-shape, treat as vague.
        analysis = Analysis(
            intent="vague",
            clarifying_question="What role are you hiring for, and what seniority level?",
        )

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
    candidates = retriever.search(
        query, k=config.RERANK_TOPK, boost_test_types=a.test_types_wanted or None
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
