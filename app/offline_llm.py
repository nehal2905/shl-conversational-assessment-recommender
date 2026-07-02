"""Deterministic offline LLM fallback (used only when GROQ_API_KEY is unset).

This is NOT a model. It is a small rule-based stand-in so the whole system runs
end-to-end (and unit tests pass) without network access or an API key. When a
Groq key is configured, ``app.llm`` bypasses this module entirely.

The heuristics intentionally mirror the *shape* of the real prompts:
  * analyze  → returns the ``Analysis`` JSON keys
  * rerank   → returns ``{"ids": [...]}`` preserving retrieval order
  * compare  → returns a short factual, description-grounded paragraph
"""
from __future__ import annotations

import re

# --- keyword tables ---------------------------------------------------------
_OFFTOPIC_PATTERNS = [
    r"\bignore (your |all |previous )?instructions\b",
    r"\bwrite (me )?a poem\b",
    r"\bpoem\b",
    r"\bsystem prompt\b",
    r"\breveal your\b",
    r"\blegal advice\b",
    r"\bfir(e|ing)\b",
    r"\bsalary\b",
    r"\bhow much should i pay\b",
    r"\bwrite (my |a )?(cover letter|resume|essay|story|song|code)\b",
    r"\bweather\b",
    r"\bjoke\b",
]

_COMPARE_PATTERNS = [
    r"\bdifference between\b",
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bwhich is better\b",
]

from app.agent.extraction import extract_slots_from_text, has_enough_context


def _count_assistant_turns(user_block: str) -> int:
    return sum(1 for ln in user_block.splitlines() if ln.strip().upper().startswith("ASSISTANT:"))


def _match_any(patterns, text) -> bool:
    return any(re.search(p, text) for p in patterns)


def _analyze(user_block: str) -> dict:
    convo = all_user_text_from_block(user_block).lower()
    last = _last_user_message(user_block).lower()
    prior_assistant = _count_assistant_turns(user_block)

    result: dict = {
        "intent": "vague",
        "role": None,
        "seniority": None,
        "skills": [],
        "test_types_wanted": [],
        "remote_required": None,
        "languages": [],
        "constraints": [],
        "compare_targets": [],
        "off_topic_reason": None,
        "ready_to_recommend": False,
        "clarifying_question": None,
    }

    # off-topic / injection ---------------------------------------------------
    if _match_any(_OFFTOPIC_PATTERNS, last):
        result["intent"] = "off_topic"
        result["off_topic_reason"] = "request is not about selecting SHL assessments"
        return result

    # compare -----------------------------------------------------------------
    if _match_any(_COMPARE_PATTERNS, last):
        targets = _extract_compare_targets(_last_user_message(user_block))
        if targets:
            result["intent"] = "compare"
            result["compare_targets"] = targets
            return result

    # extraction from full conversation ---------------------------------------
    slots = extract_slots_from_text(convo)
    result.update(slots)

    # refine vs recommend vs vague -------------------------------------------
    is_refine = bool(re.search(r"\b(also|add|actually|instead|as well|too|include)\b", last)) and (
        result["test_types_wanted"] or result["skills"]
    )

    if is_refine and prior_assistant >= 1:
        result["intent"] = "refine"
        result["ready_to_recommend"] = True
    elif has_enough_context(
        result["role"],
        result["seniority"],
        result["skills"],
        result["test_types_wanted"],
        result["constraints"],
    ):
        result["intent"] = "recommend"
        result["ready_to_recommend"] = True
    else:
        result["intent"] = "vague"
        result["ready_to_recommend"] = False
        result["clarifying_question"] = (
            "What role are you hiring for, and what seniority level?"
        )

    return result


def _last_user_message(user_block: str) -> str:
    """Extract the most recent USER line from a formatted conversation block."""
    lines = [ln.strip() for ln in user_block.splitlines()]
    user_lines = [ln[len("USER:"):].strip() for ln in lines if ln.upper().startswith("USER:")]
    if user_lines:
        return user_lines[-1]
    return user_block


def all_user_text_from_block(user_block: str) -> str:
    lines = [ln.strip() for ln in user_block.splitlines()]
    parts = [ln[len("USER:"):].strip() for ln in lines if ln.upper().startswith("USER:")]
    return " ".join(parts) if parts else user_block


def _extract_compare_targets(sentence: str) -> list[str]:
    """Pull candidate assessment names around 'between X and Y' / 'X vs Y'."""
    s = sentence
    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)[\?\.\!]?$", s, re.IGNORECASE)
    if m:
        return [_clean_target(m.group(1)), _clean_target(m.group(2))]
    m = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)[\?\.\!]?$", s, re.IGNORECASE)
    if m:
        return [_clean_target(m.group(1)), _clean_target(m.group(2))]
    m = re.search(r"compare\s+(.+?)\s+and\s+(.+?)[\?\.\!]?$", s, re.IGNORECASE)
    if m:
        return [_clean_target(m.group(1)), _clean_target(m.group(2))]
    return []


def _clean_target(t: str) -> str:
    t = re.sub(r"^(the|a|an)\s+", "", t.strip(), flags=re.IGNORECASE)
    return t.strip(" ?.!,")


def _rerank(user_block: str) -> dict:
    """Preserve the retrieval order (candidates arrive pre-ranked)."""
    ids = re.findall(r"id=([^\s|]+)", user_block)
    return {"ids": ids[:10] if ids else []}


def _compare(system: str, user: str) -> str:
    """Produce a short, description-grounded comparison from the prompt payload."""
    blocks = re.findall(
        r"Assessment [AB][^\n]*?—\s*(?P<name>[^:]+):\s*(?P<desc>.+?)\s*\(types:\s*(?P<types>[^,]*),",
        user,
        re.DOTALL,
    )
    if len(blocks) < 2:
        return (
            "I can only compare items in the SHL catalog, and I couldn't find both "
            "of those assessments. Could you name two catalog assessments to compare?"
        )
    a, b = blocks[0], blocks[1]

    def _short(desc: str) -> str:
        desc = desc.strip().replace("\n", " ")
        return (desc[:220] + "…") if len(desc) > 220 else desc

    return (
        f"Here's how they differ, based on the SHL catalog descriptions:\n\n"
        f"- {a[0].strip()} (type {a[2].strip()}): {_short(a[1])}\n"
        f"- {b[0].strip()} (type {b[2].strip()}): {_short(b[1])}\n\n"
        f"In short, choose {a[0].strip()} if you need what its description emphasizes, "
        f"and {b[0].strip()} for the focus described above."
    )


# --- public entry points ----------------------------------------------------
def offline_chat_json(system: str, user: str) -> dict:
    sys_l = system.lower()
    if "analysis module" in sys_l:
        return _analyze(user)
    if "select the best" in sys_l or '"ids"' in system or "candidate list" in sys_l:
        return _rerank(user)
    # Unknown JSON call — return empty object.
    return {}


def offline_chat_text(system: str, user: str) -> str:
    if "explain differences" in system.lower() or "Assessment A" in user:
        return _compare(system, user)
    return ""
