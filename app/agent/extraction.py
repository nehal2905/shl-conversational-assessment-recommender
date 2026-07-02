"""Deterministic slot extraction from the full conversation (ARCHITECTURE.md §7.5).

Used to re-derive role/seniority/skills every turn so multi-turn clarification
accumulates context even when the LLM focuses on the latest user message.
"""
from __future__ import annotations

import re
from typing import List, Optional

from app.agent.state import Analysis

_TEST_TYPE_KEYWORDS = {
    "P": ["personality", "behaviou?r", "opq", "motivation"],
    "A": ["ability", "aptitude", "cognitive", "reasoning", "numerical", "verbal", "inductive"],
    "K": ["knowledge", "skill", "technical", "coding test", "language test"],
    "C": ["competenc"],
    "B": ["situational judg", "biodata"],
    "S": ["simulation"],
    "E": ["assessment exercise", "in-tray", "in tray"],
    "D": ["360", "development"],
}

_SENIORITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("mid", re.compile(r"\b(mid[- ]level|mid level|intermediate)\b", re.I)),
    ("junior", re.compile(r"\b(junior|entry[- ]level|entry|intern|fresher|graduate)\b", re.I)),
    ("senior", re.compile(r"\b(senior|lead|principal|staff|manager|director|head)\b", re.I)),
]

_ROLE_HINTS = [
    "developer", "engineer", "programmer", "analyst", "manager", "designer",
    "administrator", "consultant", "accountant", "sales", "representative",
    "executive", "scientist", "architect", "technician", "specialist",
    "clerk", "officer", "nurse", "teacher", "agent", "supervisor",
]

_SKILL_HINTS = [
    "java", "python", "javascript", "typescript", "c++", "c#", ".net", "sql", "html", "css",
    "react", "angular", "node", "aws", "azure", "spring", "django", "excel",
    "salesforce", "sap", "linux", "docker", "kubernetes", "php", "ruby", "go",
    "stakeholder", "communication", "leadership", "customer",
    "rest", "api", "boot", "backend", "frontend", "microservices",
]

_SENIORITY_PREFIX_RE = re.compile(
    r"^(junior|senior|mid[- ]level|mid level|mid|entry[- ]level|entry level|"
    r"entry|intern|fresher|graduate|lead|principal|staff)\s+",
    re.I,
)

_ROLE_FILLER_PREFIX_RE = re.compile(
    r"^(?:(?:i(?:'m| am)|we(?:'re| are))\s+)?"
    r"(?:(?:hiring|looking for|seeking|need(?:ing)?)\s+(?:a|an|the)\s+)?",
    re.I,
)

_REFINE_RE = re.compile(r"\b(also|add|actually|instead|as well|too|include)\b", re.I)


def all_user_text(messages: List[dict]) -> str:
    """Concatenate every user message in chronological order."""
    parts = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
    return " ".join(p for p in parts if p).strip()


def last_user_text(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _clean_role_phrase(phrase: str) -> str:
    cleaned = phrase.strip()
    cleaned = _ROLE_FILLER_PREFIX_RE.sub("", cleaned)
    while True:
        nxt = _SENIORITY_PREFIX_RE.sub("", cleaned, count=1).strip()
        if nxt == cleaned:
            break
        cleaned = nxt
    return cleaned.strip()


def extract_role(convo: str) -> Optional[str]:
    text = convo.lower()
    best: Optional[str] = None
    for hint in _ROLE_HINTS:
        pattern = rf"((?:[\w-]+\s+)*){re.escape(hint)}\b"
        for m in re.finditer(pattern, text):
            phrase = _clean_role_phrase(m.group(0).strip())
            if phrase and (best is None or len(phrase) > len(best)):
                best = phrase
    return best


def extract_seniority(text: str) -> Optional[str]:
    for level, pattern in _SENIORITY_PATTERNS:
        if pattern.search(text):
            return level
    m = re.search(r"(\d+)\s*(?:\+)?\s*years?", text, re.I)
    if m:
        yrs = int(m.group(1))
        return "junior" if yrs <= 2 else ("mid" if yrs <= 6 else "senior")
    return None


def extract_slots_from_text(convo: str) -> dict:
    """Extract hiring slots from concatenated user text."""
    text = (convo or "").lower()
    test_types_wanted: List[str] = []
    for tt, kws in _TEST_TYPE_KEYWORDS.items():
        if any(re.search(kw, text) for kw in kws):
            test_types_wanted.append(tt)

    skills = [s for s in _SKILL_HINTS if re.search(rf"\b{re.escape(s)}\b", text)]
    if re.search(r"\bspring\s+boot\b", text):
        if "spring" not in skills:
            skills.append("spring")

    seniority = extract_seniority(text)

    role = extract_role(text)
    remote_required = True if "remote" in text else None

    return {
        "role": role,
        "seniority": seniority,
        "skills": skills,
        "test_types_wanted": test_types_wanted,
        "remote_required": remote_required,
    }


def extract_slots_from_messages(messages: List[dict]) -> dict:
    return extract_slots_from_text(all_user_text(messages))


def has_enough_context(
    role: Optional[str],
    seniority: Optional[str],
    skills: List[str],
    test_types_wanted: List[str],
    constraints: List[str],
) -> bool:
    if role and (seniority or skills or test_types_wanted or constraints):
        return True
    return bool(skills or test_types_wanted)


def is_refine_turn(messages: List[dict]) -> bool:
    last = last_user_text(messages).lower()
    if not _REFINE_RE.search(last):
        return False
    slots = extract_slots_from_text(all_user_text(messages))
    return bool(slots["test_types_wanted"] or slots["skills"])


def _merge_unique(a: List[str], b: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in a + b:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def enrich_analysis(analysis: Analysis, messages: List[dict]) -> Analysis:
    """Merge LLM analysis with slots re-derived from the full conversation."""
    if analysis.intent == "off_topic":
        return analysis
    if analysis.intent == "compare" and analysis.compare_targets:
        return analysis

    slots = extract_slots_from_messages(messages)
    role = analysis.role or slots["role"]
    seniority = analysis.seniority or slots["seniority"]
    skills = _merge_unique(analysis.skills, slots["skills"])
    test_types_wanted = _merge_unique(analysis.test_types_wanted, slots["test_types_wanted"])
    remote_required = (
        analysis.remote_required
        if analysis.remote_required is not None
        else slots["remote_required"]
    )

    updates: dict = {
        "role": role,
        "seniority": seniority,
        "skills": skills,
        "test_types_wanted": test_types_wanted,
        "remote_required": remote_required,
    }

    prior_assistant = sum(1 for m in messages if m.get("role") == "assistant")
    enough = has_enough_context(
        role, seniority, skills, test_types_wanted, analysis.constraints
    )

    if is_refine_turn(messages) and prior_assistant >= 1:
        updates["intent"] = "refine"
        updates["ready_to_recommend"] = True
    elif enough:
        updates["intent"] = "recommend"
        updates["ready_to_recommend"] = True

    return analysis.model_copy(update=updates)
