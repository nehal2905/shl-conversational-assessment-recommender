"""Unit tests for deterministic conversation slot extraction."""
from __future__ import annotations

from app.agent.extraction import (
    enrich_analysis,
    extract_role,
    extract_seniority,
    extract_slots_from_messages,
    has_enough_context,
)
from app.agent.state import Analysis

FULLY_SPECIFIED = (
    "I'm hiring a mid-level Java backend developer with Spring Boot, SQL, and REST APIs."
)


def test_extract_role_and_seniority_from_single_message():
    slots = extract_slots_from_messages([{"role": "user", "content": FULLY_SPECIFIED}])
    assert slots["role"] is not None
    assert "developer" in slots["role"]
    assert slots["seniority"] == "mid"
    assert "java" in slots["skills"]
    assert "sql" in slots["skills"]
    assert has_enough_context(
        slots["role"],
        slots["seniority"],
        slots["skills"],
        slots["test_types_wanted"],
        [],
    )


def test_extract_seniority_mid_level_hyphen():
    assert extract_seniority("mid-level engineer") == "mid"
    assert extract_seniority("senior architect") == "senior"


def test_extract_role_strips_seniority_prefix():
    role = extract_role("hiring a mid-level java backend developer")
    assert role == "java backend developer"


def test_enrich_analysis_upgrades_vague_llm_output():
    messages = [{"role": "user", "content": FULLY_SPECIFIED}]
    vague = Analysis(
        intent="vague",
        ready_to_recommend=False,
        clarifying_question="What role are you hiring for, and what seniority level?",
    )
    enriched = enrich_analysis(vague, messages)
    assert enriched.ready_to_recommend is True
    assert enriched.intent == "recommend"
    assert enriched.role is not None
    assert enriched.seniority == "mid"
