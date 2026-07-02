"""Analysis + GraphState (ARCHITECTURE.md §7.5)."""
from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from pydantic import BaseModel


class Analysis(BaseModel):
    intent: Literal["vague", "recommend", "refine", "compare", "off_topic"]
    role: Optional[str] = None
    seniority: Optional[str] = None
    skills: List[str] = []
    test_types_wanted: List[str] = []  # e.g. ["P"] when user asks for personality tests
    remote_required: Optional[bool] = None
    languages: List[str] = []
    constraints: List[str] = []
    compare_targets: List[str] = []  # assessment names to compare
    off_topic_reason: Optional[str] = None
    ready_to_recommend: bool = False
    clarifying_question: Optional[str] = None


class GraphState(TypedDict):
    messages: list  # list[dict]
    analysis: Optional[Analysis]
    candidates: list  # list[CatalogEntry]
    recommendations: list  # list[Recommendation]
    reply: str
    end_of_conversation: bool
