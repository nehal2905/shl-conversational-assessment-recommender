"""API schemas — LOCKED per ARCHITECTURE.md §3.

Do not change field names or types: the automated evaluator grades against this
exact contract and any deviation scores zero.
"""
from typing import List, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # SHL code(s), e.g. "K", "P", or "C P" if multiple


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False
