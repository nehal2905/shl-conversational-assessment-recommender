"""LangGraph assembly + conditional edges (ARCHITECTURE.md §7.7)."""
from __future__ import annotations

from typing import List

from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.state import Analysis, GraphState


def route(state: GraphState) -> str:
    a: Analysis = state["analysis"]
    if a.intent == "off_topic":
        return "refuse"
    if a.intent == "compare" and a.compare_targets:
        return "compare"
    if a.ready_to_recommend or a.intent in ("recommend", "refine"):
        return "recommend"
    return "clarify"  # vague and not ready


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("analyze", nodes.analyze)
    g.add_node("clarify", nodes.clarify)
    g.add_node("recommend", nodes.recommend)
    g.add_node("compare", nodes.compare)
    g.add_node("refuse", nodes.refuse)
    g.add_node("format", nodes.format_node)

    g.add_edge(START, "analyze")
    g.add_conditional_edges(
        "analyze",
        route,
        {
            "clarify": "clarify",
            "recommend": "recommend",
            "compare": "compare",
            "refuse": "refuse",
        },
    )
    for node in ("clarify", "recommend", "compare", "refuse"):
        g.add_edge(node, "format")
    g.add_edge("format", END)
    return g.compile()


def run_graph(graph, messages: List[dict]) -> GraphState:
    """Run the compiled graph over a messages list, returning the final state."""
    init: GraphState = {
        "messages": messages,
        "analysis": None,
        "candidates": [],
        "recommendations": [],
        "reply": "",
        "end_of_conversation": False,
    }
    return graph.invoke(init)
