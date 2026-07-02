"""FastAPI app: /health, /chat (ARCHITECTURE.md §7.8, §3)."""
from __future__ import annotations

import logging

from fastapi import FastAPI

from app.agent.graph import build_graph, run_graph
from app.schemas import ChatRequest, ChatResponse, Recommendation

logger = logging.getLogger("shl-recommender")

app = FastAPI(title="SHL Assessment Recommender")

GRAPH = build_graph()  # compiled once at import


@app.on_event("startup")
async def _startup() -> None:
    """Warm the retriever in the background so /health stays instant."""
    import asyncio

    async def _warm() -> None:
        try:
            from app.runtime import warm

            await asyncio.to_thread(warm)
            logger.info("Retriever warmed.")
        except Exception as exc:  # pragma: no cover - warmup best effort
            logger.warning("Retriever warm-up failed (will lazy-load): %s", exc)

    asyncio.create_task(_warm())


@app.get("/health")
def health():
    return {"status": "ok"}  # instant, even before warm-up


def _sanitize_messages(req: ChatRequest) -> list[dict]:
    """Basic input hardening: drop empty/non-string content, ensure the history
    ends on a user turn so the agent has something to answer."""
    msgs = [
        {"role": m.role, "content": m.content}
        for m in req.messages
        if isinstance(m.content, str) and m.content.strip()
    ]
    # Trim trailing assistant turns so we always analyze the latest user input.
    while msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
    return msgs


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        messages = _sanitize_messages(req)
        if not messages:
            return ChatResponse(
                reply="What role are you hiring for? Tell me the role and seniority and I'll suggest SHL assessments.",
                recommendations=[],
                end_of_conversation=False,
            )
        state = run_graph(GRAPH, messages)
        recs = state.get("recommendations") or []
        recs = [r if isinstance(r, Recommendation) else Recommendation(**r) for r in recs]
        return ChatResponse(
            reply=state.get("reply") or "",
            recommendations=recs,
            end_of_conversation=bool(state.get("end_of_conversation", False)),
        )
    except Exception:  # invariant 5: always return valid schema
        logger.exception("Unhandled error in /chat")
        return ChatResponse(
            reply="Sorry — could you rephrase what role you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )
