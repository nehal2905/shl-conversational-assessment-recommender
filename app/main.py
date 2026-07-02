"""FastAPI app: /, /health, /chat (ARCHITECTURE.md §7.8, §3)."""
from __future__ import annotations

import logging

from fastapi import FastAPI

from app.agent.graph import build_graph, run_graph
from app.schemas import ChatRequest, ChatResponse, Recommendation

logger = logging.getLogger("shl-recommender")

app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    version="1.0.0",
    description="Stateless conversational API for recommending SHL assessments."
)

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
        except Exception as exc:  # pragma: no cover
            logger.warning("Retriever warm-up failed (will lazy-load): %s", exc)

    asyncio.create_task(_warm())


@app.get("/")
def root():
    """Simple landing endpoint for deployed API."""
    return {
        "name": "SHL Conversational Assessment Recommender",
        "status": "running",
        "version": "1.0.0",
        "description": "Stateless conversational API for recommending SHL Individual Test Solutions.",
        "health": "/health",
        "docs": "/docs",
        "chat": "/chat",
    }


@app.get("/health")
def health():
    """Health endpoint."""
    return {"status": "ok"}


def _sanitize_messages(req: ChatRequest) -> list[dict]:
    """
    Drop empty/non-string messages and ensure the conversation ends
    on a user message.
    """
    msgs = [
        {"role": m.role, "content": m.content}
        for m in req.messages
        if isinstance(m.content, str) and m.content.strip()
    ]

    while msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()

    return msgs


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        messages = _sanitize_messages(req)

        if not messages:
            return ChatResponse(
                reply=(
                    "What role are you hiring for? "
                    "Tell me the role and seniority and I'll recommend suitable SHL assessments."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        state = run_graph(GRAPH, messages)

        recs = state.get("recommendations") or []
        recs = [
            r if isinstance(r, Recommendation) else Recommendation(**r)
            for r in recs
        ]

        return ChatResponse(
            reply=state.get("reply") or "",
            recommendations=recs,
            end_of_conversation=bool(
                state.get("end_of_conversation", False)
            ),
        )

    except Exception:
        logger.exception("Unhandled error in /chat")

        return ChatResponse(
            reply="Sorry — could you rephrase what role you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )