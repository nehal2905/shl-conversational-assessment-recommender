"""Phase 4 — end-to-end through the FastAPI app."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_schema_valid():
    r = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "I need an assessment"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert body["recommendations"] == []
    assert body["end_of_conversation"] is False


def test_chat_recommend_end_to_end():
    r = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "Hiring a mid-level Java developer who works with stakeholders"}
            ]
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert 1 <= len(body["recommendations"]) <= 10
    for rec in body["recommendations"]:
        assert rec["url"].startswith("https://www.shl.com/")
        assert rec["name"]
        assert rec["test_type"]


def test_chat_history_ending_on_assistant_is_handled():
    r = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "Hiring a Java developer, mid-level"},
                {"role": "assistant", "content": "Here are some options."},
            ]
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"reply", "recommendations", "end_of_conversation"}


def test_chat_empty_content_hardening():
    r = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "   "}]},
    )
    assert r.status_code == 200
    assert r.json()["recommendations"] == []
