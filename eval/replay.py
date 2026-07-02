"""Simulated-user replay harness (ARCHITECTURE.md §6 Phase 6).

Each trace is a JSON file in eval/traces/ shaped like:

    {
      "id": "java-mid",
      "persona": "Hiring manager filling a mid-level Java role",
      "initial_query": "I'm hiring a Java developer",
      "facts": {
        "seniority": "mid-level, around 4 years",
        "skills": "Java, works with stakeholders",
        "remote": "yes"
      },
      "relevant_ids": ["java-8-new", "opq32r", "verify-general-ability-screen-gsa"]
    }

The simulated user answers the agent's clarifying questions from ``facts`` (a
naive keyword match), says "no particular preference" for anything outside its
facts, and stops when a shortlist appears. Capped at 8 combined turns.

By default the simulated user is rule-based (no LLM) so replay runs offline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.graph import build_graph, run_graph  # noqa: E402
from eval.metrics import mean_recall_at_k, recall_at_k  # noqa: E402

TRACES_DIR = ROOT / "eval" / "traces"
MAX_TURNS = 8


def simulated_user_reply(question: str, facts: Dict[str, str]) -> str:
    """Answer the agent's clarifying question from the trace facts."""
    q = question.lower()
    hints = {
        "senior": "seniority",
        "level": "seniority",
        "experience": "seniority",
        "skill": "skills",
        "tech": "skills",
        "language": "skills",
        "remote": "remote",
        "role": "role",
    }
    for kw, key in hints.items():
        if kw in q and key in facts:
            return facts[key]
    # Fall back to dumping any remaining facts once.
    if facts:
        return "; ".join(f"{k}: {v}" for k, v in facts.items())
    return "No particular preference."


def run_trace(graph, trace: dict) -> List[str]:
    """Drive a conversation and return the returned recommendation ids."""
    from app.catalog import get_catalog

    url_to_id = {e.url: e.id for e in get_catalog()}

    messages: List[dict] = [{"role": "user", "content": trace["initial_query"]}]
    facts = dict(trace.get("facts", {}))

    returned_ids: List[str] = []
    turns = 1  # initial user turn
    while turns < MAX_TURNS:
        state = run_graph(graph, messages)
        reply = state.get("reply", "")
        recs = state.get("recommendations", [])
        turns += 1  # assistant turn
        messages.append({"role": "assistant", "content": reply})

        if recs:
            returned_ids = [url_to_id.get(r.url, "") for r in recs]
            returned_ids = [i for i in returned_ids if i]
            break

        # Agent asked something → simulated user answers.
        if turns >= MAX_TURNS:
            break
        answer = simulated_user_reply(reply, facts)
        messages.append({"role": "user", "content": answer})
        turns += 1

    return returned_ids


def main() -> None:
    graph = build_graph()
    trace_files = sorted(TRACES_DIR.glob("*.json"))
    if not trace_files:
        print(f"No traces found in {TRACES_DIR}. Drop trace JSON files there.")
        return

    per_trace_recall: List[float] = []
    print(f"Replaying {len(trace_files)} trace(s)\n" + "=" * 50)
    for tf in trace_files:
        trace = json.loads(tf.read_text(encoding="utf-8"))
        returned = run_trace(graph, trace)
        relevant = trace.get("relevant_ids", [])
        r = recall_at_k(relevant, returned, k=10)
        per_trace_recall.append(r)
        print(f"[{trace.get('id', tf.stem)}] recall@10={r:.3f} "
              f"returned={returned}")

    print("=" * 50)
    print(f"Mean Recall@10 over {len(per_trace_recall)} traces: "
          f"{mean_recall_at_k(per_trace_recall):.3f}")


if __name__ == "__main__":
    main()
