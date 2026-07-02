"""Convert C1.md–C10.md public conversation traces to eval/replay.py JSON format."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
TRACES_OUT = ROOT / "eval" / "traces"

# Hand-authored persona + facts derived from each trace's conversation.
TRACE_META: dict[str, dict] = {
    "c1": {
        "id": "c1-executive-leadership",
        "persona": "HR leader selecting OPQ-based leadership assessments for executive hiring",
        "facts": {
            "role": "senior leadership — CXOs and director-level executives",
            "seniority": "15+ years experience, CXO and director pool",
            "skills": "leadership selection with benchmark comparison",
            "constraints": "selection use case, not developmental feedback",
        },
    },
    "c2": {
        "id": "c2-senior-rust-engineer",
        "persona": "Engineering manager hiring a senior Rust/networking infrastructure engineer",
        "facts": {
            "role": "senior Rust engineer for high-performance networking infrastructure",
            "seniority": "senior individual contributor",
            "skills": "Rust, Linux systems, networking; wants cognitive ability test too",
            "remote": "yes, remote testing preferred",
        },
    },
    "c3": {
        "id": "c3-contact-centre-volume",
        "persona": "Recruiter screening 500 entry-level inbound contact-centre agents",
        "facts": {
            "role": "entry-level contact centre agent, inbound customer service",
            "seniority": "entry-level, high-volume hire (500 candidates)",
            "skills": "spoken English, call handling, customer service simulation",
            "languages": "English US accent for SVAR",
        },
    },
    "c4": {
        "id": "c4-graduate-financial-analyst",
        "persona": "Campus recruiter hiring final-year graduate financial analysts",
        "facts": {
            "role": "graduate financial analyst",
            "seniority": "final-year students, no work experience",
            "skills": "numerical reasoning, finance knowledge, situational judgement for graduates",
        },
    },
    "c5": {
        "id": "c5-sales-reskilling",
        "persona": "L&D lead running a sales organization re-skilling and talent audit",
        "facts": {
            "role": "sales organization audit and development",
            "seniority": "individual contributors and first-line sales managers",
            "skills": "self-reported skills (GSA), personality, sales-specific OPQ reports, digital selling",
            "constraints": "OPQ for everyone; MQ only where motivators needed in Sales Report",
        },
    },
    "c6": {
        "id": "c6-plant-operator-safety",
        "persona": "Plant HR manager hiring chemical-facility operators with safety-first culture",
        "facts": {
            "role": "plant operator in a chemical manufacturing facility",
            "seniority": "frontline operator",
            "skills": "safety behaviour, procedure compliance, dependability",
            "constraints": "industrial facility — prefer Manufacturing & Industrial Safety & Dependability 8.0",
        },
    },
    "c7": {
        "id": "c7-bilingual-healthcare-admin",
        "persona": "Healthcare recruiter hiring bilingual admin staff in South Texas",
        "facts": {
            "role": "healthcare admin handling patient records",
            "seniority": "mid-level admin",
            "skills": "HIPAA compliance, medical terminology, Microsoft Word; personality in Spanish",
            "languages": "functionally bilingual — English for written knowledge tests, Latin American Spanish for personality",
            "remote": "yes",
        },
    },
    "c8": {
        "id": "c8-admin-assistant-office",
        "persona": "Office manager screening admin assistants for daily Excel and Word work",
        "facts": {
            "role": "administrative assistant",
            "seniority": "entry to mid-level",
            "skills": "Microsoft Excel and Word — wants simulations not just knowledge tests",
            "constraints": "quick screen but willing to add 365 simulations for capability depth",
        },
    },
    "c9": {
        "id": "c9-senior-fullstack-backend",
        "persona": "Engineering hiring manager filling a senior backend-leaning full-stack role",
        "facts": {
            "role": "senior full-stack engineer (backend-leaning senior IC)",
            "seniority": "senior IC, 5+ years, leads design on own services, mentors but no direct reports",
            "skills": "Core Java Advanced, Spring, SQL, AWS, Docker; keep Verify G+; drop REST; Angular secondary",
            "remote": "yes",
        },
    },
    "c10": {
        "id": "c10-graduate-management-trainee",
        "persona": "Graduate programme lead building a management trainee assessment battery",
        "facts": {
            "role": "graduate management trainee",
            "seniority": "recent graduates, no prior work experience",
            "skills": "cognitive ability, situational judgement; OPQ removed as too long",
            "constraints": "final battery is Verify G+ and Graduate Scenarios only",
        },
    },
}


def slug_from_url(url: str) -> str:
    return urlparse(url.rstrip("/")).path.rsplit("/", 1)[-1]


# Trace URL slugs that differ from our catalog slugs.
SLUG_ALIASES: dict[str, str] = {
    "opq-leadership-report": "occupational-personality-questionnaire-leadership-report",
    "opq-universal-competency-report-2-0": "occupational-personality-questionnaire-universal-competency-report",
    "opq-mq-sales-report": "occupational-personality-questionnaire-sales-report",
}


def find_md_sources() -> list[Path]:
    """Prefer user-downloaded C*.md in project root, then eval/traces/, then traces_md/."""
    candidates: list[Path] = []
    for i in range(1, 11):
        for base in (ROOT, ROOT / "eval" / "traces", ROOT / "eval" / "traces_md"):
            for name in (f"C{i}.md", f"c{i}.md"):
                p = base / name
                if p.exists():
                    candidates.append(p)
                    break
            else:
                continue
            break
    # De-dupe by stem, keeping first (higher-priority location).
    seen: set[str] = set()
    out: list[Path] = []
    for p in candidates:
        stem = p.stem.lower()
        if stem not in seen:
            seen.add(stem)
            out.append(p)
    return sorted(out, key=lambda p: p.stem.lower())


def load_url_index() -> dict[str, str]:
    catalog = json.loads((ROOT / "data" / "catalog.json").read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    for entry in catalog:
        index[entry["url"].rstrip("/")] = entry["id"]
        index[slug_from_url(entry["url"])] = entry["id"]
    return index


def parse_md(path: Path) -> dict:
    content = path.read_text(encoding="utf-8", errors="replace")
    stem = path.stem.lower()
    meta = TRACE_META[stem]

    turn_pattern = re.compile(
        r"###\s+Turn\s+(\d+)(.*?)(?=###\s+Turn\s+\d+|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    turns = turn_pattern.findall(content)

    user_messages: list[str] = []
    final_urls: list[str] = []
    last_rec_urls: list[str] = []

    url_pattern = re.compile(r"<(https://www\.shl\.com/products/product-catalog/view/[^>]+)>")

    for _turn_num, turn_content in turns:
        user_match = re.search(
            r"\*\*User\*\*\s*\n+>\s*(.+?)(?=\n\n|\*\*Agent\*\*|\Z)",
            turn_content,
            re.DOTALL,
        )
        if user_match:
            msg = user_match.group(1).strip()
            msg = re.sub(r"\n+>\s*", "\n", msg).strip()
            user_messages.append(msg)

        urls = url_pattern.findall(turn_content)
        if urls:
            last_rec_urls = urls

        if re.search(r"end_of_conversation[`'\"]*:\s*\*\*true\*\*", turn_content, re.I):
            final_urls = last_rec_urls

    if not final_urls:
        final_urls = last_rec_urls

    initial_query = user_messages[0] if user_messages else ""

    return {
        **meta,
        "initial_query": initial_query,
        "relevant_ids": [],  # filled after URL mapping
        "_urls": final_urls,
        "_user_messages": user_messages,
    }


def url_to_id(url: str, index: dict[str, str]) -> str:
    normalized = url.rstrip("/")
    slug = slug_from_url(url)
    slug = SLUG_ALIASES.get(slug, slug)
    if normalized in index:
        return index[normalized]
    if slug in index:
        return index[slug]
    for key, cid in index.items():
        if isinstance(key, str) and key.endswith(slug):
            return cid
    # Preserve expected shortlist slug when catalog lacks the entry.
    return slug


def main() -> None:
    url_index = load_url_index()
    sources = find_md_sources()
    if not sources:
        print("No C*.md trace files found.", file=sys.stderr)
        sys.exit(1)

    TRACES_OUT.mkdir(parents=True, exist_ok=True)

    catalog_ids = set(url_index.values())

    for md_path in sources:
        trace = parse_md(md_path)
        urls = trace.pop("_urls")
        trace.pop("_user_messages", None)

        ids: list[str] = []
        unmapped: list[str] = []
        for url in urls:
            cid = url_to_id(url, url_index)
            if cid not in catalog_ids:
                unmapped.append(url)
            if cid not in ids:
                ids.append(cid)

        trace["relevant_ids"] = ids
        out_path = TRACES_OUT / f"{md_path.stem.lower()}.json"
        out_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        note = f", {len(unmapped)} not in catalog" if unmapped else ""
        print(f"{out_path.name}: {len(ids)} relevant_ids{note}")

    print(f"Wrote {len(sources)} traces to {TRACES_OUT}")


if __name__ == "__main__":
    main()
