"""Phase 1 — scrape SHL *Individual Test Solutions* into data/catalog.json.

The SHL product catalog (https://www.shl.com/products/product-catalog/) is a
paginated table. Individual Test Solutions use ``type=2``; pre-packaged Job
Solutions (``type=1``) are out of scope for the final catalog.

When the public listing table is unavailable (SHL now redirects catalog pages),
the scraper discovers product URLs via:
  1. Live catalog pagination (preferred)
  2. The SHL online product portal (online.shl.com)
  3. Supplemental URL seeds from archived catalog listings

For each discovered URL it attempts a live detail-page fetch; metadata falls back
to portal / archived rows when the detail page redirects.

Run:
    playwright install chromium
    python scripts/scrape_catalog.py

Output: data/catalog.json as list[CatalogEntry].
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

BASE = "https://www.shl.com"
CATALOG_URL = f"{BASE}/products/product-catalog/"
PAGE_SIZE = 12
TYPE_INDIVIDUAL = "2"
TYPE_JOB = "1"
MAX_LISTING_PAGES = 40

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "catalog.json"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
)

TEST_TYPE_CODES = {"A", "B", "C", "D", "E", "K", "P", "S"}

KEY_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

ARCHIVED_CSV_URL = (
    "https://raw.githubusercontent.com/AYUSHKHAIRE/"
    "SHL-Assessments-Recommandation-system/main/data/assessments_details.csv"
)


def slugify_url(url: str) -> str:
    """Stable id = trailing path slug of the detail URL."""
    path = urlparse(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else url
    return re.sub(r"[^a-z0-9\-]+", "-", slug.lower()).strip("-")


def slugify_name(name: str) -> str:
    """Guess catalog slug from a product name."""
    cleaned = re.sub(r"\([^)]*\)", "", name)
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned.lower()).strip("-")
    return cleaned


def canonical_url(slug: str) -> str:
    return f"{BASE}/products/product-catalog/view/{slug}/"


def _text(node) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)) if node else ""


def parse_test_types(raw: str) -> list[str]:
    letters = [c for c in re.findall(r"\b([ABCDEKPS])\b", raw or "") if c in TEST_TYPE_CODES]
    out: list[str] = []
    for letter in letters:
        if letter not in out:
            out.append(letter)
    return out or ["K"]


def parse_bool_remote(raw: str) -> bool:
    if not raw:
        return False
    lowered = raw.lower()
    if "no" in lowered and "remote" in lowered:
        return False
    return "yes" in lowered or "remote testing:" in lowered and "no" not in lowered


def parse_length_minutes(raw: str) -> int | None:
    if not raw:
        return None
    m = re.search(r"(\d+)", raw)
    return int(m.group(1)) if m else None


def split_csvish(raw: str) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in re.split(r"[,/]", raw) if part.strip()]


def is_individual_test(name: str, description: str = "") -> bool:
    """Exclude pre-packaged job solutions and non-test report products."""
    lowered = name.lower()
    if "report" in lowered and "questionnaire" not in lowered:
        return False
    if "short form" in lowered:
        return False
    blob = f"{name} {description}".lower()
    if "pre-packaged" in blob or "pre packaged" in blob:
        return False
    if " solution is an assessment used for job candidates" in blob:
        return False
    return True


def parse_detail_html(html: str, url: str) -> dict | None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Primary layout (legacy catalog detail pages).
    heading_div = soup.find("div", class_="row content__container typ")
    rows = soup.find_all("div", class_="product-catalogue-training-calendar__row typ")
    if heading_div and rows:
        name = heading_div.get_text(strip=True)
        description = rows[0].find("p").get_text(strip=True) if rows[0].find("p") else ""
        job_levels = rows[1].find("p").get_text(strip=True) if len(rows) > 1 and rows[1].find("p") else ""
        languages = rows[2].find("p").get_text(strip=True) if len(rows) > 2 and rows[2].find("p") else ""
        length_raw = rows[3].find("p").get_text(strip=True) if len(rows) > 3 and rows[3].find("p") else ""
        flex = rows[3].find("div", class_="d-flex") if len(rows) > 3 else None
        test_type_raw = ""
        remote_raw = ""
        if flex:
            ps = flex.find_all("p")
            if ps:
                test_type_raw = ps[0].get_text(strip=True)
            if len(ps) > 1:
                remote_raw = ps[1].get_text(strip=True)
        adaptive = bool(re.search(r"Adaptive/IRT", html, re.I)) and not bool(
            re.search(r"Adaptive/IRT[^A-Za-z]*No", html, re.I)
        )
        return _entry_from_fields(
            url=url,
            name=name,
            description=description,
            job_levels=split_csvish(job_levels),
            languages=split_csvish(languages),
            length_minutes=parse_length_minutes(length_raw),
            test_types=parse_test_types(test_type_raw),
            remote_testing=parse_bool_remote(remote_raw),
            adaptive_irt=adaptive,
        )

    # Fallback layout (newer pages).
    name_node = soup.find("h1")
    name = _text(name_node)
    if not name or name.lower() == "shl products":
        return None

    desc_parts: list[str] = []
    for p in soup.select(".product-catalogue-training-calendar__row p, .product-catalogue p, article p"):
        t = _text(p)
        if t:
            desc_parts.append(t)
    description = " ".join(desc_parts).strip()
    body = soup.get_text(" ", strip=True)

    test_types: list[str] = []
    tt_row = soup.find(string=re.compile(r"Test Type", re.I))
    if tt_row and tt_row.parent:
        container = tt_row.parent.parent or tt_row.parent
        test_types = parse_test_types(_text(container))

    remote = bool(re.search(r"Remote Testing", body, re.I)) and not bool(
        re.search(r"Remote Testing[^A-Za-z]*No", body, re.I)
    )
    adaptive = bool(re.search(r"Adaptive/IRT", body, re.I)) and not bool(
        re.search(r"Adaptive/IRT[^A-Za-z]*No", body, re.I)
    )

    languages: list[str] = []
    lang_m = re.search(r"Languages?\s*[:\-]?\s*([A-Za-z ,\(\)/]+)", body)
    if lang_m:
        languages = split_csvish(lang_m.group(1))[:20]

    length_minutes = None
    len_m = re.search(r"(?:Approximate\s+)?(?:Completion\s+Time|Assessment\s+length)[^0-9]*(\d+)", body, re.I)
    if len_m:
        length_minutes = int(len_m.group(1))

    job_levels: list[str] = []
    jl_m = re.search(r"Job\s+levels?\s*[:\-]?\s*([A-Za-z ,\-/]+)", body, re.I)
    if jl_m:
        job_levels = split_csvish(jl_m.group(1))[:20]

    return _entry_from_fields(
        url=url,
        name=name,
        description=description,
        job_levels=job_levels,
        languages=languages,
        length_minutes=length_minutes,
        test_types=test_types,
        remote_testing=remote,
        adaptive_irt=adaptive,
    )


def _entry_from_fields(
    *,
    url: str,
    name: str,
    description: str,
    job_levels: list[str],
    languages: list[str],
    length_minutes: int | None,
    test_types: list[str],
    remote_testing: bool,
    adaptive_irt: bool,
) -> dict:
    if not name:
        return None
    return {
        "id": slugify_url(url),
        "name": name.strip(),
        "url": url,
        "test_types": test_types or ["K"],
        "remote_testing": remote_testing,
        "adaptive_irt": adaptive_irt,
        "description": description[:4000],
        "job_levels": job_levels,
        "languages": languages,
        "length_minutes": length_minutes,
    }


def fetch_html(url: str, client: httpx.Client) -> str | None:
    try:
        resp = client.get(url, timeout=45)
        if resp.status_code != 200:
            return None
        if "product-catalog/view/" in url and "SHL Products" in resp.text and "Test Type" not in resp.text:
            return None
        return resp.text
    except Exception:
        return None


def scrape_listing_urls(client: httpx.Client, catalog_type: str) -> list[str]:
    from bs4 import BeautifulSoup

    links: list[str] = []
    empty_pages = 0
    for page_idx in range(MAX_LISTING_PAGES):
        start = page_idx * PAGE_SIZE
        list_url = f"{CATALOG_URL}?start={start}&type={catalog_type}"
        html = fetch_html(list_url, client)
        if not html:
            empty_pages += 1
            if empty_pages >= 2:
                break
            continue

        soup = BeautifulSoup(html, "html.parser")
        page_links: list[str] = []
        for td in soup.find_all("td", class_="custom__table-heading__title"):
            anchor = td.find("a", href=True)
            if anchor:
                page_links.append(urljoin(BASE, anchor["href"]))

        if not page_links:
            for a in soup.select("a[href*='/product-catalog/view/']"):
                page_links.append(urljoin(BASE, a["href"]))

        page_links = list(dict.fromkeys(page_links))
        if not page_links:
            empty_pages += 1
            if empty_pages >= 2:
                break
            continue

        empty_pages = 0
        links.extend(page_links)
        print(f"  listing type={catalog_type} start={start}: +{len(page_links)} links", file=sys.stderr)

    return list(dict.fromkeys(links))


def dismiss_cookiebot(page) -> None:
    for sel in [
        "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
        "button:has-text('Allow all')",
        "#onetrust-accept-btn-handler",
        "button:has-text('Accept All')",
    ]:
        try:
            page.locator(sel).first.click(timeout=2000)
            page.wait_for_timeout(500)
        except Exception:
            pass


def scrape_online_portal() -> dict[str, dict]:
    """Return slug -> {name, description} from online.shl.com."""
    from bs4 import BeautifulSoup
    from playwright.sync_api import sync_playwright

    portal_rows: dict[str, dict] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            )
        )
        page.goto(
            "https://online.shl.com/gb/en-us/products?producttypes=1",
            wait_until="networkidle",
            timeout=90000,
        )
        dismiss_cookiebot(page)
        page.wait_for_timeout(1500)

        try:
            page.select_option("#dt-length-0", "100")
            page.wait_for_timeout(2500)
        except Exception:
            pass

        for _ in range(10):
            soup = BeautifulSoup(page.content(), "html.parser")
            for tr in soup.select("#myTable tbody tr"):
                cols = tr.find_all("td")
                if len(cols) < 2:
                    continue
                name = cols[1].get_text(" ", strip=True)
                desc = cols[2].get_text(" ", strip=True) if len(cols) > 2 else ""
                if not name:
                    continue
                slug = slugify_name(name)
                portal_rows[slug] = {"name": name, "description": desc}

            next_btn = page.locator("button.dt-paging-button.next:not(.disabled)")
            if next_btn.count() == 0:
                break
            dismiss_cookiebot(page)
            try:
                next_btn.first.click(timeout=5000)
            except Exception:
                page.evaluate(
                    """() => {
                        const next = document.querySelector('button.dt-paging-button.next:not(.disabled)');
                        if (next) next.click();
                    }"""
                )
            page.wait_for_timeout(2500)

        browser.close()

    print(f"  online portal: {len(portal_rows)} products", file=sys.stderr)
    return portal_rows


def load_archived_csv_rows(client: httpx.Client) -> list[dict]:
    try:
        resp = client.get(ARCHIVED_CSV_URL, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  ! archived CSV unavailable: {exc}", file=sys.stderr)
        return []

    rows = list(csv.DictReader(io.StringIO(resp.text)))
    print(f"  archived CSV seeds: {len(rows)} rows", file=sys.stderr)
    return rows


def discover_urls(client: httpx.Client, portal_rows: dict[str, dict], archived_rows: list[dict]) -> list[str]:
    urls: list[str] = []

    for catalog_type in (TYPE_INDIVIDUAL, TYPE_JOB):
        urls.extend(scrape_listing_urls(client, catalog_type))

    for slug in portal_rows:
        urls.append(canonical_url(slug))

    for row in archived_rows:
        link = (row.get("link") or "").strip()
        if link:
            urls.append(link if link.endswith("/") else link + "/")

    deduped = list(dict.fromkeys(urls))
    print(f"  discovered {len(deduped)} unique URLs", file=sys.stderr)
    return deduped


def entry_from_archived_row(row: dict) -> dict | None:
    url = (row.get("link") or "").strip()
    if not url:
        return None
    if not url.endswith("/"):
        url += "/"
    name = (row.get("heading") or "").strip()
    description = (row.get("desc") or "").strip()
    if not is_individual_test(name, description):
        return None
    return _entry_from_fields(
        url=url,
        name=name,
        description=description,
        job_levels=split_csvish(row.get("job_levels") or ""),
        languages=split_csvish(row.get("languages") or ""),
        length_minutes=parse_length_minutes(row.get("assessment_length") or ""),
        test_types=parse_test_types(row.get("test_type") or ""),
        remote_testing=parse_bool_remote(row.get("remote_testing") or ""),
        adaptive_irt=False,
    )


def entry_from_portal(url: str, portal: dict) -> dict | None:
    name = portal.get("name", "").strip()
    description = portal.get("description", "").strip()
    if not name or not is_individual_test(name, description):
        return None
    return _entry_from_fields(
        url=url,
        name=name,
        description=description,
        job_levels=[],
        languages=[],
        length_minutes=parse_length_minutes(description),
        test_types=["K"],
        remote_testing=True,
        adaptive_irt="adaptive" in description.lower(),
    )


def scrape() -> list[dict]:
    client = httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        follow_redirects=True,
        timeout=45,
    )

    portal_rows = scrape_online_portal()
    archived_rows = load_archived_csv_rows(client)
    archived_by_slug = {slugify_url(r["link"]): r for r in archived_rows if r.get("link")}

    urls = discover_urls(client, portal_rows, archived_rows)
    entries: dict[str, dict] = {}

    for idx, url in enumerate(urls, start=1):
        slug = slugify_url(url)
        if slug in entries:
            continue

        entry = None
        html = fetch_html(url, client)
        if html:
            entry = parse_detail_html(html, url)

        if entry is None and slug in archived_by_slug:
            entry = entry_from_archived_row(archived_by_slug[slug])

        if entry is None and slug in portal_rows:
            entry = entry_from_portal(url, portal_rows[slug])

        if entry is None:
            continue
        if not is_individual_test(entry["name"], entry["description"]):
            continue

        entries[entry["id"]] = entry
        if idx % 25 == 0:
            print(f"  processed {idx}/{len(urls)} urls, {len(entries)} kept", file=sys.stderr)
        time.sleep(0.15)

    client.close()
    return list(entries.values())


def main() -> None:
    entries = scrape()
    if len(entries) < 300:
        print(
            f"Warning: only {len(entries)} individual-test entries (expected >= 300).",
            file=sys.stderr,
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(entries)} entries -> {OUT_PATH}")


if __name__ == "__main__":
    main()
