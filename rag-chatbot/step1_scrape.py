"""
Step 1: Scrape IKT course descriptions from UiA's website.
==========================================================
Discovers course URLs from UiA study program pages, then scrapes
each course page for structured content (learning outcomes, content,
assessment, etc.).

HOW IT WORKS:
  1. Visit study program pages (IT bachelor, IKT, Data engineering, etc.)
  2. Collect all course links (/studier/emner/...)
  3. For each course, extract structured sections from the HTML
  4. Save as JSON for indexing in step 2
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re

BASE = "https://www.uia.no"

# Study programs to crawl for course links
PROGRAM_URLS = [
    f"{BASE}/studier/program/it-og-informasjonssystemer-bachelor/studieplaner/2025h.html",
    f"{BASE}/studier/program/ikt-arsstudium/studieplaner/2025h.html",
    f"{BASE}/studier/program/informasjonssystemer-master-2-ar/studieplaner/2025h.html",
    f"{BASE}/studier/program/data-ingeniorutdanning-bachelor/studieplaner/2025h.html",
    f"{BASE}/studier/program/cybersikkerhetsledelse-master-2-ar/studieplaner/2025h.html",
    f"{BASE}/studier/program/it-og-informasjonssystemer-arsstudium/studieplaner/2025h.html",
    f"{BASE}/studier/program/evu-strategisk-bruk-av-ikt/studieplaner/2025h.html",
]


def collect_course_urls():
    """Crawl program pages to find all course page URLs."""
    course_links = set()

    for prog_url in PROGRAM_URLS:
        try:
            r = requests.get(prog_url, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/studier/emner/" in href:
                    full = href if href.startswith("http") else BASE + href
                    course_links.add(full)
        except Exception as e:
            print(f"  Failed: {prog_url} ({e})")

    return sorted(course_links)


def scrape_course(url):
    """Extract structured content from a single course page."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        # Title: e.g. "IKT100 Nettverk, sikkerhet og personvern (Host 2025)"
        h1 = soup.find("h1")
        if not h1:
            return None
        title = h1.get_text(strip=True)

        # Extract course code from title or URL
        code_match = re.search(r"([A-Z]{2,4}[-]?\d{3})", title.upper().replace(" ", ""))
        if not code_match:
            code_match = re.search(r"([a-z]{2,4}[-]?\d{3})", url)
        code = code_match.group(1).upper().replace("-", "-") if code_match else ""

        # Extract name (everything after code, before parenthesis)
        name = re.sub(r"^[A-Za-z]{2,4}[-]?\d{3}\s*", "", title)
        name = re.sub(r"\s*\(.*\)\s*$", "", name).strip()

        # Gather all sections: h2 heading -> content until next h2
        main = soup.find("main") or soup
        sections = {}
        current_heading = None
        current_text = []

        for tag in main.find_all(["h2", "p", "li", "td"]):
            if tag.name == "h2":
                if current_heading and current_text:
                    sections[current_heading] = " ".join(current_text)
                current_heading = tag.get_text(strip=True)
                current_text = []
            elif current_heading:
                text = tag.get_text(strip=True)
                if text and len(text) > 3:
                    current_text.append(text)

        if current_heading and current_text:
            sections[current_heading] = " ".join(current_text)

        # Build full text content from relevant sections
        content_parts = []
        for heading, text in sections.items():
            # Skip table-of-contents and evaluation boilerplate
            if "innholdsfortegnelse" in heading.lower():
                continue
            content_parts.append(f"{heading}: {text}")

        if not content_parts:
            return None

        # Extract credits from URL or content
        credits = ""
        for text in sections.values():
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:studiepoeng|stp|credits|ECTS)", text, re.I)
            if m:
                credits = m.group(1) + " ECTS"
                break

        # Also check for connected study programs
        programs = sections.get("Emnet er tilknyttet f\u00f8lgende studieprogram", "")

        return {
            "code": code,
            "name": name,
            "url": url,
            "credits": credits,
            "programs": programs,
            "content": "\n".join(content_parts),
        }

    except Exception as e:
        return None


def main():
    print("=" * 60)
    print("  Step 1: Scraping IKT Course Data from UiA")
    print("=" * 60)

    print("\n[1/2] Collecting course URLs from study programs ...")
    urls = collect_course_urls()
    print(f"  Found {len(urls)} course page URLs")

    print(f"\n[2/2] Scraping course pages ...")
    courses = []
    seen = set()

    for i, url in enumerate(urls):
        course = scrape_course(url)
        if course and course["code"] and course["code"] not in seen:
            seen.add(course["code"])
            courses.append(course)
            print(f"  [{len(courses):3d}] {course['code']:<10s} {course['name'][:50]}")

        # Rate limit
        if (i + 1) % 15 == 0:
            time.sleep(0.5)

    print(f"\n  Total courses scraped: {len(courses)}")

    # Save
    output = "rag-chatbot/courses.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(courses, f, indent=2, ensure_ascii=False)
    print(f"  Saved -> {output}")

    # Stats
    ikt_count = sum(1 for c in courses if c["code"].startswith("IKT"))
    is_count = sum(1 for c in courses if c["code"].startswith("IS"))
    dat_count = sum(1 for c in courses if c["code"].startswith("DAT"))
    other = len(courses) - ikt_count - is_count - dat_count
    print(f"\n  Breakdown: {ikt_count} IKT, {is_count} IS, {dat_count} DAT, {other} other")

    if courses:
        print(f"\n  Sample ({courses[0]['code']}):")
        print(f"  {courses[0]['content'][:300]}...")

    print("\nDone.")


if __name__ == "__main__":
    main()
