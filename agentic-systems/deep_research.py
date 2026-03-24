"""
Assignment 10: Agentic Systems - Deep Research Assistant
=========================================================
Implements three agentic architectures for deep research:
  1. Sequential: single agent, step-by-step
  2. Parallel: sub-queries searched concurrently
  3. Hierarchical: planner + worker agents

Uses an LLM (via Ollama or HuggingFace) with web search tools.

ARCHITECTURE OVERVIEW:
  User Query
      |
  [Decomposer] -> sub-queries
      |
  [Searcher(s)] -> web results per sub-query
      |
  [Validator] -> filter/verify results
      |
  [Synthesizer] -> markdown report
"""

import json
import os
import sys
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuration ────────────────────────────────────────────────────────────

# OpenRouter API (free tier with Nemotron 120B)
OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
OPENROUTER_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "stepfun/step-3.5-flash:free",
    "minimax/minimax-m2.5:free",
]

# Search settings
MAX_SEARCH_RESULTS = 5
MAX_RETRIES = 2


# ══════════════════════════════════════════════════════════════════════════════
# LLM INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt, system="You are a helpful research assistant.", max_tokens=1024):
    """Call LLM via OpenRouter API with automatic model fallback."""
    import requests
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "models": OPENROUTER_MODELS,
                "route": "fallback",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content:
                return content.strip()
        else:
            print(f"  [OpenRouter error {resp.status_code}: {resp.text[:200]}]")
    except Exception as e:
        print(f"  [LLM error: {e}]")
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH TOOL
# ══════════════════════════════════════════════════════════════════════════════

def web_search(query, max_results=MAX_SEARCH_RESULTS):
    """Search the web using DuckDuckGo (no API key needed).

    This is our agent's 'tool' - the thing that connects it to
    external information. Without this, the LLM can only use its
    training data (which has a knowledge cutoff).
    """
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            }
            for r in results
        ]
    except Exception as e:
        print(f"  [Search error for '{query[:40]}...': {e}]")
        return []


def validate_search_results(results, query):
    """Validate search results - filter out irrelevant or low-quality hits.

    Validation is crucial in agentic systems because web search
    can return noisy, outdated, or irrelevant results.
    """
    validated = []
    query_words = set(query.lower().split())

    for r in results:
        snippet = r.get("snippet", "").lower()
        title = r.get("title", "").lower()

        # Check relevance: at least some query terms should appear
        combined = snippet + " " + title
        overlap = sum(1 for w in query_words if len(w) > 3 and w in combined)

        # Filter out very short snippets
        if len(snippet) < 30:
            continue

        # Filter out obvious non-content (login pages, error pages)
        skip_patterns = ["sign in", "log in", "404", "page not found", "cookie"]
        if any(p in snippet for p in skip_patterns):
            continue

        r["relevance_score"] = overlap / max(len(query_words), 1)
        validated.append(r)

    # Sort by relevance
    validated.sort(key=lambda x: x["relevance_score"], reverse=True)
    return validated


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: QUERY DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def decompose_query(query):
    """Break a complex query into searchable sub-queries.

    WHY: Complex research questions can't be answered by a single search.
    Decomposition lets us search for specific aspects independently.

    The LLM acts as a 'planner' here - deciding what information
    is needed to answer the full question.
    """
    prompt = f"""Break this research question into 3-5 specific, searchable sub-queries.
Each sub-query should target a specific fact or comparison needed to answer the full question.
Return ONLY a JSON list of strings, no other text.

Question: {query}

Sub-queries (JSON list):"""

    response = call_llm(prompt, system="You decompose complex questions into searchable sub-queries. Return only valid JSON.")

    # Parse the sub-queries from LLM response
    try:
        # Try to extract JSON list from response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            sub_queries = json.loads(match.group())
            if isinstance(sub_queries, list) and all(isinstance(s, str) for s in sub_queries):
                return sub_queries[:5]
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: split by newlines or numbered items
    lines = [l.strip().lstrip("0123456789.-) ") for l in response.split("\n") if l.strip()]
    lines = [l for l in lines if len(l) > 10 and not l.startswith("[") and not l.startswith("{")]

    if lines:
        return lines[:5]

    # Last resort: create basic sub-queries by extracting key entities/concepts
    # This heuristic splits comparison queries into their components
    parts = re.split(r'\b(?:versus|vs\.?|against|compared?\s+to|and)\b', query, flags=re.I)
    if len(parts) >= 2:
        sub_qs = [p.strip() for p in parts if len(p.strip()) > 15]
        if sub_qs:
            return sub_qs[:5]
    return [query]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: SEARCH AND ANSWER SUB-QUERIES
# ══════════════════════════════════════════════════════════════════════════════

def search_and_answer(sub_query, index=0):
    """Search for a sub-query and generate an answer from results.

    This is the core agent loop for each sub-query:
      1. Search the web
      2. Validate results
      3. If results are poor, retry with modified query
      4. Summarize findings into an answer
    """
    print(f"  [{index+1}] Searching: {sub_query[:60]}...")

    # Search with retry
    results = []
    for attempt in range(MAX_RETRIES):
        raw_results = web_search(sub_query)
        results = validate_search_results(raw_results, sub_query)

        if results:
            break

        # Retry with simplified query
        if attempt < MAX_RETRIES - 1:
            words = sub_query.split()
            sub_query = " ".join(words[:max(len(words)//2, 3)])
            print(f"       Retrying with: {sub_query[:50]}...")

    if not results:
        return {
            "sub_query": sub_query,
            "answer": "No relevant results found for this sub-query.",
            "sources": [],
            "confidence": "low",
        }

    # Build context from search results
    context = "\n\n".join([
        f"Source: {r['title']}\n{r['snippet']}" for r in results[:3]
    ])

    # Ask LLM to answer based on search results
    answer_prompt = f"""Based on these search results, answer this question concisely.
Cite specific facts and numbers when available. If the results don't fully answer
the question, say what's missing.

Question: {sub_query}

Search Results:
{context}

Answer:"""

    answer = call_llm(answer_prompt, max_tokens=500)

    sources = [{"title": r["title"], "url": r["url"]} for r in results[:3]]

    # Assess confidence
    confidence = "high" if len(results) >= 3 else "medium" if results else "low"

    print(f"       Found {len(results)} results, confidence: {confidence}")

    return {
        "sub_query": sub_query,
        "answer": answer if answer else context[:500],
        "sources": sources,
        "confidence": confidence,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════════

def synthesize_report(query, sub_results):
    """Combine sub-query answers into a coherent markdown report.

    The synthesizer's job is to:
      1. Merge overlapping information
      2. Identify contradictions or gaps
      3. Structure everything into a readable document
    """
    # Build the findings section
    findings = ""
    all_sources = []
    for i, r in enumerate(sub_results):
        findings += f"\n### Sub-question {i+1}: {r['sub_query']}\n"
        findings += f"**Confidence:** {r['confidence']}\n\n"
        findings += f"{r['answer']}\n"
        all_sources.extend(r.get("sources", []))

    synthesis_prompt = f"""You are writing a research report. Synthesize these findings into a
coherent markdown document with an introduction, key findings, analysis, and conclusion.
Do NOT just list the sub-answers -- integrate and compare them.

Original Question: {query}

Findings:
{findings}

Write the synthesis (markdown format):"""

    synthesis = call_llm(synthesis_prompt, max_tokens=1500)

    # Build the full report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# Deep Research Report

**Query:** {query}

**Generated:** {timestamp}

---

## Executive Summary

{synthesis if synthesis else 'Synthesis unavailable. See individual findings below.'}

---

## Detailed Findings

{findings}

---

## Sources

"""
    seen_urls = set()
    for i, s in enumerate(all_sources):
        if s["url"] not in seen_urls:
            seen_urls.add(s["url"])
            report += f"{len(seen_urls)}. [{s['title']}]({s['url']})\n"

    report += f"\n---\n\n*Report generated by Deep Research Agent on {timestamp}*\n"

    return report


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def run_sequential(query):
    """Architecture 1: Sequential pipeline.

    One agent does everything step by step:
      decompose -> search1 -> search2 -> ... -> synthesize

    PROS: Simple, deterministic, each step can use prior results
    CONS: Slow (no parallelism), sub-queries can't inform each other
    """
    print("\n" + "=" * 60)
    print("  Architecture 1: SEQUENTIAL")
    print("=" * 60)

    start = time.time()

    # Step 1: Decompose
    print("\n  [Decomposing query ...]")
    sub_queries = decompose_query(query)
    print(f"  Generated {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries):
        print(f"    {i+1}. {sq[:70]}")

    # Step 2: Search and answer each sub-query sequentially
    print("\n  [Searching sequentially ...]")
    sub_results = []
    for i, sq in enumerate(sub_queries):
        result = search_and_answer(sq, i)
        sub_results.append(result)

    # Step 3: Synthesize
    print("\n  [Synthesizing report ...]")
    report = synthesize_report(query, sub_results)

    elapsed = time.time() - start
    print(f"\n  Sequential completed in {elapsed:.1f}s")
    return report, elapsed


def run_parallel(query):
    """Architecture 2: Parallel search pipeline.

    Decompose first, then search ALL sub-queries at the same time:
      decompose -> [search1 || search2 || search3] -> synthesize

    PROS: Much faster (searches run concurrently)
    CONS: Sub-queries are independent (can't use result of one to
          inform another). Works well when sub-queries are truly
          independent; poorly when there are dependencies.
    """
    print("\n" + "=" * 60)
    print("  Architecture 2: PARALLEL")
    print("=" * 60)

    start = time.time()

    # Step 1: Decompose
    print("\n  [Decomposing query ...]")
    sub_queries = decompose_query(query)
    print(f"  Generated {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries):
        print(f"    {i+1}. {sq[:70]}")

    # Step 2: Search all sub-queries in parallel
    print("\n  [Searching in parallel ...]")
    sub_results = [None] * len(sub_queries)

    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as executor:
        futures = {
            executor.submit(search_and_answer, sq, i): i
            for i, sq in enumerate(sub_queries)
        }
        for future in as_completed(futures):
            idx = futures[future]
            sub_results[idx] = future.result()

    # Step 3: Synthesize
    print("\n  [Synthesizing report ...]")
    report = synthesize_report(query, sub_results)

    elapsed = time.time() - start
    print(f"\n  Parallel completed in {elapsed:.1f}s")
    return report, elapsed


def run_hierarchical(query):
    """Architecture 3: Hierarchical (planner + workers).

    A 'planner' agent decomposes AND decides the search order,
    identifying dependencies between sub-queries. Then workers
    execute in dependency order.

    decompose_with_deps -> [independent searches in parallel]
                        -> [dependent searches sequentially]
                        -> synthesize

    PROS: Handles dependencies between sub-queries
    CONS: More complex, planner can make mistakes about dependencies
    """
    print("\n" + "=" * 60)
    print("  Architecture 3: HIERARCHICAL (planner + workers)")
    print("=" * 60)

    start = time.time()

    # Step 1: Planner decomposes with dependency analysis
    print("\n  [Planner decomposing with dependencies ...]")
    plan_prompt = f"""Break this research question into 3-5 sub-queries.
For each, indicate if it depends on another sub-query's answer.

Return a JSON list of objects with "query" and "depends_on" (null or index).

Question: {query}

Plan (JSON):"""

    plan_response = call_llm(plan_prompt, system="You are a research planner. Return only valid JSON.")

    # Parse plan
    sub_queries = []
    dependencies = []
    try:
        match = re.search(r'\[.*?\]', plan_response, re.DOTALL)
        if match:
            plan = json.loads(match.group())
            for item in plan:
                if isinstance(item, dict):
                    sub_queries.append(item.get("query", str(item)))
                    dep = item.get("depends_on")
                    dependencies.append(dep)
                elif isinstance(item, str):
                    sub_queries.append(item)
                    dependencies.append(None)
    except (json.JSONDecodeError, AttributeError):
        # Fallback to basic decomposition
        sub_queries = decompose_query(query)
        dependencies = [None] * len(sub_queries)

    if not sub_queries:
        sub_queries = decompose_query(query)
        dependencies = [None] * len(sub_queries)

    print(f"  Plan: {len(sub_queries)} sub-queries")
    for i, (sq, dep) in enumerate(zip(sub_queries, dependencies)):
        dep_str = f" (depends on #{dep+1})" if dep is not None else " (independent)"
        print(f"    {i+1}. {sq[:60]}{dep_str}")

    # Step 2: Execute in dependency order
    print("\n  [Executing plan ...]")
    sub_results = [None] * len(sub_queries)

    # First pass: all independent queries in parallel
    independent = [i for i, d in enumerate(dependencies) if d is None]
    dependent = [i for i, d in enumerate(dependencies) if d is not None]

    print(f"  Phase 1: {len(independent)} independent queries (parallel)")
    with ThreadPoolExecutor(max_workers=min(len(independent), 4)) as executor:
        futures = {
            executor.submit(search_and_answer, sub_queries[i], i): i
            for i in independent
        }
        for future in as_completed(futures):
            idx = futures[future]
            sub_results[idx] = future.result()

    # Second pass: dependent queries sequentially
    if dependent:
        print(f"  Phase 2: {len(dependent)} dependent queries (sequential)")
        for i in dependent:
            dep_idx = dependencies[i]
            if dep_idx is not None and dep_idx < len(sub_results) and sub_results[dep_idx]:
                # Enrich the sub-query with context from its dependency
                prior_answer = sub_results[dep_idx]["answer"][:200]
                enriched = f"{sub_queries[i]} (Context: {prior_answer})"
                sub_results[i] = search_and_answer(enriched, i)
            else:
                sub_results[i] = search_and_answer(sub_queries[i], i)

    # Fill any None results
    for i in range(len(sub_results)):
        if sub_results[i] is None:
            sub_results[i] = search_and_answer(sub_queries[i], i)

    # Step 3: Synthesize
    print("\n  [Synthesizing report ...]")
    report = synthesize_report(query, sub_results)

    elapsed = time.time() - start
    print(f"\n  Hierarchical completed in {elapsed:.1f}s")
    return report, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

TEST_QUERIES = [
    (
        "Investigate the performance-to-compute efficiency of Test-Time Scaling "
        "(System 2 thinking) versus traditional Pre-training Scaling Laws. "
        "Specifically, compare the cost-effectiveness of DeepSeek-R1's reinforcement "
        "learning approach against OpenAI o3's reasoning tiers for mathematical "
        "reasoning tasks."
    ),
    (
        "Analyze the 'Lost-in-the-Middle' phenomenon in 2026's ultra-long context "
        "models (1M+ tokens). Compare the retrieval accuracy of Meta Llama 4 Scout "
        "(10M token window) against hybrid Mamba-Transformer (Jamba) architectures."
    ),
    (
        "Evaluate the shift from Late Fusion to Early Fusion in 2026 Vision-Language "
        "Models (VLMs). Compare the visual grounding capabilities of Gemini 2.5 Pro "
        "and InternVL3-78B on the MMMU benchmark."
    ),
]


def main():
    print("=" * 60)
    print("  Assignment 10: Deep Research Agent")
    print("=" * 60)

    # Use first test query for the architecture comparison
    query = TEST_QUERIES[0]
    print(f"\n  Query: {query[:80]}...")

    # Run all three architectures
    results = {}

    for name, runner in [
        ("Sequential", run_sequential),
        ("Parallel", run_parallel),
        ("Hierarchical", run_hierarchical),
    ]:
        report, elapsed = runner(query)
        results[name] = {"report": report, "time": elapsed}

        # Save the report
        safe_name = name.lower()
        output_path = f"agentic-systems/report_{safe_name}.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Saved -> {output_path}")

    # ── Comparison ───────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  ARCHITECTURE COMPARISON")
    print("=" * 60)
    print(f"\n  {'Architecture':<20s} {'Time (s)':>10s} {'Report Length':>14s}")
    print(f"  {'-'*20} {'-'*10} {'-'*14}")
    for name, data in results.items():
        print(f"  {name:<20s} {data['time']:>10.1f} {len(data['report']):>14,}")

    # ── Run remaining test queries with best architecture ────────────────────
    print("\n\n" + "=" * 60)
    print("  Running remaining queries with parallel architecture")
    print("=" * 60)

    for i, query in enumerate(TEST_QUERIES[1:], 2):
        print(f"\n  Query {i}: {query[:60]}...")
        report, elapsed = run_parallel(query)
        output_path = f"agentic-systems/report_query{i}.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Saved -> {output_path} ({elapsed:.1f}s)")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
