"""
Step 3: RAG Chatbot for UiA IKT Courses
========================================
Vanilla RAG pipeline:
  Question -> Embed -> Retrieve top-k -> Build prompt -> LLM -> Answer

Uses:
  - ChromaDB for vector retrieval
  - Sentence-transformers for embeddings
  - HuggingFace transformers (local LLM) or Ollama for generation

The chatbot answers questions about IKT courses at UiA based on
the scraped and indexed course descriptions.
"""

import chromadb
from chromadb.utils import embedding_functions
import sys
import os

# ── Configuration ────────────────────────────────────────────────────────────

# Retrieval parameters (experiment with these!)
DEFAULT_K = 3          # Number of chunks to retrieve
CHUNK_SIZE = 500       # Must match what was used in step2

# System prompt - tells the LLM how to behave
SYSTEM_PROMPT = """You are a helpful assistant for students at the University of Agder (UiA).
You answer questions about IKT (IT and Information Systems) courses based on the provided context.
If the context does not contain enough information to answer, say so honestly.
Keep answers concise and helpful. Reference course codes when relevant."""

# ── Vector Store Setup ───────────────────────────────────────────────────────

def get_collection():
    """Connect to the existing ChromaDB collection."""
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-base"
    )
    client = chromadb.PersistentClient(path="rag-chatbot/chroma_db")
    return client.get_collection(
        name="uia_ikt_courses",
        embedding_function=embed_fn,
    )


def retrieve(collection, query, k=DEFAULT_K):
    """Retrieve top-k most relevant chunks for a query.

    This is the R in RAG. ChromaDB:
      1. Embeds the query using the same model as our documents
      2. Finds the k nearest vectors by cosine similarity
      3. Returns the corresponding text chunks + metadata
    """
    results = collection.query(
        query_texts=[query],
        n_results=k,
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "course_code": meta["course_code"],
            "course_name": meta["course_name"],
            "distance": dist,
        })
    return chunks


# ── Prompt Construction ──────────────────────────────────────────────────────

def build_prompt(query, chunks):
    """Construct the augmented prompt with retrieved context.

    This is the A in RAG - we AUGMENT the prompt with relevant info.
    The LLM sees the actual course data, so it can give grounded answers
    instead of hallucinating.
    """
    context = "\n\n---\n\n".join([
        f"[{c['course_code']}] {c['text']}" for c in chunks
    ])

    prompt = f"""{SYSTEM_PROMPT}

Context from UiA course descriptions:
{context}

Student question: {query}

Answer:"""
    return prompt


# ── LLM Generation ───────────────────────────────────────────────────────────

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


def call_openrouter(prompt, system=SYSTEM_PROMPT, max_tokens=500):
    """Call OpenRouter API with automatic model fallback routing."""
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
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            model_used = data.get("model", "unknown")
            if content:
                return content.strip(), model_used
        else:
            print(f"  OpenRouter error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  OpenRouter error: {e}")
    return None, None


def generate(prompt):
    """Generate answer via OpenRouter (free models with fallback)."""
    user_content = prompt.replace(SYSTEM_PROMPT, "").strip()
    answer, model = call_openrouter(user_content)
    if answer:
        return answer, f"OpenRouter ({model})"

    # Last resort: just return the retrieved context
    return "[LLM unavailable] Here are the relevant course excerpts:\n" + \
           prompt.split("Context from UiA course descriptions:")[1].split("Student question:")[0], \
           "None (retrieval only)"


# ── Chatbot Loop ─────────────────────────────────────────────────────────────

def run_demo(collection):
    """Run pre-defined test queries to demonstrate the system."""
    test_questions = [
        "What courses cover machine learning?",
        "Which IKT course teaches about databases?",
        "What are the prerequisites for IKT200?",
        "Does UiA offer courses on cybersecurity?",
        "What programming languages are taught in IKT courses?",
    ]

    print("\n" + "=" * 60)
    print("  RAG Demo: 5 Test Questions")
    print("=" * 60)

    for i, q in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"  Q{i}: {q}")
        print(f"{'='*60}")

        # Retrieve
        chunks = retrieve(collection, q, k=DEFAULT_K)
        print(f"\n  Retrieved {len(chunks)} chunks:")
        for j, c in enumerate(chunks):
            print(f"    [{j+1}] {c['course_code']} (distance={c['distance']:.4f})")
            print(f"        {c['text'][:100]}...")

        # Generate
        prompt = build_prompt(q, chunks)
        answer, model_used = generate(prompt)
        print(f"\n  Answer (via {model_used}):")
        print(f"  {answer[:500]}")

    return test_questions


def run_k_experiment(collection):
    """Experiment with different k values."""
    print("\n\n" + "=" * 60)
    print("  Experiment: Effect of k (number of retrieved chunks)")
    print("=" * 60)

    query = "What courses are related to artificial intelligence?"

    for k in [1, 3, 5, 7]:
        chunks = retrieve(collection, query, k=k)
        codes = [c["course_code"] for c in chunks]
        avg_dist = sum(c["distance"] for c in chunks) / len(chunks)
        print(f"\n  k={k}: Retrieved {codes}")
        print(f"        Avg distance: {avg_dist:.4f}")
        print(f"        Chunk sizes: {[len(c['text']) for c in chunks]}")


def run_interactive(collection):
    """Interactive chat mode."""
    print("\n\n" + "=" * 60)
    print("  Interactive Mode (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            query = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        chunks = retrieve(collection, query, k=DEFAULT_K)
        prompt = build_prompt(query, chunks)
        answer, model = generate(prompt)

        print(f"\n  [{model}]")
        print(f"  Bot: {answer[:500]}")
        print(f"\n  Sources: {', '.join(set(c['course_code'] for c in chunks))}")


def main():
    print("=" * 60)
    print("  Step 3: RAG Chatbot for UiA IKT Courses")
    print("=" * 60)

    print("\nConnecting to vector store ...")
    collection = get_collection()
    print(f"  Collection: {collection.count()} indexed chunks")

    # Run demo queries
    run_demo(collection)

    # Run k experiment
    run_k_experiment(collection)

    # Interactive mode (exit cleanly on EOF)
    run_interactive(collection)

    print("\nDone.")


if __name__ == "__main__":
    main()
