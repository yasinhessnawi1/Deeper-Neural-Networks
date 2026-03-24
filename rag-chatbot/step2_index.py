"""
Step 2: Chunk course data and index into ChromaDB vector store.
===============================================================
HOW IT WORKS:
  1. Load scraped course JSON
  2. Chunk each course's text into smaller pieces (with overlap)
  3. Embed each chunk using a sentence-transformer model
  4. Store embeddings + metadata in ChromaDB (local vector DB)

KEY CONCEPTS:
  - Chunking: Split text into small overlapping pieces for precise retrieval
  - Embeddings: Convert text -> dense vector (numbers that capture meaning)
  - Vector store: Database that supports fast similarity search on vectors
"""

import json
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_courses(path="rag-chatbot/courses.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_courses(courses, chunk_size=500, chunk_overlap=100):
    """Split course content into overlapping chunks.

    Why overlap? If a relevant sentence spans a chunk boundary,
    the overlap ensures it appears in at least one chunk fully.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for course in courses:
        # Prepend course code and name so each chunk has context
        header = f"Course: {course['code']} - {course['name']}"
        if course.get("credits"):
            header += f" ({course['credits']})"

        text = f"{header}\n{course['content']}"
        splits = splitter.split_text(text)

        for i, chunk_text in enumerate(splits):
            chunks.append({
                "id": f"{course['code']}_chunk_{i}",
                "text": chunk_text,
                "metadata": {
                    "course_code": course["code"],
                    "course_name": course["name"],
                    "chunk_index": i,
                    "url": course.get("url", ""),
                },
            })

    return chunks


def index_chunks(chunks, collection_name="uia_ikt_courses"):
    """Store chunks in ChromaDB with sentence-transformer embeddings.

    ChromaDB handles:
      - Embedding computation (via the embedding function we provide)
      - Storage of vectors + metadata
      - Fast approximate nearest neighbour (ANN) search at query time
    """
    # Use a multilingual embedding model for cross-lingual retrieval
    # (Norwegian course content + English queries)
    # intfloat/multilingual-e5-base: 768-dim, handles 100+ languages
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-base"
    )

    # Create/reset local ChromaDB (persisted to disk)
    client = chromadb.PersistentClient(path="rag-chatbot/chroma_db")

    # Delete if exists (for clean re-indexing)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )

    # ChromaDB batches internally, but let's add in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    return collection


def main():
    print("=" * 60)
    print("  Step 2: Chunking and Indexing Course Data")
    print("=" * 60)

    print("\n[1/3] Loading scraped courses ...")
    courses = load_courses()
    print(f"  Loaded {len(courses)} courses")

    print("\n[2/3] Chunking with overlap ...")
    # Try different chunk sizes to demonstrate the parameter effect
    for size in [300, 500, 800]:
        test_chunks = chunk_courses(courses, chunk_size=size, chunk_overlap=100)
        print(f"  chunk_size={size}: {len(test_chunks)} chunks "
              f"(avg {sum(len(c['text']) for c in test_chunks) / max(len(test_chunks), 1):.0f} chars)")

    # Use 500 as our default
    chunks = chunk_courses(courses, chunk_size=500, chunk_overlap=100)
    print(f"\n  Using chunk_size=500: {len(chunks)} chunks")
    if chunks:
        print(f"\n  Sample chunk:")
        print(f"  ID: {chunks[0]['id']}")
        print(f"  Text: {chunks[0]['text'][:200]}...")

    print("\n[3/3] Indexing into ChromaDB ...")
    collection = index_chunks(chunks)
    print(f"  Indexed {collection.count()} chunks into ChromaDB")

    # Quick test: verify retrieval works
    print("\n  Quick test query: 'machine learning'")
    results = collection.query(query_texts=["machine learning"], n_results=3)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"    [{i+1}] {meta['course_code']}: {doc[:80]}...")

    print("\nDone.")


if __name__ == "__main__":
    main()
