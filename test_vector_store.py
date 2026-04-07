"""
Demo: test the VectorStore end-to-end.
Shows chunking, embedding dimensions, storage, and similarity search.
Run: python test_vector_store.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Use a fresh in-memory store for this test (don't touch faiss_db/)
import faiss
import numpy as np
from retrieval.embedder import embed_texts, chunk_text

# ── 1. Sample documents ───────────────────────────────────────────────────────
docs = [
    {
        "url": "https://example.com/vector-db",
        "text": (
            "Vector databases are specialized storage systems designed to store and search "
            "high-dimensional vectors. They use approximate nearest neighbor algorithms to "
            "find similar vectors efficiently. Common use cases include semantic search, "
            "recommendation systems, and retrieval-augmented generation pipelines."
        ),
    },
    {
        "url": "https://example.com/faiss",
        "text": (
            "FAISS (Facebook AI Similarity Search) is an open-source library for efficient "
            "similarity search and clustering of dense vectors. It supports both exact and "
            "approximate search and can handle billions of vectors. IndexFlatL2 performs "
            "exact search using Euclidean distance."
        ),
    },
    {
        "url": "https://example.com/llm",
        "text": (
            "Large language models like GPT and Llama are trained on massive text corpora. "
            "They generate text by predicting the next token given a context window. "
            "Fine-tuning adapts a pretrained model to a specific task or domain."
        ),
    },
]

# ── 2. Chunk each document ────────────────────────────────────────────────────
print("\n=== STEP 1: CHUNKING ===")
all_chunks = []   # (text, url)
for doc in docs:
    chunks = chunk_text(doc["text"], chunk_size=50, overlap=5)  # small size for demo
    print(f"\nDoc: {doc['url']}")
    print(f"  Words: {len(doc['text'].split())} → {len(chunks)} chunk(s)")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: \"{c[:80]}...\"" if len(c) > 80 else f"  Chunk {i}: \"{c}\"")
        all_chunks.append((c, doc["url"]))

# ── 3. Embed all chunks ───────────────────────────────────────────────────────
print("\n=== STEP 2: EMBEDDING ===")
texts = [c[0] for c in all_chunks]
vectors = embed_texts(texts)
print(f"  Input : {len(texts)} chunks")
print(f"  Output: matrix shape = {vectors.shape}  (rows=chunks, cols=dimensions)")
print(f"  Each chunk → a {vectors.shape[1]}-dimensional vector")
print(f"\n  Sample vector (chunk 0, first 8 dims):")
print(f"  {vectors[0][:8].tolist()}")

# ── 4. Build FAISS index ──────────────────────────────────────────────────────
print("\n=== STEP 3: STORING IN FAISS ===")
dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors)
print(f"  Index type : IndexFlatL2 (exact search, Euclidean distance)")
print(f"  Dimension  : {dim}")
print(f"  Vectors stored: {index.ntotal}")

# ── 5. Search ─────────────────────────────────────────────────────────────────
queries = [
    "how does similarity search work in vector databases",
    "training large language models on text data",
    "FAISS approximate nearest neighbor search",
]

print("\n=== STEP 4: SEARCHING ===")
for query in queries:
    print(f"\n  Query: \"{query}\"")
    q_vec = embed_texts([query])
    distances, indices = index.search(q_vec, k=2)

    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        chunk_text_val, source = all_chunks[idx]
        preview = chunk_text_val[:90] + "..." if len(chunk_text_val) > 90 else chunk_text_val
        print(f"  Rank {rank+1} | dist={dist:.4f} | source={source}")
        print(f"         \"{preview}\"")

print("\n=== DONE ===")
print("Lower distance = more semantically similar to the query.")
