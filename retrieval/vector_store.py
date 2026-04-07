import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import faiss
import pickle
import numpy as np
from retrieval.embedder import embed_texts, chunk_text
from logger import get_logger

log = get_logger("VectorStore")

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_db")
INDEX_FILE = os.path.join(DB_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(DB_DIR, "chunks.pkl")


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        os.makedirs(DB_DIR, exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(CHUNKS_FILE, "rb") as f:
                self.chunks = pickle.load(f)
            log.info(f"Loaded FAISS index | {self.index.ntotal} vectors")

    def _save(self):
        faiss.write_index(self.index, INDEX_FILE)
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)

    def add_documents(self, text: str, source_url: str):
        new_chunks = chunk_text(text)
        if not new_chunks:
            return
        vectors = embed_texts(new_chunks)
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend([(chunk, source_url) for chunk in new_chunks])
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = embed_texts([query])
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                text, source = self.chunks[idx]
                results.append({"text": text, "source": source, "score": float(dist)})
        return results
