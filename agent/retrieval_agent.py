"""
RetrievalAgent: queries the FAISS vector store to find relevant chunks.
Tool: vector_search
"""
from agent.base_agent import BaseAgent
from retrieval.vector_store import VectorStore


class RetrievalAgent(BaseAgent):
    def __init__(self, store: VectorStore):
        self.store = store
        # Wrap vector_search with the store injected
        tools = {
            "vector_search": {
                "fn": lambda query: store.search(query, top_k=8),
                "description": "Search indexed documents for relevant chunks. Input: query string.",
            }
        }
        super().__init__(
            name="RetrievalAgent",
            role="retrieval specialist who finds the most relevant information from indexed documents",
            tools=tools,
            max_steps=4,
        )
if __name__ == "__main__":
    import sys
    from retrieval.vector_store import VectorStore

    store = VectorStore()

    agent = RetrievalAgent(store)
    query = "Vector database in production"
    results = agent.store.search(query, top_k=3)
    for i, chunk in enumerate(results):
        print(f"[{i+1}] {chunk}")
