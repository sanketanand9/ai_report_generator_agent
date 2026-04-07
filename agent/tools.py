"""
All tools available to agents.
Each tool is a plain function. The tool registry maps name -> (fn, description).
"""
import os
import requests
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ── Tool 1: Web Search ────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Tavily. Returns list of {url, title, content}."""
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "include_raw_content": True,
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    results = []
    for r in response.json().get("results", []):
        results.append({
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("raw_content") or r.get("content", ""),
        })
    return results


# ── Tool 2: Fetch URL ─────────────────────────────────────────────────────────
def fetch_url(url: str) -> dict:
    """Fetch raw text content from a URL. Returns {url, content}."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
        resp.raise_for_status()
        # crude text extraction — strip tags
        import re
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()
        return {"url": url, "content": text}  # cap at 5k chars
    except Exception as e:
        return {"url": url, "content": f"Error fetching URL: {e}"}


# ── Tool 3: Vector Search (injected at runtime) ───────────────────────────────
# This is a placeholder; ResearchAgent injects the real store at runtime.
def vector_search(query: str, store=None, top_k: int = 8) -> list[dict]:
    """Search the FAISS vector store for relevant chunks."""
    if store is None:
        return []
    return store.search(query, top_k=top_k)


# ── Tool Registry ─────────────────────────────────────────────────────────────
TOOL_REGISTRY = {
    "web_search": {
        "fn": web_search,
        "description": "Search the web for information. Input: query string.",
    },
    "fetch_url": {
        "fn": fetch_url,
        "description": "Fetch and read the content of a specific URL. Input: url string.",
    },
    "vector_search": {
        "fn": vector_search,
        "description": "Search indexed documents in memory for relevant chunks. Input: query string.",
    },
}

if __name__ == "__main__":
    # Test web_search
    print("=== web_search ===")
    results = web_search("Python async best practices", max_results=2)
    for r in results:
        print(r["title"], "-", r["url"])

    # Test fetch_url
    print("\n=== fetch_url ===")
    result = fetch_url("https://discuss.python.org/t/asyncio-best-practices/12576")
    print(result["content"])

    # Test vector_search with no store (should return empty)
    print("\n=== vector_search (no store) ===")
    print(vector_search("vector databse in production"))
