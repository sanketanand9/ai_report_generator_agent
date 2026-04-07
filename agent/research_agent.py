"""
ResearchAgent: searches the web and indexes content into the vector store.
Tools: web_search, fetch_url
"""
import json
from agent.base_agent import BaseAgent
from agent.tools import TOOL_REGISTRY
from retrieval.vector_store import VectorStore
from logger import get_logger

log = get_logger("ResearchAgent")


class ResearchAgent(BaseAgent):
    def __init__(self, store: VectorStore):
        self.store = store
        self.indexed_urls = []  # track what was indexed for source collection
        tools = {
            "web_search": TOOL_REGISTRY["web_search"],
            "fetch_url": TOOL_REGISTRY["fetch_url"],
        }
        super().__init__(
            name="ResearchAgent",
            role="web researcher who finds and indexes information from the internet",
            tools=tools,
            max_steps=6,
        )

    def _run_tool(self, tool_name: str, tool_input: str) -> str:
        """Override to auto-index ALL retrieved content into vector store."""
        result_str = super()._run_tool(tool_name, tool_input)

        try:
            result = json.loads(result_str)
            if isinstance(result, list):
                for item in result:
                    if item.get("content") and item.get("url"):
                        self.store.add_documents(item["content"], item["url"])
                        self.indexed_urls.append({"url": item["url"], "title": item.get("title", "")})
                        log.info(f"Indexed: {item['url']}")
            elif isinstance(result, dict):
                if result.get("content") and result.get("url"):
                    self.store.add_documents(result["content"], result["url"])
                    self.indexed_urls.append({"url": result["url"], "title": result.get("title", "")})
                    log.info(f"Indexed: {result['url']}")
        except Exception:
            pass

        return result_str
