"""
OrchestratorAgent: top-level agent that plans and delegates to sub-agents.
"""
import re
from agent.llm import call_llm
from agent.research_agent import ResearchAgent
from agent.retrieval_agent import RetrievalAgent
from agent.writer_agent import WriterAgent
from agent.report_builder import build_report, save_report
from retrieval.vector_store import VectorStore
from logger import get_logger


class OrchestratorAgent:
    def __init__(self):
        self.store = VectorStore()
        self.research_agent = ResearchAgent(self.store)
        self.retrieval_agent = RetrievalAgent(self.store)
        self.writer_agent = WriterAgent()
        self.memory = []
        self.max_steps = 8
        self.log = get_logger("Orchestrator")

    def _agent_descriptions(self) -> str:
        return (
            "- research_agent: Searches the web and indexes content.\n"
            "- retrieval_agent: Queries indexed documents for relevant chunks.\n"
            "- writer_agent: Synthesizes context into a structured report.\n"
            "- FINISH: End the research."
        )

    def _build_prompt(self, topic: str, step: int) -> str:
        history = ""
        for entry in self.memory:
            obs_preview = entry["observation"][:300].replace("\n", " ")
            history += (
                f"\nThought: {entry['thought']}"
                f"\nAction: {entry['action']}"
                f"\nObservation: {obs_preview}...\n"
            )
        return f"""You are the OrchestratorAgent managing a research team.
Your goal: produce a comprehensive research report on: "{topic}"

Your team:
{self._agent_descriptions()}

Previous steps:{history}

Step {step + 1}: Respond in EXACT format (one Thought, one Action only):
Thought: <your reasoning>
Action: <agent_name> | <task>

Rules:
- Call research_agent at least twice with different search queries
- Call retrieval_agent after research to get relevant chunks
- Call writer_agent once you have enough context
- Call FINISH only after writer_agent has produced a summary
"""

    def _parse_action(self, response: str) -> tuple[str, str, str]:
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+)", response)
        thought = thought_match.group(1).strip() if thought_match else response[:200]
        action_raw = (action_match.group(1).strip() if action_match else "FINISH | done").split("\n")[0]
        if "|" in action_raw:
            agent_name, task = action_raw.split("|", 1)
            return thought, agent_name.strip(), task.strip()
        return thought, "FINISH", action_raw.strip()

    def _delegate(self, agent_name: str, task: str) -> str:
        if agent_name == "research_agent":
            return self.research_agent.run(task)
        elif agent_name == "retrieval_agent":
            return self.retrieval_agent.run(task)
        elif agent_name == "writer_agent":
            return self.writer_agent.run(task)
        return f"Unknown agent: {agent_name}"

    def run(self, topic: str) -> dict:
        self.log.info(f"Starting: {topic}")
        final_summary = ""

        for step in range(self.max_steps):
            response = call_llm(self._build_prompt(topic, step))
            thought, agent_name, task = self._parse_action(response)
            self.log.info(f"[Step {step+1}] → {agent_name}")

            if agent_name == "FINISH":
                break

            if agent_name == "writer_agent":
                chunks = self.store.search(topic, top_k=10)
                if chunks:
                    context = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in chunks])
                    task = f"Write a detailed research report on: '{topic}'\n\nSource material:\n\n{context}"

            observation = self._delegate(agent_name, task)

            if agent_name == "writer_agent":
                final_summary = observation

            self.memory.append({
                "thought": thought,
                "action": f"{agent_name} | {task[:100]}",
                "observation": observation[:500],
            })

        if not final_summary:
            self.log.info("[Fallback] Running writer directly...")
            chunks = self.store.search(topic, top_k=10)
            context = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in chunks])
            final_summary = self.writer_agent.run(
                f"Write a detailed research report on: '{topic}'\n\nSource material:\n\n{context}"
            )

        seen = set()
        unique_sources = []
        for s in self.research_agent.indexed_urls:
            if s["url"] not in seen:
                seen.add(s["url"])
                unique_sources.append(s)

        total_vectors = self.store.index.ntotal if self.store.index else 0
        self.log.info(f"Done | {len(unique_sources)} sources | {total_vectors} vectors")

        result = {
            "topic": topic,
            "summary": final_summary,
            "sources": unique_sources,
            "chunks_retrieved": total_vectors,
        }
        save_report(result)
        return result
