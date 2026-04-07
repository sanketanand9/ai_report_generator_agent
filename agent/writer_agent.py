from agent.llm import call_llm
from logger import get_logger

log = get_logger("WriterAgent")


class WriterAgent:
    def run(self, task: str) -> str:
        log.info("→ Synthesizing report...")
        prompt = f"""You are a research writer. Write a detailed, well-structured research report.

{task}

Structure:
1. Overview
2. Key Concepts
3. How It Works
4. Production Considerations
5. Key Findings

Be specific and detailed. Write the report directly — no meta-commentary.
"""
        result = call_llm(prompt)
        log.info(f"  Done | {len(result)} chars")
        return result
