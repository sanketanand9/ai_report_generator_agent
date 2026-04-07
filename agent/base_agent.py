import json
import re
from agent.llm import call_llm
from logger import get_logger


class BaseAgent:
    def __init__(self, name: str, role: str, tools: dict, max_steps: int = 6):
        self.name = name
        self.role = role
        self.tools = tools
        self.max_steps = max_steps
        self.memory = []
        self.log = get_logger(name)

    def _tool_descriptions(self) -> str:
        return "\n".join([f"- {n}: {m['description']}" for n, m in self.tools.items()])

    def _build_prompt(self, task: str, step: int) -> str:
        history = ""
        for entry in self.memory:
            history += (
                f"\nThought: {entry['thought']}"
                f"\nAction: {entry['action']}"
                f"\nObservation: {entry['observation']}\n"
            )
        return f"""You are {self.name}, a {self.role}.

Available tools:
{self._tool_descriptions()}

Your task: {task}

Previous steps:{history}

Step {step + 1}: Respond in EXACT format (no extra text):
Thought: <your reasoning>
Action: <tool_name> | <input>

If done:
Thought: <reasoning>
Action: FINISH | <final answer>
"""

    def _parse_action(self, response: str) -> tuple[str, str, str]:
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+)", response)
        thought = thought_match.group(1).strip() if thought_match else response[:200]
        action_raw = (action_match.group(1).strip() if action_match else "FINISH | done").split("\n")[0]
        if "|" in action_raw:
            tool_name, tool_input = action_raw.split("|", 1)
            return thought, tool_name.strip(), tool_input.strip()
        return thought, "FINISH", action_raw.strip()

    def _run_tool(self, tool_name: str, tool_input: str) -> str:
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        try:
            result = self.tools[tool_name]["fn"](tool_input)
            return json.dumps(result, indent=2) if isinstance(result, (list, dict)) else str(result)
        except Exception as e:
            return f"Tool error: {e}"

    def run(self, task: str) -> str:
        self.log.info(f"→ Task: {task[:100]}")
        self.memory = []
        for step in range(self.max_steps):
            response = call_llm(self._build_prompt(task, step))
            thought, tool_name, tool_input = self._parse_action(response)
            self.log.info(f"  Action: {tool_name} | {tool_input[:80]}")
            if tool_name == "FINISH":
                return tool_input
            observation = self._run_tool(tool_name, tool_input)
            self.memory.append({"thought": thought, "action": f"{tool_name} | {tool_input}", "observation": observation})
        return self.memory[-1]["observation"][:500] if self.memory else ""
