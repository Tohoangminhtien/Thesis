import re
from model.llm_model import AzuzeModel
from pathlib import Path


class BaseAgent:
    def __init__(self, model: AzuzeModel, instruction_prompt: Path):
        self.model = model

        if not instruction_prompt.exists():
            return f"Prompt file not found: {instruction_prompt}"

        with open(instruction_prompt, "r", encoding="utf-8") as f:
            self.instruction = f.read().strip()

        self.expected_vars = set(re.findall(r"{(\w+)}", self.instruction))

    def ask(self, **kwargs):
        missing = self.expected_vars - kwargs.keys()
        if missing:
            return f"Missing variable(s): {missing}"
        extra = kwargs.keys() - self.expected_vars
        if extra:
            return f"Unexpected variable(s): {extra}"

        system_prompt = self.instruction
        for key in self.expected_vars:
            system_prompt = system_prompt.replace(
                f"{{{key}}}", str(kwargs[key]))

        user_prompt = "\n".join(
            f"{key.capitalize()}: {value}" for key, value in kwargs.items()
        )

        return self.model.chat(system_prompt, user_prompt)
