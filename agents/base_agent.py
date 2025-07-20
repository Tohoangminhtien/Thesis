from model.llm_model import AzureModel
from pathlib import Path


class BaseAgent:

    def __init__(self, model: AzureModel, system_prompt: Path, max_tokens: int = 100):
        self.model = model
        self.max_tokens = max_tokens

        if not system_prompt.exists():
            raise FileNotFoundError(f"Prompt file not found: {system_prompt}")

        with open(system_prompt, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    def ask(self, user_prompt: str):
        return self.model.chat(
            self.system_prompt, user_prompt, max_tokens=self.max_tokens
        )
