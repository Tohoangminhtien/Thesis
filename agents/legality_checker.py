from agents.base_agent import BaseAgent
import openai


class LegalityChecker(BaseAgent):
    def __init__(self, model, instruction_prompt):
        super().__init__(model, instruction_prompt, max_tokens=1)

    def check(self, statement: str):
        try:
            user_prompt = f"{statement}"
            response = self.ask(user_prompt=user_prompt)
        except openai.BadRequestError:
            return ""
        return response
