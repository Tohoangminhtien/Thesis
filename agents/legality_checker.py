from agents.base_agent import BaseAgent
import openai


class LegalityChecker(BaseAgent):
    def __init__(self, model, instruction_prompt):
        super().__init__(model, instruction_prompt)

    def check(self, **kwangs):
        try:
            response = self.ask(**kwangs, max_token=1)
        except openai.BadRequestError:
            return ""
        return response
