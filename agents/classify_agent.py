from agents.base_agent import BaseAgent


class ClassifyAgent(BaseAgent):
    def __init__(self, model, instruction_prompt, max_tokens=100):
        super().__init__(model, instruction_prompt, max_tokens)

    def classify(self, statement: str):
        user_prompt = f"Statement: {statement}\nOutput:"
        response = self.ask(user_prompt=user_prompt)
        return response
