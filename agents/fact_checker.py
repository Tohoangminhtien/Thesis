from agents.base_agent import BaseAgent


class FactCheckerAgent(BaseAgent):
    def __init__(self, model, instruction_prompt, max_tokens=100):
        super().__init__(model, instruction_prompt, max_tokens=max_tokens)

    def check(self, statement: str, search_results: str):
        user_prompt = (
            f"Statement: {statement}\nSearch Results: {search_results}\nOutput:"
        )
        response = self.ask(user_prompt=user_prompt)
        return response
