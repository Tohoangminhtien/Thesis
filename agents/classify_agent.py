from agents.base_agent import BaseAgent


class ClassifyAgent(BaseAgent):
    def __init__(self, model, instruction_prompt):
        super().__init__(model, instruction_prompt)

    def classify(self, **kwangs):
        response = self.ask(**kwangs)
        return response
