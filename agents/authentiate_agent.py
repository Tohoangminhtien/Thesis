from agents.base_agent import BaseAgent


class AuthenticateAgent(BaseAgent):
    def __init__(self, model, instruction_prompt, max_tokens):
        super().__init__(model, instruction_prompt, max_tokens)

    def check(self, metadata: dict):
        target_keys = ["subject", "speaker", "job_title", "state_info", "party_affiliation", "context"]
        
        user_prompt = ""
        for key in metadata.keys():
            if key not in target_keys:
                raise ValueError(f"Invalid key '{key}' in metadata. Expected one of {target_keys}.")
            
            user_prompt += f"{key}: {metadata[key]}\n"
            
        response = self.ask(user_prompt=user_prompt)
        return response
