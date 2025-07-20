from openai import AzureOpenAI


class AzureModel:
    def __init__(
        self,
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str = "2025-01-01-preview",
    ):
        self.deployment = deployment
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 100):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
        )

        return completion.choices[0].message.content
