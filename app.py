import os
from dotenv import load_dotenv
from agents.base_agent import BaseAgent
from model.llm_model import AzuzeModel
from pathlib import Path

load_dotenv()

endpoint = "https://ditestgpt4o3.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
api_key = "2433baddb9ce409abd6fe70f37e974ae"

model = AzuzeModel(endpoint=endpoint, deployment=deployment, api_key=api_key)
agent = BaseAgent(model, instruction_prompt=Path("./prompt/classify_prompt.txt"))

statement = "Toàn cảnh phát ngôn của ông Trump: Vai trò NATO, viện trợ vũ khí cho Ukraine, trừng phạt Nga"

response = agent.ask(statement=statement)

print(response)
