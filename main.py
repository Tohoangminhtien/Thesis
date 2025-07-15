from pathlib import Path
from model.llm_model import AzuzeModel
from agents.classify_agent import ClassifyAgent
from dotenv import load_dotenv
import os
import pandas as pd
import json
load_dotenv()

NUM_TEST = 10

df = pd.read_csv('./data/train.csv')
df = df[:NUM_TEST]
statements = df['statement'].tolist()
label = df['label'].tolist()

endpoint = "https://ditestgpt4o3.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
api_key = "2433baddb9ce409abd6fe70f37e974ae"

model = AzuzeModel(endpoint=endpoint, deployment=deployment, api_key=api_key)
agent = ClassifyAgent(model, instruction_prompt=Path(
    "./prompt/classify_prompt.txt"))

rows = []
count = 0

for X, Y in zip(statements, label):
    response = agent.ask(statement=X)
    response = json.loads(response)
    Y_pred = response['label']
    confidence = response['confidence']

    rows.append({
        'statement': X,
        'expected_label': Y,
        'predicted_label': Y_pred,
        'confidence': confidence
    })

    print(f"Statement: {X}")
    print(f"Expected Label: {Y}")
    print(f"Predicted Label: {Y_pred}, Confidence: {confidence}")
    print("-" * 30)

    if Y == Y_pred:
        count += 1

df = pd.DataFrame(rows)

print(f"Accuracy: {count / len(label) * 100:.2f}%")
df.to_csv('results.csv', index=False)
