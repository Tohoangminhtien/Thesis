import pandas as pd
from agents.fact_checker import FactCheckerAgent
from agents.search_agent import SearchAgent
from agents.classify_agent import ClassifyAgent
from agents.legality_checker import LegalityChecker
from agents.authentiate_agent import AuthenticateAgent
from model.llm_model import AzureModel
from pathlib import Path
from dotenv import load_dotenv
from openai import BadRequestError
import os
import json

load_dotenv()

# Set up azure OpenAI model
endpoint = os.getenv("END_POINT")
deployment = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("API_KEY")

model = AzureModel(endpoint=endpoint, deployment=deployment, api_key=api_key)

# Set up multiple agents
fact_checker = FactCheckerAgent(
    model, instruction_prompt=Path("./system_prompt/fact_checker.txt")
)
search_agent = SearchAgent(max_results=3)
classify_agent = ClassifyAgent(
    model, instruction_prompt=Path("./system_prompt/classify_prompt.txt", max_tokens=10)
)
legal_agent = LegalityChecker(
    model, instruction_prompt=Path("./system_prompt/legal.txt")
)
authentication_agent = AuthenticateAgent(
    model, instruction_prompt=Path("./system_prompt/authenticate.txt"), max_tokens=10
)

# Load the dataset
df = pd.read_csv("./data/test_binary.csv")

new_df = pd.DataFrame(
    columns=["classify_confidence", "fact_confidence", "authen_confidence", "label"]
)


for i in range(len(df)):
    statement = df["statement"][i]
    label = df["label"][i]
    metadata = {
        "subject": df["subject"][i],
        "speaker": df["speaker"][i],
        "job_title": df["job_title"][i],
        "state_info": df["state_info"][i],
        "party_affiliation": df["party_affiliation"][i],
        "context": df["context"][i],
    }

    # Legality Agent
    legal_response = legal_agent.check(statement=statement)
    if not legal_response:
        print("Violation of OpenAI policy")
        continue

    # Classify Agent
    classify_response = classify_agent.classify(statement=statement)
    classify_response = json.loads(classify_response)

    # Search Agent
    search_response = search_agent.search(query=statement)

    # Fact Checker Agent
    try:
        fact_response = fact_checker.check(
            statement=statement, search_results=search_response
        )
        fact_response = json.loads(fact_response)
    except BadRequestError as e:
        print(f"Violation of OpenAI policy")
        continue

    # Authenticate Agent
    auth_response = authentication_agent.check(metadata=metadata)
    auth_response = json.loads(auth_response)

    new_row = pd.DataFrame(
        [
            {
                "classify_confidence": classify_response["confidence"],
                "fact_confidence": fact_response["confidence"],
                "authen_confidence": auth_response["confidence"],
                "label": label,
            }
        ]
    )

    new_df = pd.concat([new_df, new_row], ignore_index=True)
    new_df.to_csv("./results/agent_results.csv", index=False)
