import pandas as pd
from agents.fact_checker import FactCheckerAgent
from agents.search_agent import SearchAgent
from agents.classify_agent import ClassifyAgent
from agents.legality_checker import LegalityChecker
from agents.authentiate_agent import AuthenticateAgent
from model.llm_model import AzureModel
from pathlib import Path
from dotenv import load_dotenv
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
df = pd.read_csv("data/test.csv")
new_df = pd.DataFrame(
    columns=[
        "statement",
        "fact_confidence",
        "classify_confidence",
        "authentication_confidence",
        "label_prediction",
        "label",
    ]
)
NUM_TEST = 10
for i in range(NUM_TEST):
    statement = df["statement"][i]
    subject = df["subject"][i]
    job_title = df["job_title"][i]
    speaker = df["speaker"][i]
    party_affiliation = df["party_affiliation"][i]
    state_info = df["state_info"][i]
    context = df["context"][i]
    label = df["label"][i]
    metadata = {
        "Subject": subject,
        "Speaker": speaker,
        "Job title": job_title,
        "State info": state_info,
        "Party_affiliation": party_affiliation,
        "Context": context,
    }

    print(f"Statement {statement}")

    # Legality Agent
    legal_response = legal_agent.check(statement=statement)
    if not legal_response:
        continue

    # Classify Agent
    classify_response = classify_agent.classify(statement=statement)
    classify_response = json.loads(classify_response)

    # Search Agent
    search_response = search_agent.search(query=statement)

    # Fact Checker Agent
    fact_response = fact_checker.check(
        statement=statement, search_results=search_response
    )
    fact_response = json.loads(fact_response)

    # Authenticate Agent
    auth_response = authentication_agent.check(metadata=metadata)
    auth_response = json.loads(auth_response)

    new_row = pd.DataFrame(
        [
            {
                "statement": statement,
                "fact_confidence": fact_response["confidence"],
                "classify_confidence": classify_response["confidence"],
                "authentication_confidence": auth_response["confidence"],
                "label_prediction": classify_response["label"],
                "label": label,
            }
        ]
    )

    new_df = pd.concat([new_df, new_row], ignore_index=True)
    # Save the results to a new CSV file
    new_df.to_csv("data/output.csv", index=False)
