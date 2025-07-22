import pandas as pd
from model.llm_model import AzureModel
from dotenv import load_dotenv
import os
from tqdm import tqdm
from openai import BadRequestError


load_dotenv()

# Set up azure OpenAI model
endpoint = os.getenv("END_POINT")
deployment = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("API_KEY")

# Create an instance of the AzureModel
model = AzureModel(endpoint=endpoint, deployment=deployment, api_key=api_key)

# Load the dataset
df = pd.read_csv("./data/test_binary.csv")
statements = df["statement"].tolist()
labels = df["label"].tolist()

# Load txt file with prompts
with open("./system_prompt/zero_shot.txt", "r") as file:
    system_prompt = file.read()

# Process each statement with the model
results = []
for statement, label in tqdm(zip(statements, labels), total=len(statements)):

    user_prompt = f"Statement:\n{statement}"

    try:
        result = model.chat(system_prompt=system_prompt, user_prompt=user_prompt)
    except BadRequestError:
        continue

    results.append(
        {
            "statement": statement,
            "predict": result,
            "label": label,
        }
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv("./data/zero_shot_results.csv", index=False)
