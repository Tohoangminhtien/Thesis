from flask import Flask, render_template, request
import os, json, ast, joblib, numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import BadRequestError

from agents.fact_checker import FactCheckerAgent
from agents.search_agent import SearchAgent
from agents.classify_agent import ClassifyAgent
from agents.legality_checker import LegalityChecker
from agents.authentiate_agent import AuthenticateAgent
from model.llm_model import AzureModel

# --- Load environment ---
load_dotenv()
endpoint = os.getenv("END_POINT")
deployment = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("API_KEY")

# --- Setup LLM + Agents ---
model = AzureModel(endpoint=endpoint, deployment=deployment, api_key=api_key)
fact_checker = FactCheckerAgent(
    model, instruction_prompt=Path("./system_prompt/fact_checker.txt")
)
search_agent = SearchAgent(max_results=3)
classify_agent = ClassifyAgent(
    model, instruction_prompt=Path("./system_prompt/classify_prompt.txt"), max_tokens=20
)
legal_agent = LegalityChecker(
    model, instruction_prompt=Path("./system_prompt/legal.txt")
)
authentication_agent = AuthenticateAgent(
    model, instruction_prompt=Path("./system_prompt/authenticate.txt"), max_tokens=10
)

# --- Load classifier model ---
BEST_MODEL_PATH = "./best_models/SVC_best.pkl"
clf = joblib.load(BEST_MODEL_PATH)

# --- Flask app ---
app = Flask(__name__)


def parse_agent_output(raw_output):
    """Chuyển output agent thành dict an toàn"""
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw_output)
        except Exception:
            return {}


def predict_from_confidence(classify_conf, fact_conf, auth_conf):
    """Dự đoán label từ 3 giá trị confidence"""
    X_input = np.array([[classify_conf, fact_conf, auth_conf]])
    return clf.predict(X_input)[0]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prediction = None
    error = None

    if request.method == "POST":
        statement = request.form.get("statement")
        metadata = {
            "subject": request.form.get("subject"),
            "speaker": request.form.get("speaker"),
            "job_title": request.form.get("job_title"),
            "state_info": request.form.get("state_info"),
            "party_affiliation": request.form.get("party_affiliation"),
            "context": request.form.get("context"),
        }

        # Nếu metadata trống -> authenticate = 1
        auth_conf = (
            1.0 if all(not v or v.strip() == "" for v in metadata.values()) else None
        )

        try:
            # Legality check
            if not legal_agent.check(statement=statement):
                error = "❌ Violation of OpenAI policy"
            else:
                # Classify agent
                classify_response = parse_agent_output(
                    classify_agent.classify(statement=statement)
                )
                classify_conf = classify_response.get("confidence", 0.0)

                # Search + Fact check agent
                search_results = search_agent.search(query=statement)
                fact_response = parse_agent_output(
                    fact_checker.check(
                        statement=statement, search_results=search_results
                    )
                )
                fact_conf = fact_response.get("confidence", 0.0)

                # Authenticate agent
                auth_confidence = (
                    auth_conf
                    if auth_conf is not None
                    else parse_agent_output(
                        authentication_agent.check(metadata=metadata)
                    ).get("confidence", 0.0)
                )

                # Lưu 3 confidence
                result = {
                    "Classify Confidence": classify_conf,
                    "Fact Confidence": fact_conf,
                    "Authenticate Confidence": auth_confidence,
                }

                # Predict label
                prediction = predict_from_confidence(
                    classify_conf, fact_conf, auth_confidence
                )

        except BadRequestError:
            error = "❌ Violation of OpenAI policy"
        except Exception as e:
            error = f"❌ Unexpected error: {e}"

    return render_template(
        "index.html", result=result, prediction=prediction, error=error
    )


if __name__ == "__main__":
    app.run(debug=False)
