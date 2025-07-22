from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from collections import Counter


def evaluate_method(csv_file: str):

    df = pd.read_csv(csv_file)
    y = df["label"].tolist()
    y_pred = df["predict"].tolist()
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred, average="weighted"))
    print("Recall:", recall_score(y, y_pred, average="weighted"))
    print("F1: ", f1_score(y, y_pred, average="weighted"))
    print("-" * 50)


evaluate_method("./data/zero_shot_results.csv")
evaluate_method("./data/few_shot_results.csv")
