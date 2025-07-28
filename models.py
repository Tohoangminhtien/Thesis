import json
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer

# Import mô hình
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Load dữ liệu
df = pd.read_csv("./results/agent_results.csv")
df["label"] = df["label"].astype(int)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Load config từ JSON
with open("./grid_search/param.json", "r") as f:
    config = json.load(f)

# Model map
model_map = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "KNeighborsClassifier": KNeighborsClassifier,
    "GaussianNB": GaussianNB,
    "XGBClassifier": XGBClassifier,
    "MLPClassifier": MLPClassifier,
}

# KFold strategy
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lưu kết quả để vẽ biểu đồ
model_names = []
accuracy_scores = []
f1_scores_avg = []

for model_name, model_info in config.items():
    model_class = model_map[model_info["model"]]

    param_grid = model_info["params"]
    for k, v in param_grid.items():
        param_grid[k] = [None if val is None or val == "null" else val for val in v]

    print(f"\n🔍 Running GridSearch for {model_name}...")

    grid = GridSearchCV(
        estimator=model_class(),
        param_grid=param_grid,
        cv=skf,
        scoring=make_scorer(f1_score, average="binary"),
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    # Đánh giá lại trên K-Fold
    acc_scores = []
    f1_scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average="binary"))

    mean_acc = np.mean(acc_scores) * 100
    mean_f1 = np.mean(f1_scores) * 100

    print(f"✅ Best Params: {best_params}")
    print(f"📈 Accuracy: {mean_acc:.2f}%")
    print(f"📊 F1 Score: {mean_f1:.2f}%")

    # Lưu mô hình tốt nhất
    os.makedirs("./best_models", exist_ok=True)
    joblib.dump(best_model, f"./best_models/{model_info['model']}_best.pkl")

    # Lưu để vẽ biểu đồ
    model_names.append(model_name)
    accuracy_scores.append(mean_acc)
    f1_scores_avg.append(mean_f1)

# Vẽ biểu đồ sau khi tất cả mô hình đã chạy xong
x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(14, 6))
plt.bar(x - width / 2, accuracy_scores, width, label="Accuracy", color="skyblue")
plt.bar(x + width / 2, f1_scores_avg, width, label="F1 Score", color="salmon")

plt.ylabel("Score (%)")
plt.title(
    "Model Comparison: Accuracy vs. F1 Score (Binary Classification - 10-Fold CV)"
)
plt.xticks(x, model_names, rotation=45)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

os.makedirs("./assets", exist_ok=True)
plt.savefig("./assets/model_accuracy_comparison.png", dpi=300)
plt.show()
