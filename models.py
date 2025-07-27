import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

# Đọc dữ liệu
df = pd.read_csv("./results/agent_results.csv")
df["label"] = df["label"].astype(int)
df = df.values

# Tách features và labels
X = df[:, :-1]
y = df[:, -1]

# Danh sách các mô hình
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", gamma="scale", random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
}

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
model_accuracies = {}
model_f1_scores = {}

for model_name, model in models.items():
    accuracies = []
    f1_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Binary classification nên dùng average="binary"
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average="weighted"))

    model_accuracies[model_name] = np.mean(accuracies) * 100
    model_f1_scores[model_name] = np.mean(f1_scores) * 100

    print(
        f"{model_name}: Accuracy = {model_accuracies[model_name]:.2f}%, F1 Score = {model_f1_scores[model_name]:.2f}%"
    )

# Vẽ biểu đồ
labels = list(model_accuracies.keys())
accuracy_vals = [model_accuracies[name] for name in labels]
f1_vals = [model_f1_scores[name] for name in labels]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width / 2, accuracy_vals, width, label="Accuracy", color="skyblue")
plt.bar(x + width / 2, f1_vals, width, label="F1 Score", color="salmon")

plt.ylabel("Score (%)")
plt.title(
    "Model Comparison: Accuracy vs. F1 Score (Binary Classification - 10-Fold CV)"
)
plt.xticks(x, labels, rotation=45, ha="right")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

# Lưu hình
plt.savefig("./assets/model_accuracy_comparison.png")
plt.show()
