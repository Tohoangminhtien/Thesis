import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
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
# Dictionary lưu kết quả accuracy của từng mô hình
model_accuracies = {}

for model_name, model in models.items():
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    mean_acc = np.mean(accuracies)
    model_accuracies[model_name] = mean_acc * 100
    print(f"{model_name}: {mean_acc * 100:.2f}%")

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.bar(model_accuracies.keys(), model_accuracies.values(), color="skyblue")
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison (10-Fold CV Accuracy)")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 100)
plt.tight_layout()

# Lưu hình
plt.savefig("./assets/model_accuracy_comparison.png")
plt.show()
