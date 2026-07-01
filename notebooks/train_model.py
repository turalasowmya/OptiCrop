"""
OptiCrop - Model Exploration & Training
Compares KNN, Logistic Regression, Decision Tree, and Random Forest
on the crop recommendation dataset, then saves the best model.
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())
print("\nClass counts:\n", df["label"].value_counts())

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

X = df[FEATURES]
y = df[TARGET]

# ---------------------------------------------------------
# 2. EDA plots (saved to disk, not shown interactively)
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df[FEATURES].corr(), annot=True, cmap="YlGnBu")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "correlation_heatmap.png"))
plt.close()

plt.figure(figsize=(14, 8))
df[FEATURES].boxplot()
plt.title("Feature Distributions")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "feature_boxplots.png"))
plt.close()

# ---------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (needed for KNN and Logistic Regression; harmless for trees)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Train & compare models
# ---------------------------------------------------------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}

for name, model in models.items():
    # Tree-based models don't need scaling, but using scaled data doesn't hurt them
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    results[name] = {
        "model": model,
        "test_accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }
    print(f"\n=== {name} ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(classification_report(y_test, preds))

# ---------------------------------------------------------
# 5. Pick best model by test accuracy
# ---------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["test_accuracy"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (Test Accuracy: {results[best_name]['test_accuracy']:.4f})")

# Confusion matrix for the best model
preds_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, preds_best, labels=sorted(y.unique()))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.close()

# ---------------------------------------------------------
# 6. Save best model + scaler for Flask app
# ---------------------------------------------------------
with open(os.path.join(MODEL_DIR, "crop_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(f"\nSaved {best_name} -> model/crop_model.pkl")
print("Saved StandardScaler -> model/scaler.pkl")

# Save a summary of model comparison
summary = pd.DataFrame({
    name: {"test_accuracy": r["test_accuracy"], "cv_mean": r["cv_mean"]}
    for name, r in results.items()
}).T
summary.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"))
print("\nModel comparison saved -> model/model_comparison.csv")
print(summary)
