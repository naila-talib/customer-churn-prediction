from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "features.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
GRAPH_DIR = BASE_DIR / "static" / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
info = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
TARGET = info["target"]
FEATURES = info["features"]

df = df[df[TARGET].isin(["Stayed", "Churned"])].copy()
X = df[FEATURES].copy()
y = df[TARGET].map({"Stayed": 0, "Churned": 1})

for col in X.columns:
    if X[col].dtype == "object":
        cleaned = X[col].astype(str).str.replace(r"[$,]", "", regex=True)
        converted = pd.to_numeric(cleaned, errors="coerce")
        if converted.notna().mean() >= 0.90:
            X[col] = converted

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = joblib.load(MODEL_PATH)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

results = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, proba)),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "churn_rate": float(y.mean())
}

payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
payload["best_model_metrics"] = results
METRICS_PATH.write_text(json.dumps(payload, indent=4), encoding="utf-8")

# Churn distribution
counts = df[TARGET].value_counts().reindex(["Stayed", "Churned"]).fillna(0)
plt.figure(figsize=(7, 5))
plt.bar(counts.index, counts.values)
plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "churn_distribution.png", dpi=200)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Churned"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax)
ax.set_title("Confusion Matrix")
fig.tight_layout()
fig.savefig(GRAPH_DIR / "confusion_matrix.png", dpi=200)
plt.close(fig)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {results['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(GRAPH_DIR / "roc_curve.png", dpi=200)
plt.close()

# Feature importance where supported
try:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        imp = imp.sort_values("Importance", ascending=False).head(15)

        plt.figure(figsize=(9, 7))
        plt.barh(imp["Feature"][::-1], imp["Importance"][::-1])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Top 15 Features Influencing Churn")
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / "feature_importance.png", dpi=200)
        plt.close()
except Exception as e:
    print("Feature importance graph skipped:", e)

print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)
for k, v in results.items():
    print(f"{k}: {v}")
print(f"Graphs saved in: {GRAPH_DIR}")
