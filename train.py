from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "churn_model.pkl"
FEATURES_PATH = MODEL_DIR / "features.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. TARGET COLUMN
# ============================================================

TARGET = "Customer Status"

# We only use customers whose final status is known
df = df[df[TARGET].isin(["Stayed", "Churned"])].copy()

print(f"Records used for training: {len(df)}")


# ============================================================
# 4. FEATURES
# ============================================================

# These columns must NOT be used because they leak the answer.
excluded_columns = [
    "Customer ID",
    "Customer Status",
    "Churn Category",
    "Churn Reason"
]

FEATURES = [
    column
    for column in df.columns
    if column not in excluded_columns
]

X = df[FEATURES].copy()

# Churned = 1
# Stayed  = 0

y = df[TARGET].map({
    "Stayed": 0,
    "Churned": 1
})


# ============================================================
# 5. CONVERT NUMERIC-LOOKING COLUMNS
# ============================================================

print("\nPreparing data...")

for column in X.columns:

    if X[column].dtype == "object":

        cleaned = (
            X[column]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )

        converted = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

        # If at least 90% of values can be numeric,
        # treat the column as numeric.
        if converted.notna().mean() >= 0.90:
            X[column] = converted


# ============================================================
# 6. IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# ============================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    column
    for column in X.columns
    if column not in numeric_features
]

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 9. MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
}


# ============================================================
# 10. TRAIN MODELS
# ============================================================

results = {}

best_model = None
best_model_name = None
best_f1 = -1


for name, algorithm in models.items():

    print("\n" + "-" * 60)
    print(f"Training: {name}")
    print("-" * 60)

    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            algorithm
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    results[name] = {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1": float(f1),

        "roc_auc": float(roc_auc)
    }


    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


    # Select best model based on F1 score
    if f1 > best_f1:

        best_f1 = f1

        best_model = pipeline

        best_model_name = name


# ============================================================
# 11. SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Best model: {best_model_name}")

joblib.dump(
    best_model,
    MODEL_PATH
)

print(f"\nModel saved successfully:")
print(MODEL_PATH)


# ============================================================
# 12. SAVE FEATURES
# ============================================================

feature_information = {

    "target": TARGET,

    "positive_class": "Churned",

    "negative_class": "Stayed",

    "features": FEATURES,

    "categorical_fields": categorical_features,

    "numeric_fields": numeric_features
}

FEATURES_PATH.write_text(
    json.dumps(
        feature_information,
        indent=4
    ),
    encoding="utf-8"
)


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics_information = {

    "best_model": best_model_name,

    "best_model_metrics":
        results[best_model_name],

    "all_models":
        results,

    "training_records":
        len(X_train),

    "testing_records":
        len(X_test),

    "churn_rate":
        float(y.mean())
}


METRICS_PATH.write_text(
    json.dumps(
        metrics_information,
        indent=4
    ),
    encoding="utf-8"
)


# ============================================================
# 14. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nCreated files:")

print("✅ models/churn_model.pkl")
print("✅ models/features.json")
print("✅ models/metrics.json")

print("\nYou can now run:")
print("python evaluation.py")

print("After evaluation:")
print("python app.py")

print("=" * 60)