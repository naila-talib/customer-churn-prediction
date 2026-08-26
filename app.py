from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "features.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
info = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

FEATURES = info["features"]

# These fields are categorical even if a CSV reader or preprocessing step
# represents them unexpectedly. They are always shown as dropdowns.
CATEGORICAL_FIELDS = [
    "Gender",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Multiple Lines",
    "Phone Service",
]

df_options = pd.read_csv(BASE_DIR / "data" / "customer_churn.csv")

OPTIONS = {}
for col in FEATURES:
    if col in CATEGORICAL_FIELDS:
        values = (
            df_options[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        OPTIONS[col] = sorted(values)

DEFAULTS = {}
for col in FEATURES:
    if col in OPTIONS:
        DEFAULTS[col] = OPTIONS[col][0]
    else:
        series = pd.to_numeric(df_options[col], errors="coerce")
        DEFAULTS[col] = float(series.median()) if series.notna().any() else 0


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    risk = None
    error = None
    values = DEFAULTS.copy()

    if request.method == "POST":
        try:
            for feature in FEATURES:
                raw = request.form.get(feature, "").strip()

                if feature in OPTIONS:
                    values[feature] = raw if raw else DEFAULTS[feature]
                else:
                    values[feature] = (
                        float(raw) if raw != "" else DEFAULTS[feature]
                    )

            input_data = pd.DataFrame([values], columns=FEATURES)

            prediction = int(model.predict(input_data)[0])
            probability = float(model.predict_proba(input_data)[0][1] * 100)

            if probability >= 70:
                risk = "HIGH"
            elif probability >= 40:
                risk = "MEDIUM"
            else:
                risk = "LOW"

        except ValueError:
            error = "Please enter valid values in the numeric fields."
        except Exception as e:
            error = str(e)
            print("Prediction error:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        risk=risk,
        error=error,
        values=values,
        options=OPTIONS,
        features=FEATURES,
        metrics=metrics,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
