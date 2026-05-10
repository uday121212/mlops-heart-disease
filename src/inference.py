"""Load the trained model and run a prediction on a sample input."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"


def load_model(path: Path | str = DEFAULT_MODEL_PATH):
    return joblib.load(path)


def predict_one(model, payload: dict[str, Any]) -> dict[str, Any]:
    df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    proba = float(model.predict_proba(df)[0, 1])
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability": proba}


SAMPLE = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
    "ca": 0, "thal": 1,
}


if __name__ == "__main__":
    m = load_model()
    print(predict_one(m, SAMPLE))
