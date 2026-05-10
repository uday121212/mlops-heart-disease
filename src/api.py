"""FastAPI service for heart disease prediction."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.preprocessing import FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("heart-api")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "heart_disease_model.joblib"))

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict the risk of heart disease from patient features.",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model file not found: {MODEL_PATH}")
        logger.info("Loading model from %s", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model


class HeartFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=0)
    chol: float = Field(..., ge=0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float
    slope: int = Field(..., ge=0, le=2)
    ca: float = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    confidence: float


@app.on_event("startup")
def _startup() -> None:
    try:
        get_model()
        logger.info("Model loaded successfully at startup")
    except Exception as e:  # pragma: no cover
        logger.warning("Model not loaded at startup: %s", e)


@app.get("/")
def root():
    return {
        "service": "heart-disease-api",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/predict/batch", "/metrics", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HeartFeatures):
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    df = pd.DataFrame([features.model_dump()], columns=FEATURE_COLUMNS)
    proba = float(model.predict_proba(df)[0, 1])
    pred = int(proba >= 0.5)
    label = "disease" if pred == 1 else "no_disease"
    confidence = proba if pred == 1 else 1.0 - proba
    logger.info("Prediction: pred=%s prob=%.4f", pred, proba)
    return PredictionResponse(
        prediction=pred, probability=proba, label=label, confidence=confidence
    )


@app.post("/predict/batch")
def predict_batch(items: List[HeartFeatures]):
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    df = pd.DataFrame([i.model_dump() for i in items], columns=FEATURE_COLUMNS)
    probas = model.predict_proba(df)[:, 1].tolist()
    preds = [int(p >= 0.5) for p in probas]
    return {"predictions": preds, "probabilities": probas, "count": len(preds)}
