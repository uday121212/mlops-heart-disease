import os
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.preprocessing import build_pipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
    "ca": 0, "thal": 1,
}


@pytest.fixture(scope="module", autouse=True)
def _ensure_model(tmp_path_factory):
    model_path = ROOT / "models" / "heart_disease_model.joblib"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([SAMPLE] * 40)
        y = [0, 1] * 20
        pipe = build_pipeline(LogisticRegression(max_iter=200))
        pipe.fit(df, y)
        joblib.dump(pipe, model_path)
    os.environ["MODEL_PATH"] = str(model_path)
    yield


@pytest.fixture(scope="module")
def client():
    from src.api import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("disease", "no_disease")


def test_predict_batch(client):
    r = client.post("/predict/batch", json=[SAMPLE, SAMPLE])
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_validation_error(client):
    bad = {**SAMPLE, "sex": 9}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_request" in r.text or "process_" in r.text
