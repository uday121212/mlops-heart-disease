import pandas as pd

from src.preprocessing import FEATURE_COLUMNS
from src.train import evaluate
from sklearn.linear_model import LogisticRegression
from src.preprocessing import build_pipeline


def test_evaluate_returns_metrics():
    df = pd.DataFrame({
        "age": [63, 50, 41, 55, 60, 45, 70, 35],
        "sex": [1, 0, 1, 0, 1, 0, 1, 0],
        "cp": [3, 2, 1, 0, 3, 2, 1, 0],
        "trestbps": [145, 130, 120, 140, 150, 110, 160, 125],
        "chol": [233, 250, 180, 220, 280, 200, 300, 210],
        "fbs": [1, 0, 0, 1, 0, 0, 1, 0],
        "restecg": [0, 1, 0, 2, 1, 0, 1, 2],
        "thalach": [150, 160, 170, 140, 130, 175, 120, 165],
        "exang": [0, 1, 0, 1, 0, 0, 1, 0],
        "oldpeak": [2.3, 1.0, 0.5, 1.5, 2.0, 0.0, 3.0, 0.8],
        "slope": [0, 1, 2, 1, 0, 2, 0, 1],
        "ca": [0, 1, 0, 2, 3, 0, 1, 0],
        "thal": [1, 2, 3, 1, 2, 3, 1, 2],
    })
    y = [1, 0, 0, 1, 1, 0, 1, 0]
    pipe = build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(df[FEATURE_COLUMNS], y)
    metrics = evaluate(pipe, df[FEATURE_COLUMNS], y)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
