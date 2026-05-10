"""Train heart-disease classifiers, log to MLflow, and persist the best model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, build_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease_clean.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "heart_disease_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run: python -m src.download_data"
        )
    return pd.read_csv(DATA_PATH)


def evaluate(model, X, y) -> dict:
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds)),
        "recall": float(recall_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def train_and_log(name: str, estimator, param_grid: dict, X_train, y_train, X_test, y_test):
    pipe = build_pipeline(estimator)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)

    with mlflow.start_run(run_name=name) as run:
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        metrics = evaluate(best, X_test, y_test)
        cv_score = float(grid.best_score_)

        mlflow.log_params({f"{name}__{k}": v for k, v in grid.best_params_.items()})
        mlflow.log_metric("cv_roc_auc", cv_score)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(best, artifact_path="model")
        print(f"[{name}] cv_roc_auc={cv_score:.4f} test={metrics}")
        return {"name": name, "estimator": best, "cv_roc_auc": cv_score, "metrics": metrics, "run_id": run.info.run_id}


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")
    mlflow.set_experiment("heart-disease")

    candidates = [
        train_and_log(
            "logistic_regression",
            LogisticRegression(max_iter=2000, solver="liblinear"),
            {"model__C": [0.1, 1.0, 10.0], "model__penalty": ["l1", "l2"]},
            X_train, y_train, X_test, y_test,
        ),
        train_and_log(
            "random_forest",
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
            },
            X_train, y_train, X_test, y_test,
        ),
    ]

    best = max(candidates, key=lambda c: c["cv_roc_auc"])
    print(f"Best model: {best['name']} (cv_roc_auc={best['cv_roc_auc']:.4f})")

    joblib.dump(best["estimator"], MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

    summary = {
        "best_model": best["name"],
        "cv_roc_auc": best["cv_roc_auc"],
        "test_metrics": best["metrics"],
        "candidates": [
            {"name": c["name"], "cv_roc_auc": c["cv_roc_auc"], "test_metrics": c["metrics"]}
            for c in candidates
        ],
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics -> {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
