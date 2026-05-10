# MLOps Heart Disease Prediction

End-to-end MLOps project that trains, packages, deploys, and monitors a binary classifier predicting heart disease risk from the UCI Heart Disease (Cleveland) dataset.

[![CI/CD](https://github.com/uday121212/mlops-heart-disease/actions/workflows/ci.yml/badge.svg)](./.github/workflows/ci.yml)

## Architecture

```
┌────────────┐   ┌───────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
│ UCI Data   │──▶│ Train +   │──▶│ MLflow       │   │ FastAPI     │──▶│ Prometheus + │
│ (download) │   │ GridSearch│   │ Experiments  │   │ /predict    │   │ Grafana      │
└────────────┘   └─────┬─────┘   └──────────────┘   └──────┬──────┘   └──────────────┘
                       │                                    │
                       ▼                                    ▼
                 models/*.joblib                   Docker → Kubernetes (Helm)
```

## Repository Layout

| Path | Description |
| --- | --- |
| [src/](src) | Source code (data, preprocessing, training, API, inference) |
| [tests/](tests) | Pytest suite (unit + API integration) |
| [notebooks/](notebooks) | EDA + training notebook |
| [data/](data) | Raw + cleaned CSV (downloaded) |
| [models/](models) | Persisted joblib model |
| [reports/](reports) | Metrics + figures |
| [k8s/](k8s) | Kubernetes manifests |
| [helm/heart-api/](helm/heart-api) | Helm chart |
| [monitoring/](monitoring) | Prometheus + Grafana provisioning |
| [docs/](docs) | Architecture & report |
| [.github/workflows/](.github/workflows) | CI/CD pipeline |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## End-to-end run

```bash
# 1. Download + clean dataset
python -m src.download_data

# 2. Train + log to MLflow + persist best model
python -m src.train

# 3. Run unit tests
pytest

# 4. Serve API locally
uvicorn src.api:app --reload --port 8000
# open http://localhost:8000/docs

# 5. Sample prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

## MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
# http://localhost:5000
```

## Docker

```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest
```

Compose (API + Prometheus + Grafana):

```bash
docker compose up --build
# Grafana: http://localhost:3000  (admin / admin)
# Prometheus: http://localhost:9090
```

## Kubernetes

Plain manifests:

```bash
kubectl apply -f k8s/
kubectl get svc heart-api
```

Helm:

```bash
helm install heart-api ./helm/heart-api
kubectl port-forward svc/heart-api-heart-api 8080:80
```

## CI/CD

GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs lint → data download → training → tests → coverage → Docker build & smoke test on every push/PR.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness/readiness check |
| POST | `/predict` | Single prediction (JSON body) |
| POST | `/predict/batch` | Batch prediction |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

## Report

See [docs/REPORT.md](docs/REPORT.md) for the full project report.
