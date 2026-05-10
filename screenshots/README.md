# Screenshots

Captured artifacts (auto-generated where possible):

| File | What it shows |
| --- | --- |
| `swagger-ui.png` | FastAPI auto-generated Swagger UI listing all endpoints |
| `api-metrics.png` | `/metrics` Prometheus exposition output from the live API |
| `mlflow-experiment.png` | MLflow experiment landing page |
| `mlflow-runs.png` | MLflow Training-runs view with both `logistic_regression` and `random_forest` runs |

Add when running on Kubernetes / CI:

| File | What it shows |
| --- | --- |
| `ci.png` | GitHub Actions workflow run (lint → train → tests → docker build) |
| `k8s-pods.png` | `kubectl get pods` output |
| `k8s-service.png` | `kubectl get svc heart-api` showing the LoadBalancer |
| `predict.png` | `curl -X POST .../predict` returning prediction JSON |
| `grafana.png` | Grafana dashboard with request-rate / latency panels |
| `prometheus.png` | Prometheus targets / heart-api job up |
