# End-to-End MLOps Pipeline for Heart Disease Prediction

## Project Overview
This project implements an end-to-end **MLOps pipeline** for predicting the risk of heart disease using patient health data.  
It demonstrates modern MLOps practices including data preprocessing, experiment tracking, CI/CD automation, containerized model serving, and production-style deployment.

This project is developed as part of the **MLOps Experimental Learning Assignment**.

---

## Problem Statement
Given clinical attributes such as age, cholesterol level, blood pressure, and ECG results, the goal is to build a machine learning classifier that predicts the **presence or absence of heart disease**.

---

## Dataset
- **Dataset Name:** Heart Disease UCI Dataset  
- **Source:** UCI Machine Learning Repository  
- **Target Variable:** Presence of heart disease (binary)

---

## Project Structure
```
mlops-uci-heart-disease/
│   .dockerignore
│   .gitignore
│   Dockerfile
│   LICENSE
│   mlflow.db
│   prometheus.yml
│   README.md
│   requirements.txt
│
├───.github
│   └───workflows
│           ci.yml
│
├───.pytest_cache
│   │   .gitignore
│   │   CACHEDIR.TAG
│   │   README.md
│   │
│   └───v
│       └───cache
│               lastfailed
│               nodeids
│
├───app
│   │   main.py
│   │
│   └───__pycache__
│           main.cpython-310.pyc
│
├───artifacts
│       model.pkl
│       scaler.pkl
│
├───data
│   └───raw
│           heart.csv
│
├───helm
│   └───heart-disease
│       │   .helmignore
│       │   Chart.yaml
│       │   values.yaml
│       │
│       ├───charts
│       └───templates
│               deployment.yaml
│               service.yaml
│
├───k8s
│       deployment.yaml
│       ingress.yaml
│       service.yaml
│
├───mlruns
│   └───1
│       └───models
│           ├───m-3a5314de0cbb4d66ba3d9d8ffaf614c5
│           │   └───artifacts
│           │           conda.yaml
│           │           MLmodel
│           │           model.pkl
│           │           python_env.yaml
│           │           requirements.txt
│           │
│           ├───m-86f2ea915dfa4cd38060c13d638bc360
│           │   └───artifacts
│           │           conda.yaml
│           │           MLmodel
│           │           model.pkl
│           │           python_env.yaml
│           │           requirements.txt
│           │
│           └───m-c05e2439b78a45a4855166db3eae3f40
│               └───artifacts
│                       conda.yaml
│                       MLmodel
│                       model.pkl
│                       python_env.yaml
│                       requirements.txt
│
├───notebooks
│       01_EDA.ipynb
│
├───src
│   ├───data
│   │       download_data.py
│   │
│   ├───features
│   ├───models
│   │       train.py
│   │
│   └───utils
└───tests
    │   test_data.py
    │   test_model.py
    │
    └───__pycache__
            test_data.cpython-310-pytest-9.0.2.pyc
            test_model.cpython-310-pytest-9.0.2.pyc
```

---

## 4. Setup Instructions

### 4.1 Clone Repository

```
bash
	git clone <repository-url>
	cd mlops-uci-heart-disease
```

### 4.2 Install Dependencies

```
bash
	pip install -r requirements.txt
```

---

## 5. Data Acquisition & EDA

### Download Dataset
```
bash
python src/data/download_data.py
```

### Exploratory Data Analysis

EDA is performed in:

```
notebooks/01_eda.ipynb
```

Includes:

- Missing value analysis

- Feature distributions

- Class balance

- Correlation heatmap

---

## 6. Feature Engineering & Model Training

### Models implemented:

- Logistic Regression

- Random Forest

- XGBoost

Training script:

```
python src/models/train.py
```

Artifacts generated:
- artifacts/model.pkl

- artifacts/scaler.pkl

---

## 7. Experiment Tracking (MLflow)

MLflow is used to track:

- Model parameters

- Evaluation metrics

- Trained models

### Start MLflow UI

```
mlflow ui
```

Access:

```
http://127.0.0.1:5000
```
---

## 8. Unit Testing

Unit tests are written using Pytest.

### Run tests:

```
pytest
```

Tests cover:

- Data availability and schema

- Model artifact creation

- Model loading

---

## 9. CI/CD Pipeline

CI pipeline is implemented using **GitHub Actions**.

Pipeline steps:

- Dependency installation

- Linting

- Unit testing

- Model training

- Artifact upload per workflow run

Workflow file:

```
.github/workflows/ci.yml
```
---

## 10. FastAPI Inference API

### Run locally

```
uvicorn app.main:app --reload
```
Available Endpoints
- / – Health check

- /predict – Model inference

- /metrics – Monitoring metrics (Prometheus compatible)

Swagger UI:

arduino
```
http://127.0.0.1:8000/docs
```

---

## 11. Docker Containerization

### Build Image

```
docker build -t heart-disease-api .
```

### Run Container

```
docker run -p 8000:8000 heart-disease-api
```

---

## 12. Kubernetes Deployment

### Apply manifests

```
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Access API
Docker Desktop: http://localhost/docs

Minikube:

```
minikube service heart-disease-service
```
---

## 13. Helm Deployment (Alternative)

### Install Helm chart

```
helm install heart-api helm/heart-disease
```

### Verify

```
helm list
kubectl get pods
kubectl get services
```

---

## 14. Monitoring & Logging

### Logging
- Structured logging implemented in FastAPI
- Logs accessible via:

```
docker logs <container-id>
kubectl logs <pod-name>
```

### Monitoring
- /metrics endpoint exposes:

	- Total request count

	- Average prediction latency

- Prometheus scrapes metrics

- Grafana visualizes metrics using dashboards

---

## 15. Production Readiness
- Reproducible environment using requirements.txt

- Containerized deployment

- Automated CI pipeline

- Scalable Kubernetes setup

- Monitoring and logging enabled

---

## 16. Deliverables

- GitHub repository with full source code

- CI/CD pipeline with artifacts

- Docker image and Kubernetes manifests

- Helm chart

- Monitoring setup

- Final report and screenshots

---


## Author
- **Name:** Moulik Dayal
- **Roll No:** 2024AA05811
