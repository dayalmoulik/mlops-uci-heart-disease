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

## Author
- **Name:** Moulik Dayal
- **Roll No:** 2024AA05811
