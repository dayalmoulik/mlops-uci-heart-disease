#Import necessary libraries
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import pickle
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

#Defining Data Path & Artifact Path
DATA_PATH = "data/raw/heart.csv"
ARTIFACT_PATH = "artifacts"

os.makedirs(ARTIFACT_PATH, exist_ok=True)

# Loading the dataset
data = pd.read_csv(DATA_PATH)

# Handling missing values
data['ca']=data['ca'].replace('?',np.nan)
data['thal']=data['thal'].replace('?',np.nan)
data['ca'].fillna(data['ca'].mode()[0], inplace=True)
data['thal'].fillna(data['thal'].mode()[0], inplace=True)

# Splitting features and target variable
X = data.drop('num', axis=1)
y = data['num'].apply(lambda x: 1 if x > 0 else 0) # Binary classification

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Saving the scaler
with open(os.path.join(ARTIFACT_PATH, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Model Training
# Model 1: Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Model 3: XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# Evaluating Models
def evaluate_model(name,y_true, y_pred, y_probs):
    print(f"Evaluating {name}...")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    if name == "Logistic Regression":
        model = lr_model
    elif name == "Random Forest":
        model = rf_model
    else:
        model = xgb_model
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")    
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Cross-Validation Scores: {cross_val_scores}")
    print(f"Mean CV Score: {cross_val_scores.mean():.4f}")
    print("-" * 30)

evaluate_model("Logistic Regression", y_test, lr_pred, lr_probs)
evaluate_model("Random Forest", y_test, rf_pred, rf_probs)
evaluate_model("XGBoost", y_test, xgb_pred, xgb_probs)


# Saving the best model
with open(os.path.join(ARTIFACT_PATH, "model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)

