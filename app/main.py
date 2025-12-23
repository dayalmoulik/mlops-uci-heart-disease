# Import necessary libraries
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# pickle files for model and scaler
MODEL_PATH = 'artifacts/model.pkl'
SCALER_PATH = 'artifacts/scaler.pkl'

# Load the pre-trained model & scaler
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)
with open(SCALER_PATH, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Add in-memory metrics
REQUEST_COUNT = 0
TOTAL_PREDICTION_TIME = 0.0

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

#Input Schema

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    chest_pain: int
    resting_bp: int
    chol: int
    fasting_bs: int
    rest_ecg: int
    max_hr: int
    exercise_angina: int
    oldpeak: float
    st_slope: int
    Ca: int
    thal: int

# Health Check 
@app.get("/")
def health_check():
    logger.info("Health check requested")
    return {"status": "API is running"}

# Prediction Endpoint
@app.post("/predict")
def predict_heart_disease(input_data: HeartDiseaseInput):
    global REQUEST_COUNT, TOTAL_PREDICTION_TIME

    start_time = time.time()
    REQUEST_COUNT += 1

    logger.info(f"Prediction request received: {input_data.model_dump()}")
    # Convert input data to numpy array
    input_array = np.array([[input_data.age, input_data.sex, input_data.chest_pain,
                             input_data.resting_bp, input_data.chol, input_data.fasting_bs,
                             input_data.rest_ecg, input_data.max_hr, input_data.exercise_angina,
                                input_data.oldpeak, input_data.st_slope, input_data.Ca, input_data.thal]])
    # Scale the input data
    scaled_data = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)[0][1]

    elapsed_time = time.time() - start_time
    TOTAL_PREDICTION_TIME += elapsed_time

    logger.info(
        f"Prediction completed"
        f"Prediction: {prediction}"
        f"Probability: {prediction_proba:.4f}, Time taken: {elapsed_time:.4f} seconds")    
  
    # Return the prediction result
    return {
        "prediction": int(prediction),
        "probability": round(float(prediction_proba),4),
        "latency_seconds": round(elapsed_time, 4)
    }

# Metrics Endpoint
@app.get("/metrics")
def metrics():
    avg_latency = (
        TOTAL_PREDICTION_TIME/REQUEST_COUNT 
        if REQUEST_COUNT > 0 else 0
    )

    return {
        "total_requests": REQUEST_COUNT,
        "average_latency_seconds": round(avg_latency, 4)
    }