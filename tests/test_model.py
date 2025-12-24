import os
import pickle

MODEL_PATH = "artifacts/model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

def test_model_artifact_exists():
    assert os.path.exists(MODEL_PATH), "Model artifact not found"

def test_scaler_artifact_exists():
    assert os.path.exists(SCALER_PATH), "Scaler artifact not found"

def test_model_can_be_loaded():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    assert model is not None, "Model could not be loaded"
