import os
import pandas as pd

DATA_PATH = "data/raw/heart.csv"

def test_data_file_exists():
    assert os.path.exists(DATA_PATH), "Dataset file does not exist"

def test_data_loads_correctly():
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Dataset is empty"

def test_target_column_exists():
    df = pd.read_csv(DATA_PATH)
    assert "num" in df.columns, "Target column AHD missing"
