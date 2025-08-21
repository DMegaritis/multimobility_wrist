"""
Loads package CSV data.
"""

from importlib import resources
import pandas as pd

def load_imu_data() -> pd.DataFrame:
    with resources.open_text("multimob.data", "data.csv") as f:
        return pd.read_csv(f)

def load_ICs() -> pd.DataFrame:
    with resources.open_text("multimob.data", "ics.csv") as f:
        return pd.read_csv(f)
