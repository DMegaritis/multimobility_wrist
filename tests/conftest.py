import importlib
import pytest
import pandas as pd
import numpy as np

def import_class(module_path: str, class_name: str):
    """Dynamically import a class from a module path."""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def _as_df_or_empty(data):
    """Ensure output is a pandas DataFrame, or return empty DataFrame if None/empty."""
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    try:
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

def _synthesize_walk(n_samples=1000, sampling_rate_hz=100):
    """Generate synthetic wrist IMU walking data without rest periods."""
    import numpy as np
    import pandas as pd

    t = np.arange(n_samples) / sampling_rate_hz
    rng = np.random.default_rng(42)

    # Acceleration signals (wrist swinging)
    acc_is = 10 + 5 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.1, n_samples)
    acc_ml = -1 + 5 * np.sin(2 * np.pi * 1.0 * t + np.pi/4) + rng.normal(0, 0.1, n_samples)
    acc_pa = -1 + 5 * np.sin(2 * np.pi * 1.0 * t + np.pi/2) + rng.normal(0, 0.1, n_samples)

    # Gyroscope signals (simulate wrist rotation)
    gyr_is = 10 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.5, n_samples)
    gyr_ml = 10 * np.sin(2 * np.pi * 1.0 * t + np.pi/4) + rng.normal(0, 0.5, n_samples)
    gyr_pa = 10 * np.sin(2 * np.pi * 1.0 * t + np.pi/2) + rng.normal(0, 0.5, n_samples)

    imu_data = pd.DataFrame({
        "acc_is": acc_is,
        "acc_ml": acc_ml,
        "acc_pa": acc_pa,
        "gyr_is": gyr_is,
        "gyr_ml": gyr_ml,
        "gyr_pa": gyr_pa
    })

    return imu_data

@pytest.fixture
def zeros_df():
    """Return a dummy dataframe with zeros."""
    df = pd.DataFrame(
        np.zeros((1000, 6)),
        columns=["acc_is", "acc_ml", "acc_pa", "gyr_is", "gyr_ml", "gyr_pa"]
    )
    # sanity check inside the fixture
    assert list(df.columns) == ["acc_is", "acc_ml", "acc_pa", "gyr_is", "gyr_ml", "gyr_pa"]
    return df

@pytest.fixture
def imu_df():
    """A synthetic IMU dataframe simulating wrist-worn gait pattern with idle padding."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    fs = 100  # sampling rate in Hz
    t = np.arange(n_samples) / fs

    freq_step = 1.0  # step frequency ~1 Hz
    acc_amp = 5.0
    gyr_amp = 10.0

    # Axis offsets
    acc_is_offset = 10.0
    acc_ml_offset = -1.0
    acc_pa_offset = -1.0

    # Acceleration components
    acc_is = acc_is_offset + acc_amp * np.sin(2 * np.pi * freq_step * t) + rng.normal(0, 0.1, n_samples)
    acc_ml = acc_ml_offset + acc_amp * np.sin(2 * np.pi * freq_step * t + np.pi/4) + rng.normal(0, 0.1, n_samples)
    acc_pa = acc_pa_offset + acc_amp * np.sin(2 * np.pi * freq_step * t + np.pi/2) + rng.normal(0, 0.1, n_samples)

    # Gyroscope components
    gyr_is = gyr_amp * np.sin(2 * np.pi * freq_step * t) + rng.normal(0, 0.5, n_samples)
    gyr_ml = gyr_amp * np.sin(2 * np.pi * freq_step * t + np.pi/4) + rng.normal(0, 0.5, n_samples)
    gyr_pa = gyr_amp * np.sin(2 * np.pi * freq_step * t + np.pi/2) + rng.normal(0, 0.5, n_samples)

    # Pad with 350 idle samples before and after
    pad = 350
    def pad_signal(sig, offset):
        return np.concatenate([
            offset + rng.normal(0, 0.05, pad),  # before
            sig,
            offset + rng.normal(0, 0.05, pad)   # after
        ])

    df = pd.DataFrame({
        "acc_is": pad_signal(acc_is, acc_is_offset),
        "acc_ml": pad_signal(acc_ml, acc_ml_offset),
        "acc_pa": pad_signal(acc_pa, acc_pa_offset),
        "gyr_is": pad_signal(gyr_is, 0),
        "gyr_ml": pad_signal(gyr_ml, 0),
        "gyr_pa": pad_signal(gyr_pa, 0)
    })

    return df
