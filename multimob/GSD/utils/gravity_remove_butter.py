import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from mobgap.data_transform import (
    Resample,
    chain_transformers,
    ButterworthFilter
)

def gravity_motion_butterworth(data: pd.DataFrame, sampling_rate_hz: float):
    """
    Separate linear acceleration from gravity using a Butterworth low-pass filter.

    Parameters:
    raw_data (dict): Dictionary containing raw accelerometer data with keys 'Acc_X', 'Acc_Y', 'Acc_Z'.
    fs (float): Sampling frequency in Hz (default is 100 Hz).
    cutoff (float): Cutoff frequency for the low-pass filter (default is 0.25 Hz).
    order (int): Order of the Butterworth filter (default is 1 as per the paper).

    Returns:
    dict: Dictionary containing gravity and motion components for each axis (X, Y, Z) and total motion.
    """

    # Ensuring required keys exist in data
    required_keys = ['acc_is', 'acc_ml', 'acc_pa']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: '{key}' in input data")

    acc_is = data['acc_is'].values
    acc_ml = data['acc_ml'].values
    acc_pa = data['acc_pa'].values

    # Performing a low pass butterworth filter on the data
    cutoff = 0.25
    # class instance
    filter_chain = [("butter", ButterworthFilter(order=1, cutoff_freq_hz=cutoff, filter_type='lowpass'))]

    # application to all corrected axes
    acc_is_filt = np.asarray(chain_transformers(acc_is, filter_chain, sampling_rate_hz=sampling_rate_hz))
    acc_ml_filt = np.asarray(chain_transformers(acc_ml, filter_chain, sampling_rate_hz=sampling_rate_hz))
    acc_pa_filt = np.asarray(chain_transformers(acc_pa, filter_chain, sampling_rate_hz=sampling_rate_hz))


    # Compute motion components by subtracting gravity
    acc_is_no_grav = data['acc_is'] - acc_is_filt
    acc_ml_no_grav = data['acc_ml'] - acc_ml_filt
    acc_pa_no_grav = data['acc_pa'] - acc_pa_filt

    # concatinate the data
    acc_no_grav = pd.concat([acc_is_no_grav, acc_ml_no_grav, acc_pa_no_grav], axis=1)

    return acc_no_grav