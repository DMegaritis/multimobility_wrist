import numpy as np
import matplotlib.pyplot as plt

def auto_cov_unbiased(x: np.ndarray) -> np.ndarray:
    """
    Compute the unbiased autocovariance of a signal

    Parameters
    ----------
    x : np.ndarray
        The signal

    Returns
    -------
    np.ndarray
        The unbiased autocovariance
    """
    n = len(x)
    x_centred = x - np.mean(x)
    ac = np.correlate(x_centred, x_centred, mode='full')

    # Create lags from -n+1 to n-1
    lags = np.arange(-n + 1, n)

    # Calculate unbiased autocovariance
    unbiased_ac = ac / np.array([n - abs(lag) for lag in lags])

    return unbiased_ac