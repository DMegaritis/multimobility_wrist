import numpy as np

"""This module contains utility functions for stride length algos"""

def moving_average_filter_bylemans(
        data: np.ndarray,
        sampling_rate_hz: float = 100,
        window_duration_ms: float = 125) -> np.ndarray:
    # Setting the window size to 125 milliseconds
    window_size = int((window_duration_ms / 1000) * sampling_rate_hz)

    # Ensure the window size is at least 1
    if window_size < 1:
        raise ValueError("Window size must be at least 1 sample.")

    # Create a uniform kernel for the moving average
    kernel = np.ones(window_size) / window_size

    # Apply the moving average filter using convolution
    smoothed_signal = np.convolve(data, kernel, mode='same')

    return smoothed_signal
