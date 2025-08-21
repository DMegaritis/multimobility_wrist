import pandas as pd
import numpy as np


def dominant_freqency(data, sampling_rate_hz: float) -> int:
    # If data are numpy arrays, convert them to pandas Series
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # centering the signal around 0
    data = data - np.mean(data)

    # Calculating the FFT of the signal
    n = len(data)
    fft_signal = np.fft.fft(data)
    frequencies = np.fft.fftfreq(n, d=1 / sampling_rate_hz)

    # Only keep the positive frequencies (real part)
    positive_frequencies = frequencies[:n // 2]
    positive_fft_signal = np.abs(fft_signal[:n // 2])

    # Find the dominant frequency
    dominant_frequency = positive_frequencies[np.argmax(positive_fft_signal)]

    return int(dominant_frequency)
