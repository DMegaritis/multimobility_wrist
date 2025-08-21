import pandas as pd
import numpy as np
from typing import Literal
from typing_extensions import Self
from scipy import signal, integrate
from scipy.signal import find_peaks
from mobgap.data_transform import (
    chain_transformers,
    ButterworthFilter,
    CwtFilter,
    Resample
)
from multimob.ICD.utils.dominant_frequency import dominant_freqency


class PhamIC:
    """
    This implementation of the Pham initial contact (IC) detection algorithm [1] is designed and fine tuned for wrist-worn accelerometer data.
    It enhances peaks of the acceleration vector by smoothing the signal using resampling, detrending, filtering, and integration.

    References
    ----------
    [1] Pham MH, Elshehabi M, Haertner L, Del Din S, Srulijes K, Heger T, Synofzik M, Hobert MA, Faber GS, Hansen C,
        Salkovic D, Ferreira JJ, Berg D, Sanchez-Ferro Á, van Dieën JH, Becker C, Rochester L, Schmidt G, Maetzler W.
        Validation of a Step Detection Algorithm during Straight Walking and Turning in Patients with Parkinson's Disease
        and Older Adults Using an Inertial Measurement Unit at the Lower Back. Front Neurol. 2017 Sep 4;8:457.
        doi: 10.3389/fneur.2017.00457. PMID: 28928711; PMCID: PMC5591331.

    Attributes
    ----------
    _UPSAMPLED_RATE : int
        The target rate (Hz) to which the signal is upsampled for filtering (128 Hz).
    ic_list_ : pd.DataFrame
        DataFrame storing the detected initial contact indices resampled back to the original signal rate.
    final_signal_ : np.ndarray
        The processed signal after resampling, detrending, filtering, and integration.

    Notes
    -----
    - The continuous wavelet transform (CWT) center frequency is chosen adaptively based on the dominant frequency of the signal,
      constrained to 1–5 Hz for smoothing purposes.
    - The algorithm is sample-rate agnostic because all signals are internally upsampled to 128 Hz, processed, and then resampled
      back to the original sampling rate.
    - Only peaks exceeding a percentage threshold of the mean peak magnitude are considered valid ICs.
    """

    _UPSAMPLED_RATE = 128
    ic_list_: pd.DataFrame


    def __init__(self, *, version: Literal["wrist"] = "wrist") -> None:
        """
        Initialize the PhamIC detector for wrist-worn accelerometer data.

        Parameters
        ----------
        version : Literal["wrist"], optional
            The version of the algorithm to use. Default is "wrist".
            Currently, only the wrist version is implemented.
        """
        self.version = version


    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float) -> Self:
        """
        Detect initial contact (IC) events in wrist accelerometer data using the Pham algorithm.

        This method processes the acceleration signal by:
        1. Computing the vector norm of the tri-axial accelerometer.
        2. Upsampling to 128 Hz.
        3. Detrending and low-pass Butterworth filtering.
        4. Cumulative trapezoidal integration.
        5. Continuous wavelet transform (CWT) filtering using either fixed or adaptive frequency.
        6. Downsampling back to the original signal rate.
        7. Peak detection with a threshold to identify initial contacts.

        Parameters
        ----------
        data : pd.DataFrame
            Input accelerometer data. The first three columns should contain x, y, z acceleration axes.
        sampling_rate_hz : float
            Original sampling rate of the input signal in Hz.

        Returns
        -------
        Self
            Returns the instance with detected initial contacts stored in `ic_list_` and the processed signal in `final_signal_`.
        """

        self.percentage_thresh = 0.02
        self.cwt = "adaptive"

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # we use the norm of the acceleration vector for the wrist version
        acc = np.linalg.norm(data.iloc[:, 0:3].to_numpy(), axis=1)

        # Upsample data to the original sampling rate of the paper
        filter_chain = [("resample", Resample(target_sampling_rate_hz=self._UPSAMPLED_RATE))]
        acc_upsamp = np.asarray(chain_transformers(acc, filter_chain, sampling_rate_hz=self.sampling_rate_hz))

        # Detrend data
        detrended_data = signal.detrend(acc_upsamp)

        # Low pass Butterworth
        cutoff = 10
        filter_chain = [("butter", ButterworthFilter(order=2, cutoff_freq_hz=cutoff, filter_type='lowpass'))]
        acc_pa_butter = np.asarray(chain_transformers(detrended_data, filter_chain, sampling_rate_hz=self.sampling_rate_hz))

        # Cumulative trapezoidal integration
        integrated_data = integrate.cumulative_trapezoid(acc_pa_butter, initial=0)

        # CWT filter
        # No information in the papers about the centre_frequency, using 1 gives the best results
        # However, there is the option to use the dominant frequency of the signal as an adaptive approach (usually is 1).
        if self.cwt == "fixed":
            freq = 1
        elif self.cwt == "adaptive":
            freq = dominant_freqency(integrated_data, sampling_rate_hz=self._UPSAMPLED_RATE)
            # if freq is 0 it is turned to 1.0, if it is above 5 it is turned to 5.0. A frequency higher than 5 is not adequate to smooth the signal
            if freq == 0:
                freq = 1.0
            elif freq > 5:
                freq = 5.0

        filter_chain = [("cwt", CwtFilter(wavelet='gaus1', center_frequency_hz=freq))]
        data_cwt = np.asarray(chain_transformers(integrated_data, filter_chain, sampling_rate_hz=self.sampling_rate_hz))

        # Detrend data
        detrended_data = signal.detrend(data_cwt)

        # Downsample data to the original sampling rate
        filter_chain = [("resample", Resample(target_sampling_rate_hz=self.sampling_rate_hz))]
        detrended_data = np.asarray(chain_transformers(detrended_data, filter_chain, sampling_rate_hz=self._UPSAMPLED_RATE))

        self.final_signal_ = detrended_data

        # Initial contact peak detection
        inverted = -detrended_data
        ic_indices, _ = find_peaks(inverted)

        # If no peeaks deteted then we shortcut by returning an empty df
        if ic_indices.size == 0:
            self.ic_list_ = pd.DataFrame(columns=['ic'])
            self.ic_list_.index.name = 'step_id'
            return self

        # Keeping only ICs above the threshold which is calculated as a percentage of the magnitude of the peaks
        # calculating the mean of peaks magnitudes
        mean_peak = np.mean(inverted[ic_indices])
        # threshold
        thresh = mean_peak * self.percentage_thresh
        # removing ICs below the threshold
        if ic_indices.size > 0:
            ic_indices = ic_indices[inverted[ic_indices] > thresh]

        final_ics = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
        final_ics["ic"] = ic_indices

        self.ic_list_ = final_ics

        return self