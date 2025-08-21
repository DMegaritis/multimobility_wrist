import pandas as pd
import numpy as np
from typing_extensions import Self
from typing import Literal
from scipy import signal, integrate
from scipy.signal import find_peaks
from mobgap.data_transform import (
    Resample,
    chain_transformers,
    ButterworthFilter,
    CwtFilter
)
from multimob.ICD.utils.dominant_frequency import dominant_freqency


class McCamleyIC:
    """
    Detect initial contact (IC) using the McCamley algorithm adapted for use with wrist-worn accelerometer data.

    This algorithm enhances peaks of the norm of the 3D acceleration vector and uses a combination of
    resampling, filtering, integration, and peak detection to estimate initial contact events.

    References
    ----------
    [1] McCamley, J., Donati, M., Grimpampi, E., & Mazzà, C. (2012). An enhanced estimate of initial contact and
        final contact instants of time using lower trunk inertial sensor data. Gait & Posture, 36(2), 316-318.
        https://doi.org/10.1016/j.gaitpost.2012.02.019

    [2] Del Din, S., Godfrey, A., & Rochester, L. (2016). Validation of an accelerometer to quantify a comprehensive
        battery of gait characteristics in healthy older adults and Parkinson's disease: Toward clinical and at home use.
        IEEE Journal of Biomedical and Health Informatics, 20(3), 838-847. https://doi.org/10.1109/JBHI.2015.2419317

    Attributes
    ----------
    _DOWNSAMPLED_RATE : int
        The rate (Hz) to which the signal is downsampled for processing.
    ic_list_ : pd.DataFrame
        Detected initial contact indices (resampled to original rate) with step IDs as the index.
    final_signal_ : np.ndarray
        The processed signal after filtering and integration, used for IC detection.

    Notes
    -----
    - Currently the IC distance thresholds are checked dynamically, which should provide the best results.
       If two ICs are closer than 0.25s, it is possible to keep the one with the highest acceleration magnitude.
    - The algorithm is sample-rate agnostic since signals are internally resampled to 50 Hz.
    """

    _DOWNSAMPLED_RATE = 50
    ic_list_: pd.DataFrame


    def __init__(self, *, version: Literal["wrist"] = "wrist") -> None:
        """
       Initialize the class.

       Parameters
       ----------
       version : str, optional
           The version of the algorithm to use. In this release we support only "wrist".
       """

        self.version = version
        self.cwt = "adaptive"

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float) -> Self:
        """
        Detect initial contact (IC) events from wrist accelerometer data.

        Parameters
        ----------
        data : pd.DataFrame
            Input accelerometer data. Must contain 3 columns representing the x, y, z axes.
        sampling_rate_hz : float
            Sampling rate of the input data in Hz.

        Returns
        -------
        Self
            Returns self with the detected IC indices stored in `ic_list_`.
        """


        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # we use the norm of the acceleration vector for the wrist version
        acc = np.linalg.norm(data.iloc[:, 0:3].to_numpy(), axis=1)

        # Resample the signal to 50 Hz using mobgap filters
        filter_chain = [("resampling", Resample(self._DOWNSAMPLED_RATE))]
        acc_downsampled = chain_transformers(acc, filter_chain, sampling_rate_hz=self.sampling_rate_hz)

        # Detrend data
        detrended_data = signal.detrend(acc_downsampled)

        # Low pass Butterworth
        cutoff = 20
        filter_chain = [("butter", ButterworthFilter(order=4, cutoff_freq_hz=cutoff, filter_type='lowpass'))]
        acc_butter = np.asarray(chain_transformers(detrended_data, filter_chain, sampling_rate_hz=self.sampling_rate_hz))

        # Cumulative trapezoidal integration
        integrated_data = integrate.cumulative_trapezoid(acc_butter, initial=0)

        # CWT filter and upsampling to original frequency before minima detection

        # No information in the papers about the centre_frequency, using 1 gives the best results
        # However, there is the option to use the dominant frequency of the signal as an adaptive approach (usually is 1).
        if self.cwt == "fixed":
            freq = 1
        elif self.cwt == "adaptive":
            freq = dominant_freqency(integrated_data, sampling_rate_hz=self._DOWNSAMPLED_RATE)
            # if freq is 0 it is turned to 1.0, if it is above 5 it is turned to 5.0. A frequency higher than 5 is not adequate to smooth the signal
            if freq == 0:
                freq = 1.0
            elif freq > 5:
                freq = 5.0

        filter_chain = [("cwt", CwtFilter(wavelet='gaus1', center_frequency_hz=freq)),
                        ("resampling", Resample(self.sampling_rate_hz))]
        data_cwt_upsampled = np.asarray(chain_transformers(integrated_data, filter_chain, sampling_rate_hz=self._DOWNSAMPLED_RATE))

        self.final_signal_ = data_cwt_upsampled

        # Initial contact peak detection (max of the reverse of the signal)
        inverted = -data_cwt_upsampled
        ic_indices, _ = find_peaks(inverted)

        # If no ICs detected then we shortcut by returning an empty df
        if ic_indices.size == 0:
            self.ic_list_ = pd.DataFrame(columns=['ic'])
            self.ic_list_.index.name = 'step_id'
            return self

        # # This is a second cwt for toe off detection it is commented because it is not a primary aim
        # center_frequency_hz = 6.0
        # filter_chain = [("cwt", CwtFilter(wavelet='gaus1', center_frequency_hz=center_frequency_hz))]
        # data_cwt2 = np.asarray(chain_transformers(inverted, filter_chain, sampling_rate_hz=self.sampling_rate_hz))
        # coefs2 = data_cwt2[0]
        # fc_indices, properties = find_peaks(coefs2)

        # Removing ICs which are closer to 0.25s from the previous IC, we use a dynamic approach.
        # this method dynamically calculates the distance between the last valid IC and the current IC.
        # If we calculated the distance once, then we may remove ICs which are valid because we would not compare the distance with the last valid IC.
        filtered_ics_close = [ic_indices[0]]

        for i in range(1, len(ic_indices)):
            if ic_indices[i] - filtered_ics_close[-1] > 0.25 * self.sampling_rate_hz:
                filtered_ics_close.append(ic_indices[i])  # keeping only if sufficiently spaced

        # Removing ICs with a distance > 2.25s. We first compare n with n-1 and if further than 2.25s we compare n with n+1.
        # We remove n if both are further than 2.25s.
        filtered_ics_away = [filtered_ics_close[0]]

        for i in range(1, len(filtered_ics_close) - 1):  # Keeping first
            prev = filtered_ics_away[-1]
            curr = filtered_ics_close[i]
            next_val = filtered_ics_close[i + 1]

            if (curr - prev) > 2.25 * self.sampling_rate_hz:
                if (next_val - curr) > 2.25 * self.sampling_rate_hz:
                    continue  # not adding the current value
            filtered_ics_away.append(curr)

        # Checking last value with distance to the previous one
        if (filtered_ics_close[-1] - filtered_ics_away[-1]) <= 2.25 * self.sampling_rate_hz:
            filtered_ics_away.append(filtered_ics_close[-1])

        final_ics = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
        final_ics["ic"] = filtered_ics_away

        self.ic_list_ = final_ics

        return self