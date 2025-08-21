import pandas as pd
import numpy as np
from typing_extensions import Self
from scipy import signal
from mobgap.data_transform import (
    chain_transformers,
    ButterworthFilter
)
from typing import Literal
from multimob.ICD.utils.find_maxima import _find_maxima
from mobgap.initial_contacts._utils import find_zero_crossings

class ZijlstraIC:
    """
    Implementation of the Zijlstra initial contact (IC) detection algorithm [1],
    fine-tuned for wrist-worn accelerometer data. This is a signal feature detection
    approach based on filtered acceleration waveforms.

        This method processes the acceleration signal by:
    1. Computing the vector norm of the tri-axial accelerometer.
    2. Detrending the signal to center it around zero.
    3. Applying a two-stage low-pass Butterworth filter (20 Hz then cutoff at 2.5 Hz).
    4. Identifying initial contacts using either:
       - Peak detection between zero crossings ("peak" mode, default), or
       - Positive-to-negative zero crossings ("zc" mode).

    References
    ----------
    [1] Zijlstra W, Hof AL. Assessment of spatio-temporal gait parameters from trunk accelerations
        during human walking. Gait Posture. 2003 Oct;18(2):1-10.
        doi: 10.1016/s0966-6362(02)00190-x. PMID: 14654202.

    Attributes
    ----------
    ic_list_ : pd.DataFrame
        DataFrame containing the detected initial contact (IC) indices.
    final_signal_ : np.ndarray
        The processed signal after detrending and low-pass Butterworth filtering.

    Notes
    -----
    - The Butterworth cutoff frequency is set to 2.5 Hz by default, chosen for typical walking
      cadence in multimorbid cohorts. Adjust as needed for populations with different gait speeds.
    - Designed for accelerometer data sampled at ~100 Hz.
    - Algorithm adapted and fine-tuned for wrist-worn accelerometer signals, where the norm of
      the 3D acceleration vector is used.
    """

    ic_list_: pd.DataFrame

    def __init__(self, *, version: Literal["wrist"]="wrist") -> None:
        """
        Initialise the ZijlstraIC detector for wrist-worn accelerometer data.

        Parameters
        ----------
        version : Literal["wrist"], optional
            Algorithm version. Currently, only the "wrist" variant is implemented.
        """
        self.version = version


    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float) -> Self:
        """
        Detect initial contact events in wrist-worn accelerometer data using
        the Zijlstra algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            Input accelerometer data. The first three columns should contain the x, y, z axes.
        sampling_rate_hz : float
            Original sampling rate of the input signal in Hz.

        Returns
        -------
        Self
            Returns the instance with detected initial contacts stored in `ic_list_`
            and the processed signal in `final_signal_`.
        """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.cutoff = 2.5
        self.method = "peak"
        acc = np.linalg.norm(self.data.iloc[:, 0:3], axis=1)

        # Detrend data to make the signal is around 0
        detrended_data = signal.detrend(acc)

        # Low pass Butterworth as filter chain (the first filter has a fixed cutoff of 20 Hz)
        filter_chain = [("butter_1", ButterworthFilter(order=4, cutoff_freq_hz=20, filter_type='lowpass')),
                        ("butter_2", ButterworthFilter(order=4, cutoff_freq_hz=self.cutoff, filter_type='lowpass'))]

        acc_pa_butter = np.asarray(
            chain_transformers(detrended_data, filter_chain, sampling_rate_hz=self.sampling_rate_hz))

        self.final_signal_ = acc_pa_butter

        if self.method == "peak":
            # Initial contacts by finding the maxima between zero crossings
            ic_indices = _find_maxima(acc_pa_butter)
        elif self.method == "zc":
            # Initial contacts by finding zero crossings (positive to negative)
            ic_indices = find_zero_crossings(acc_pa_butter, "positive_to_negative", refine=False)

        # Creating a dataframe with the IC indices
        final_ics = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
        final_ics["ic"] = ic_indices

        self.ic_list_ = final_ics

        return self