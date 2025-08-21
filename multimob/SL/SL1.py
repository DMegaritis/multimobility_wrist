import warnings
import numpy as np
import pandas as pd
from typing import Literal
from typing_extensions import Self, Unpack
from mobgap.utils.conversions import as_samples
from mobgap.utils.interpolation import robust_step_para_to_sec
from mobgap.data_transform import (
    chain_transformers,
    ButterworthFilter
)

class WeinbergSL:
    """
    Implementation of the Weinberg [1] step length estimation algorithm for wrist-worn sensors.
    This release supports fine-tuned versions optimised for people with multimorbidity exhibiting diverse gait patterns;
    the original algorithm was developed for dead-reckoning applications.

    Steps:
    1. Preprocess the acceleration signal using the Euclidean norm of the three axes and convert to g-units.
    2. Apply a Butterworth lowpass filter (2 Hz cutoff) to reduce noise.
    3. Calculate the maximum and minimum acceleration values between consecutive initial contacts.
    4. Estimate step length using the Weinberg formula: Step length = WeinbergA * (|maxmin|^0.25) + WeinbergB.
    5. Interpolate step length to obtain per-second values.
    6. Calculate stride length by multiplying step length by 2.

    Parameters
    ----------
    version : str, optional, default="wrist"
        The version of the algorithm to use. Options:
        - "wrist": basic wrist-worn version
        - "wrist_adaptive": wrist version with RMS-based adaptive scaling
        - "wrist_foot": wrist version with foot-length offset
        - "wrist_adaptive_foot": adaptive wrist version with foot-length offset
    WeinbergB : float, optional
        Offset factor for step length calculation. Defaults to 0 for wrist-only versions
        and 26.5 cm (converted to meters) for foot-length augmented versions if not provided.

    Attributes
    ----------
    WeinbergA : float
        Scaling factor used in the step length calculation.
    WeinbergB : float
        Offset factor used in the step length calculation.
    raw_step_length_per_step_ : pd.DataFrame
        Raw step length calculated for each step.
    step_length_per_sec_ : pd.DataFrame
        Interpolated step length per second.
    stride_length_per_sec_ : pd.DataFrame
        Interpolated stride length per second.
    average_stride_length_ : float
        Mean stride length over the sequence.
    max_interpolation_gap_s : float
        Maximum allowed gap (in seconds) for interpolation.

    Notes
    -----
    - Suggested sampling rate is 100 Hz, but the algorithm is sample rate agnostic.
    - Provides better results with g-units.
    - Uses a Butterworth lowpass filter (2 Hz cutoff) for noise reduction.
    - Adaptive versions scale WeinbergA based on RMS of acceleration between steps.

    References
    ----------
    [1] Weinberg, H. (2002). Using the ADXL202 in pedometer and personal navigation applications.
        Analog Devices AN-602 application note, 2(2), 1-6.
    """

    max_interpolation_gap_s: float
    raw_step_length_per_step_: pd.DataFrame
    step_length_per_sec_: pd.DataFrame


    def __init__(
        self,
        *,
        version: Literal["wrist", "wrist_adaptive", "wrist_foot", "wrist_adaptive_foot"] = "wrist",
        WeinbergB: float = None,
    ) -> None:
        """
        Initializes the WeinbergSL algorithm with version-specific parameters.

        Parameters
        ----------
        version : str
            Specifies which version of the algorithm to use.
        WeinbergB : float, optional
            Step length offset in meters. Automatically converted from 26.5 cm for foot versions if not provided.
        """

        if version not in ["wrist", "wrist_adaptive", "wrist_foot", "wrist_adaptive_foot"]:
            raise ValueError(f"Invalid version: {version}")

        self.max_interpolation_gap_s = 3
        self.version = version

        if self.version == "wrist":
            self.WeinbergA = 0.62
            self.WeinbergB = 0
        elif self.version == "wrist_adaptive":
            self.WeinbergA = 0.60
            self.WeinbergB = 0
        elif self.version == "wrist_foot":
            self.WeinbergA = 0.21
            self.WeinbergB = (WeinbergB if WeinbergB is not None else 26.5) * 0.01 # Turning B into meters as the input is in cm
        elif self.version == "wrist_adaptive_foot":
            self.WeinbergA = 0.20
            self.WeinbergB = (WeinbergB if WeinbergB is not None else 26.5) * 0.01 # Turning B into meters as the input is in cm

    def calculate(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.DataFrame,
        *,
        sampling_rate_hz: float
    ) -> Self:
        """
        Estimates step and stride lengths between two consecutive initial contact events.

        Parameters
        ----------
        data : pd.DataFrame
            Acceleration data (3-axis) from the wrist sensor.
        initial_contacts : pd.DataFrame
            DataFrame of detected initial contact indices.
        sampling_rate_hz : float
            Sampling frequency of the input signal.

        Returns
        -------
        Self
            Returns the instance with computed step and stride lengths in attributes.

        Warnings
        --------
        - If initial_contacts are not sorted, they will be sorted automatically.
        - If fewer than two initial contacts are provided, all outputs are NaN.
        """

        self.data = data
        self.initial_contacts = initial_contacts
        self.sampling_rate_hz = sampling_rate_hz

        # Check if ICs are not sorted and sort if necessary
        if not self.initial_contacts["ic"].is_monotonic_increasing:
            warnings.warn("Initial contacts were not in ascending order. Rearranging them.", stacklevel=2)
            self.initial_contacts = self.initial_contacts.sort_values("ic").reset_index(drop=True)

        self.ic_list = self.initial_contacts["ic"].to_numpy()

        if len(self.ic_list) > 0 and (self.ic_list[0] != 0 or self.ic_list[-1] != len(self.data) - 1):
            warnings.warn(
                "Usually we assume that gait sequences are cut to the first and last detected initial "
                "contact. "
                "This is not the case for the passed initial contacts and might lead to unexpected "
                "results in the cadence calculation. "
                "Specifically, you will get NaN values at the start and the end of the output.",
                stacklevel=1,
            )

        # Remove duplicate values from ic_list
        self.ic_list = np.unique(self.ic_list)

        # Calculating the duration and the center of each second
        duration = self.data.shape[0] / self.sampling_rate_hz
        sec_centers = np.arange(0, duration) + 0.5

        # Checking if ic_list is empty or includes only 1 IC, we can not calculate step length with only one or zero initial contact
        if len(self.ic_list) <= 1:
            warnings.warn("Can not calculate step length with only one or zero initial contacts.", stacklevel=1)
            self._set_all_nan(sec_centers, self.ic_list)
            return self

        # Using the acceleration norm for the wrist version
        vacc = np.linalg.norm(self.data.iloc[:, 0:3], axis=1)

        # turning m/s^2 to g since the multimob perform better
        vacc = vacc / 9.81

        # Calling the function to calculate step length
        raw_step_length = self._calc_step_length_weinberg(vacc, self.ic_list)

        # The last step length is repeated to match the number of step lengths with the initial contacts for interpolation
        raw_step_length_padded = np.append(raw_step_length, raw_step_length[-1])

        # We interpolate the step length in per second values
        # The below function uses a default Hampel filter applied 1) in the passed step lengths and 2) the per second
        # interpolated values.
        initial_contacts_per_sec = self.ic_list / self.sampling_rate_hz
        step_length_per_sec = robust_step_para_to_sec(
            initial_contacts_per_sec,
            raw_step_length_padded,
            sec_centers,
            self.max_interpolation_gap_s
        )
        # Stride length is derived by multiplying the step length by 2
        stride_length_per_sec = step_length_per_sec * 2

        self._unify_and_set_outputs(raw_step_length, step_length_per_sec, stride_length_per_sec, sec_centers)
        return self


    def _calc_step_length_weinberg(
        self,
        vacc: np.ndarray,
        initial_contacts: np.ndarray,
    ) -> np.ndarray:
        """
        Estimates step length between initial contacts using the Weinberg formula.

        Parameters
        ----------
        vacc : np.ndarray
            Norm of acceleration vector (g-units).
        initial_contacts : np.ndarray
            Indices of detected initial contacts.

        Returns
        -------
        np.ndarray
            Raw step length estimates for each detected step.
        """

        # 1. Preprocessing with Butterworth filt
        cutoff = 2
        filter_chain = [("butter", ButterworthFilter(order=4, cutoff_freq_hz=cutoff, filter_type='lowpass'))]
        vacc_butter = np.asarray(
            chain_transformers(vacc, filter_chain, sampling_rate_hz=self.sampling_rate_hz))

        # 2. Calculating maxmin of acceleration between each initial contact
        maxmin = np.array([np.max(vacc_butter[ic:ic_next]) - np.min(vacc_butter[ic:ic_next]) for ic, ic_next in zip(initial_contacts, initial_contacts[1:])])

        # If version is adaptive calculate RMS between each initial contact
        if self.version in ["wrist_adaptive", "wrist_adaptive_foot"]:
            rms_values = np.array([np.sqrt(np.mean(np.square(vacc_butter[ic:ic_next])))
                                for ic, ic_next in zip(initial_contacts, initial_contacts[1:])])

            # calculating the mean rms
            mean_rms = np.mean(rms_values)

            # Amending WeinbergA if mean_rms is not 0
            if mean_rms == 0:
                warnings.warn(
                    "The calculated RMS is 0. Step length calculation will proceed without scaling WeinbergA.",
                    stacklevel=2)
            else:
                self.WeinbergA = self.WeinbergA * mean_rms

        # 3. Calculating step length
        step_length = self.WeinbergA * (np.abs(maxmin) ** 0.25) + self.WeinbergB

        return step_length


    def _set_all_nan(self, sec_centers: np.ndarray, ic_list: np.ndarray) -> None:
        """
        Sets all step and stride length outputs to NaN if calculation cannot proceed.

        Parameters
        ----------
        sec_centers : np.ndarray
            Time centers for each second of the sequence.
        ic_list : np.ndarray
            Array of initial contact indices.

        Returns
        -------
        None
        """

        stride_length_per_sec = np.full(len(sec_centers), np.nan)
        raw_step_length = np.full(np.clip(len(ic_list) - 1, 0, None), np.nan)
        step_length_per_sec = np.full(len(sec_centers), np.nan)
        self._unify_and_set_outputs(raw_step_length, step_length_per_sec, stride_length_per_sec, sec_centers)

    def _unify_and_set_outputs(
            self,
            raw_step_length: np.ndarray,
            step_length_per_sec: np.ndarray,
            stride_length_per_sec: np.ndarray,
            sec_centers: np.ndarray,
    ) -> None:
        """
        Assigns calculated step and stride lengths to class attributes.

        Parameters
        ----------
        raw_step_length : np.ndarray
            Raw step length per detected step.
        step_length_per_sec : np.ndarray
            Interpolated step length per second.
        stride_length_per_sec : np.ndarray
            Interpolated stride length per second.
        sec_centers : np.ndarray
            Time centers of each second for indexing.

        Returns
        -------
        None
        """

        # Convert ic_list to a pandas DataFrame temporarily for assignment
        self.raw_step_length_per_step_ = pd.DataFrame({
            "ic": self.ic_list[:-1],  # Excluding the last element
            "step_length_m": raw_step_length
        })
        index = pd.Index(as_samples(sec_centers, self.sampling_rate_hz), name="sec_center_samples")
        self.step_length_per_sec_ = pd.DataFrame({"step_length_m": step_length_per_sec}, index=index)
        self.stride_length_per_sec_ = pd.DataFrame({"stride_length_m": stride_length_per_sec}, index=index)

        # Calculate the average stride length
        self.average_stride_length_ = self.stride_length_per_sec_["stride_length_m"].mean()