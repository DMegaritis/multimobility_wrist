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
from multimob.SL.utils.SL_utils import (moving_average_filter_bylemans)


class BylemansSL:
    """
    Implementation of the Bylemans step length estimation algorithm [1].

    This class estimates step and stride lengths from acceleration signals using the intensity-based
    model proposed by Bylemans et al. (2009). The algorithm supports multiple sensor placements
    (wrist, foot) and adaptive scaling options.

    Steps:
    1. Preprocess the acceleration signal: vertical axis for low back, Euclidean norm for wrist.
    2. Apply a 4 Hz high-pass Butterworth filter to remove low-frequency noise.
    3. Smooth the signal using a moving average filter.
    4. Compute maximum and minimum accelerations between consecutive initial contacts.
    5. Compute mean acceleration between consecutive initial contacts.
    6. Compute step durations in seconds.
    7. Estimate step length using:
       `Step length = BylemansA * (|mean| * sqrt(1 / sqrt(dtime * |maxmin|)))^(1/2.7) + BylemansB`
    8. Interpolate step lengths to per-second values.
    9. Calculate stride length as twice the step length.

    Parameters
    ----------
    version : str, optional, default="lowback"
        Algorithm variant to use. Options include "lowback", "lowback_adaptive", "wrist",
        "wrist_adaptive", and foot-augmented versions.
    max_interpolation_gap_s : float, optional, default=3
        Maximum gap (in seconds) allowed for interpolation.
    BylemansB : float, optional
        Offset factor for step length. Defaults to 0 unless using foot-augmented versions (26.5 cm).

    Attributes
    ----------
    BylemansA : float
        Scaling factor for step length calculation.
    BylemansB : float
        Offset factor for step length calculation.
    raw_step_length_per_step_ : pd.DataFrame
        Step length calculated for each step.
    step_length_per_sec_ : pd.DataFrame
        Interpolated step length per second.
    stride_length_per_sec_ : pd.DataFrame
        Interpolated stride length per second.
    average_stride_length_ : float
        Mean stride length over the sequence.
    max_interpolation_gap_s : float
        Maximum interpolation gap in seconds.

    Notes
    -----
    - Best performance is achieved with 100 Hz sampling rate, though the algorithm is sample rate agnostic.
    - Uses acceleration in m/s² rather than g-units.

    References
    ----------
    [1] Bylemans, I., Weyn, M., & Klepal, M. (2009). Mobile phone-based displacement estimation for
        opportunistic localisation systems. In 2009 Third International Conference on Mobile Ubiquitous
        Computing, Systems, Services and Technologies (pp. 113-118). IEEE.
    """



    max_interpolation_gap_s: float
    raw_step_length_per_step_: pd.DataFrame
    step_length_per_sec_: pd.DataFrame

    def __init__(
        self,
        *,
        version: Literal["wrist", "wrist_adaptive", "wrist_foot", "wrist_adaptive_foot"] = "wrist",
        BylemansB: float = None
    ) -> None:
        """
        Initialise BylemansSL with the desired version and parameters.

        Parameters
        ----------
        version : str, optional
            Algorithm variant. Defaults to "wrist".
        BylemansB : float, optional
            Step length offset. If None, defaults to 0 or 26.5 cm for foot-augmented variants.
        """

        if version not in ["wrist", "wrist_adaptive", "wrist_foot", "wrist_adaptive_foot"]:
            raise ValueError(f"Invalid version: {version}")

        self.version = version
        self.max_interpolation_gap_s = 3

        if self.version == "wrist":
            self.BylemansA = 2.3
            self.BylemansB = 0
        elif self.version == "wrist_adaptive":
            self.BylemansA = 9.15
            self.BylemansB = 0
        elif self.version == "wrist_foot":
            self.BylemansA = 0.75
            self.BylemansB = (BylemansB if BylemansB is not None else 26.5) * 0.01  # Turning B into meters as the input is in cm
        elif self.version == "wrist_adaptive_foot":
            self.BylemansA = 3.46
            self.BylemansB = (BylemansB if BylemansB is not None else 26.5) * 0.01  # Turning B into meters as the input is in cm


    def calculate(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.DataFrame,
        *,
        sampling_rate_hz: float
    ) -> Self:
        """
        Calculate step and stride lengths from acceleration data.

        Parameters
        ----------
        data : pd.DataFrame
            Input acceleration data.
        initial_contacts : pd.DataFrame
            Detected initial contact events (gait events).
        sampling_rate_hz : float
            Sampling rate of the input signal in Hz.

        Returns
        -------
        Self
            Instance with step and stride lengths stored in attributes.
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

        # Checking if ic_list is empty or includes only 1 IC
        if len(self.ic_list) <= 1:
            # We can not calculate step length with only one or zero initial contact
            warnings.warn("Can not calculate step length with only one or zero initial contacts.", stacklevel=1)
            self._set_all_nan(sec_centers, self.ic_list)
            return self

        # Using the acceleration norm for the wrist version
        vacc = np.linalg.norm(self.data.iloc[:, 0:3], axis=1)

        # Calling the function to calculate step length
        raw_step_length = self._calc_step_length_bylemans(vacc, self.ic_list)

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


    def _calc_step_length_bylemans(
        self,
        vacc: np.ndarray,
        initial_contacts: np.ndarray,
    ) -> np.ndarray:
        """
        Compute step length using Bylemans algorithm.

        Parameters
        ----------
        vacc : np.ndarray
            Preprocessed acceleration signal (Euclidean norm for wrist).
        initial_contacts : np.ndarray
            Indices of initial contact events.

        Returns
        -------
        np.ndarray
            Step lengths corresponding to each step.
        """

        # 1. Preprocessing with Butterworth highpass filt
        cutoff = 4
        filter_chain = [("butter", ButterworthFilter(order=4, cutoff_freq_hz=cutoff, filter_type='highpass'))]
        vacc_butter = np.asarray(
            chain_transformers(vacc, filter_chain, sampling_rate_hz=self.sampling_rate_hz))

        # 2. Preprocessing with moving average
        vacc_butter_mav = moving_average_filter_bylemans(vacc_butter, self.sampling_rate_hz)

        # 3. Calculating maxmin of acceleration between each initial contact
        maxmin = np.array([np.max(vacc_butter_mav[ic:ic_next]) - np.min(vacc_butter_mav[ic:ic_next]) for ic, ic_next in zip(initial_contacts, initial_contacts[1:])])

        # 4. Calculating mean of acceleration between each initial contact
        mean = np.array([np.mean(vacc_butter_mav[ic:ic_next]) for ic, ic_next in zip(initial_contacts, initial_contacts[1:])])

        # 5. Calculating time in ms between steps
        dtime = np.array([np.abs(ic - ic_next) for ic, ic_next in zip(initial_contacts, initial_contacts[1:])])
        # time to seconds
        dtime = dtime / self.sampling_rate_hz

        # 6. If version is adaptive calculate RMS between each initial contact
        if self.version in ["wrist_adaptive", "wrist_adaptive_foot"]:
            rms_values = np.array([np.sqrt(np.mean(np.square(vacc_butter_mav[ic:ic_next]))) for ic, ic_next in zip(initial_contacts, initial_contacts[1:])])

            # calculating the mean rms
            mean_rms = np.mean(rms_values)

            # Amending BylemansA if mean_rms is not 0
            if mean_rms == 0:
                warnings.warn("The calculated RMS is 0. Step length calculation will proceed without scaling BylemansA.", stacklevel=2)
            else:
                self.BylemansA = self.BylemansA * mean_rms

        # 7. Calculating step length. Using absolute value of mean and maxmin to avoid negative values
        product = dtime * np.abs(maxmin)
        denominator = np.sqrt(product)
        # Create step_length safely with NaNs where denominator is zero
        step_length = np.full_like(denominator, np.nan, dtype=np.float64)
        # Identify valid indices where denominator != 0
        valid_idx = denominator != 0
        # Compute step length only for valid indices
        step_length[valid_idx] = self.BylemansA * (np.abs(mean[valid_idx]) * (1 / denominator[valid_idx]) ** 0.5) ** (
                    1 / 2.7) + self.BylemansB

        # Warn if zero denominators detected
        if not np.all(valid_idx):
            warnings.warn(
                f"Zero denominator detected in step length calculation at indices {np.where(~valid_idx)[0]}. Setting step length to NaN for these steps.",
                stacklevel=2)

        return step_length

    def _set_all_nan(self, sec_centers: np.ndarray, ic_list: np.ndarray) -> None:
        """
        Set all outputs to NaN when step length cannot be computed.

        Parameters
        ----------
        sec_centers : np.ndarray
            Centers of each second for interpolation.
        ic_list : np.ndarray
            Array of initial contact indices.
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
        Store step and stride length outputs in class attributes.

        Parameters
        ----------
        raw_step_length : np.ndarray
            Step length per initial contact.
        step_length_per_sec : np.ndarray
            Step length interpolated per second.
        stride_length_per_sec : np.ndarray
            Stride length interpolated per second.
        sec_centers : np.ndarray
            Time centers for interpolation.
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