import warnings
import numpy as np
import pandas as pd
from typing import Literal
from typing_extensions import Self, Unpack
from mobgap.utils.conversions import as_samples
from mobgap.utils.interpolation import robust_step_para_to_sec




class KimSL:
    """
    Implementation of the Kim intensity-based step length estimation algorithm.
    Supports wrist sensor placements with foot-augmented versions.

    This release supports fine-tuned versions optimised for people with multimorbidity exhibiting diverse gait patterns;
    the original algorithm was developed for dead-reckoning applications.


    Algorithm Overview
    ------------------
    1. Preprocess the acceleration signal: calculate the Euclidean norm of the 3D acceleration and convert to g-units.
    2. Compute the mean acceleration between consecutive initial contacts.
    3. Estimate step length using the Kim formula:
       Step length = KimA * (|mean_acceleration|^(1/3)) + KimB
    4. Interpolate step length to per-second values.
    5. Compute stride length by multiplying step length by 2.

    Parameters
    ----------
    version : str, optional, default="wrist"
        The version of the algorithm to use. Options are:
        - "wrist": standard wrist-based estimation
        - "wrist_adaptive": adaptive scaling based on RMS of acceleration
        - "wrist_foot": wrist-based estimation augmented with foot length
        - "wrist_adaptive_foot": adaptive scaling with foot length augmentation
    KimB : float, optional
        Offset factor for step length calculation in meters.
        For foot-augmented versions, defaults to 0.265 m if not provided.

    Attributes
    ----------
    KimA : float
        Scaling factor used in the step length calculation.
    KimB : float
        Offset factor used in the step length calculation.
    raw_step_length_per_step_ : pd.DataFrame
        Step length per step before interpolation.
    step_length_per_sec_ : pd.DataFrame
        Interpolated step length per second.
    stride_length_per_sec_ : pd.DataFrame
        Interpolated stride length per second.
    average_stride_length_ : float
        Mean stride length over the sequence.
    max_interpolation_gap_s : float
        Maximum allowed gap (seconds) for interpolation between steps.

    Notes
    -----
    - Algorithm is sampling-rate agnostic but performs best with 100 Hz signals.
    - RMS-based adaptive scaling adjusts KimA based on the magnitude of acceleration to account for individual differences.
    - Step length outputs are in meters.

    References
    ----------
    Kim, J. W., Jang, H. J., Hwang, D. H., & Park, C. (2004).
    A step, stride and heading determination for the pedestrian navigation system.
    Journal of Global Positioning Systems, 3(1-2), 273-279.
    """

    max_interpolation_gap_s: float
    raw_step_length_per_step_: pd.DataFrame
    step_length_per_sec_: pd.DataFrame

    def __init__(
        self,
        *,
        version: Literal["wrist", "wrist_adaptive", "wrist_foot", "wrist_adaptive_foot"] = "wrist",
        KimB:float = None
    ) -> None:
        """
        Initialises the KimSL step length estimator.

        Sets algorithm-specific parameters for step length calculation based on the chosen version.

        Parameters
        ----------
        version : {"wrist", "wrist_adaptive", "wrist_foot", "wrist_adaptive_foot"}, optional
            Version of the Kim algorithm to use. Default is "wrist".
        KimB : float, optional
            Offset parameter for step length calculation (in meters). Only used in foot-augmented versions.
        """

        if version not in ["wrist", "wrist_adaptive", "wrist_foot", "wrist_adaptive_foot"]:
            raise ValueError(f"Invalid version: {version}")
        
        self.version = version

        if self.version == "wrist":
            self.KimA = 0.35
            self.KimB = 0
        elif self.version == "wrist_adaptive":
            self.KimA = 0.35
            self.KimB = 0
        elif self.version == "wrist_foot":
            self.KimA = 0.10
            self.KimB = (KimB if KimB is not None else 26.5) * 0.01 # Turning B into meters as the input is in cm
        elif self.version == "wrist_adaptive_foot":
            self.KimA = 0.10
            self.KimB = (KimB if KimB is not None else 26.5) * 0.01 # Turning B into meters as the input is in cm

        self.max_interpolation_gap_s = 3


    def calculate(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.DataFrame,
        *,
        sampling_rate_hz: float
    ) -> Self:
        """
        Calculate step and stride lengths from acceleration data and detected initial contacts.

        Parameters
        ----------
        data : pd.DataFrame
            Input acceleration data (columns should contain x, y, z axes).
        initial_contacts : pd.DataFrame
            DataFrame with detected initial contact indices in a column named "ic".
        sampling_rate_hz : float
            Sampling rate of the input acceleration data in Hz.

        Returns
        -------
        Self
            The instance with calculated step lengths and stride lengths stored in attributes.

        Notes
        -----
        - Issues warnings if initial contacts are not sorted or are insufficient for calculation.
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

        # Using the acceleration norm for the wrist versions
        vacc = np.linalg.norm(self.data.iloc[:, 0:3], axis=1)

        # turning m/s^2 to g since the multimob perform better
        vacc = vacc / 9.81

        # Calling the function to calculate step length
        raw_step_length = self._calc_step_length_kim(vacc, self.ic_list)

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


    def _calc_step_length_kim(
        self,
        vacc: np.ndarray,
        initial_contacts: np.ndarray,
    ) -> np.ndarray:
        """
        Internal function to compute step lengths using Kim's formula.

        Parameters
        ----------
        vacc : np.ndarray
            Preprocessed acceleration signal (Euclidean norm in g-units).
        initial_contacts : np.ndarray
            Array of initial contact indices.

        Returns
        -------
        np.ndarray
            Array of estimated step lengths (meters) for each step.
        """

        # 1. Calculating mean of acceleration between each initial contact
        mean = np.array([np.mean(vacc[ic:ic_next]) for ic, ic_next in zip(initial_contacts, initial_contacts[1:])])

        # 2. If version is adaptive calculate RMS between each initial contact
        if self.version in ["wrist_adaptive", "wrist_adaptive_foot"]:
            rms_values = np.array([np.sqrt(np.mean(np.square(vacc[ic:ic_next]))) for ic, ic_next in zip(initial_contacts, initial_contacts[1:])])

            # calculating the mean rms
            mean_rms = np.mean(rms_values)

            # Amending KimA if mean_rms is not 0
            if mean_rms == 0:
                warnings.warn(
                    "The calculated RMS is 0. Step length calculation will proceed without scaling KimA.",
                    stacklevel=2)
            else:
                self.KimA = self.KimA * mean_rms

        # 3. Calculating step length. Using absolute values to avoid errors
        step_length = self.KimA * (np.abs(mean) ** (1/3)) + self.KimB

        return step_length


    def _set_all_nan(self, sec_centers: np.ndarray, ic_list: np.ndarray) -> None:
        """
        Set all outputs to NaN when step length cannot be calculated.

        Parameters
        ----------
        sec_centers : np.ndarray
            Array of per-second time centers.
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
        Combine step and stride lengths into pandas DataFrames and calculate average stride length.

        Parameters
        ----------
        raw_step_length : np.ndarray
            Step lengths per detected step.
        step_length_per_sec : np.ndarray
            Step lengths interpolated per second.
        stride_length_per_sec : np.ndarray
            Stride lengths interpolated per second.
        sec_centers : np.ndarray
            Per-second time centers corresponding to interpolated values.

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