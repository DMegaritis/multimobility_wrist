import pandas as pd
import numpy as np
from typing_extensions import Self
from typing import Literal


class GuIC:
    """
    GuIC algorithm for detecting initial contacts (IC) in wrist-worn accelerometer data [1].
    This implementation is fine-tuned and adapted for wrist-worn accelerometer data and people with multimorbidity
    exhibiting diverse gait patterns, with the "improved" and "adaptive" versions provided.

    Workflow
    --------
    1. Euclidean Norm Calculation:
        - Compute the Euclidean norm of the tri-axial accelerometer data.
        - If the standard deviation of acceleration is below 0.025, returns an empty DataFrame.

    2. Peak Detection and Threshold Filtering:
        - Segment the acceleration data into windows of size `k`.
        - Identify local peaks within each segment.
        - Filter peaks based on a magnitude threshold (`mag_thres`).

    3. Periodicity, Similarity, and Continuity Filtering:
        - Periodicity: Compute time intervals between peaks and filter based on
          `period_min` and `period_max`.
        - Similarity: Evaluate the difference in peak magnitudes and filter peaks
          using a similarity threshold (`sim_thres`).
        - Continuity: Assess variance in acceleration between consecutive peaks to
          ensure consistent step detection, applying a continuity threshold (`cont_thres`).

    Parameters
    ----------
    version : Literal["improved", "adaptive"], default="improved"
        Specifies the parameter set to use:
        - "improved": Parameters optimised with fixed thresholds.
        - "adaptive": Similar to "improved" but with an adaptive magnitude threshold.

    Attributes
    ----------
    ic_list_ : pd.DataFrame
        DataFrame containing the indices of the detected IC events.
    k : int
        Window size used for segmentation and peak detection.
    period_min : int
        Minimum allowed interval between consecutive ICs.
    period_max : int
        Maximum allowed interval between consecutive ICs.
    sim_thres : float
        Similarity threshold for filtering peaks based on magnitude differences.
    var_thres : float
        Variance threshold used in continuity filtering.
    mag_thres : float
        Magnitude threshold for peak selection (fixed or adaptive).
    cont_win_size : int
        Number of windows to check for continuity.
    cont_thres : int
        Threshold for variance-based continuity detection.

    Notes
    -----
    - Original smartphone-based parameterisation was not implemented; only
      wrist-optimized "improved" and "adaptive" variants are released.
    - In the "adaptive" version, the magnitude threshold is automatically
      set to the given percentile of the acceleration norm distribution.


    References
    ----------
    [1] Gu, Fuqiang & Khoshelham, Kourosh & Shang, Jianga & Yu, Fangwen & Wei, Zhuo. (2017).
        Robust and Accurate Smartphone-Based Step Counting for Indoor Localization.
        IEEE Sensors Journal. 10.1109/JSEN.2017.2685999.
    """

    ic_list_: pd.DataFrame

    def __init__(self, *, version: Literal["improved", "adaptive"] = "improved") -> None:
        """
        Initialise the GuIC algorithm for wrist-worn accelerometer data.

        Parameters
        ----------
        version : Literal["improved", "adaptive"], optional, default="improved"
            Parameterization variant:
            - "improved": Fixed thresholds.
            - "adaptive": Adaptive magnitude threshold.
        """

        if version not in ["improved", "adaptive"]:
            raise ValueError(f"Invalid version: {version}")


        self.version = version

        if self.version == "improved":
            self.k = 2
            self.period_min = 25
            self.period_max = 120
            self.sim_thres = -0.7 * 9.81
            self.var_thres = 0.0005 * (9.81**2)
            self.mag_thres = 1.1 * 9.81
        elif self.version == "adaptive":
            self.k = 2
            self.period_min = 25
            self.period_max = 110
            self.sim_thres = -0.7 * 9.81
            self.var_thres = 0.005 * (9.81**2)
            self.mag_thres = 70

        self.cont_win_size = 3
        self.cont_thres = 4


    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float) -> Self:
        """
        Detect initial contacts (IC) in wrist-worn accelerometer data using the GuIC algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            Input acceleration data with three columns (x, y, z).
        sampling_rate_hz : float
            Sampling rate of the input data in Hz.

        Returns
        -------
        Self : GuIC
            Instance of the class with the attribute `ic_list_` containing
            detected ICs in a DataFrame of shape (n_steps, 1), where:
            - Index: step_id
            - Column: "ic" (sample index of detected IC events)
        """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # 1. Euclidean norm of the data
        acc_norm = np.linalg.norm(self.data.iloc[:, 0:3], axis=1)

        # If SD of acceleration is less than 0.025, return empty dataframe
        if np.std(acc_norm) < 0.025:
            self.ic_list_ = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
            return self

        # If using the adaptive version then we need to calculate the mag_thres
        if self.version == "adaptive":
            self.mag_thres = np.percentile(acc_norm, self.mag_thres)

        # 2. Calculating peaks
        # calculating half_k
        half_k = round(self.k / 2)
        # calculating segments
        segments = np.floor(len(acc_norm) / self.k).astype(int)
        # looping through the segments, peaks stores the indices of the peaks in the first col and the magnitude of the peaks in the second col
        peaks_list = []

        for i in range(segments):
            # calculating the start and end of the segment
            start = i * self.k
            end = (i + 1) * self.k
            # calculating the peak
            peak = np.argmax(acc_norm[start:end]) + start

            # checking peak validity in the neighbourhood
            start_index_control = peak - half_k
            end_index_control = peak + half_k + 1
            if start_index_control < 0:
                start_index_control = 0
            if end_index_control > len(acc_norm):
                end_index_control = len(acc_norm)

            # confirm local maxima in the neighbourhood
            check_lock = np.argmax(acc_norm[start_index_control:end_index_control])
            if check_lock == (half_k):
                peaks_list.append({"index": peak, "magnitude": acc_norm[peak]})

        peaks = pd.concat([pd.DataFrame([peak]) for peak in peaks_list], ignore_index=True)

        # filtering peaks based on magnitude threshold
        peaks = peaks[peaks["magnitude"] > self.mag_thres]

        # 3. Periodicity, Similarity and Continuity calculation
        # checking if there are more than 2 peaks. Compared to the Shimmer Engineering implementation in R, I have
        # removed the double loop since the first loop checked for more than 10 peaks and the second for more than 2
        if len(peaks) > 2:
            # PERIODICITY
            # calculate periodicity and adding a col with the difference between consecutive peaks
            peaks["periodicity"] = peaks["index"].diff()

            # filtering peaks based on period_min. Keeping NaN values
            peaks = peaks[(peaks["periodicity"] > self.period_min) | (peaks["periodicity"].isna())]
            # filtering peaks based on period_max
            peaks = peaks[(peaks["periodicity"] < self.period_max) | (peaks["periodicity"].isna())]

            # resetting index after filtering
            peaks.reset_index(drop=True, inplace=True)

            # if no peaks are left after filtering, return empty dataframe
            if peaks.empty:
                final_ics = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
                self.ic_list_ = final_ics
                return self

            # SIMILARITY
            # calculating the second difference of peak magnitudes
            peaks["similarity"] = -np.abs(peaks["magnitude"].diff(periods=2))
            # filtering based on similarity threshold
            peaks = peaks[(peaks["similarity"] > self.sim_thres) | (peaks["similarity"].isna())]
            # resetting index after filtering
            peaks.reset_index(drop=True, inplace=True)

            # if no peaks are left after filtering, return empty dataframe
            if peaks.empty:
                final_ics = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
                self.ic_list_ = final_ics
                return self

            # CONTINUITY
            if len(peaks) > 5:
                end_for = len(peaks) - 1
                for i in range(self.cont_thres - 1, end_for):
                    v_count = 0  # counting how many windows were over the variance thres
                    for x in range(1, self.cont_thres + 1): # selecting windows surrounding the peak
                        start_idx = int(peaks.iloc[i - x + 1]['index'])
                        end_idx = int(peaks.iloc[i - x + 2]['index']) + 1

                        # calculating variance of acceleration between consecutive peak windows
                        acc_var = np.var(acc_norm[start_idx:end_idx], ddof=1)
                        if acc_var > self.var_thres:
                            v_count += 1

                    if v_count >= self.cont_win_size:
                        peaks.at[i, 'continuity'] = 1
                    else:
                        peaks.at[i, 'continuity'] = 0

                # filtering continuity, keeping only 1s and NAs (since NAs have not been tested and we dont want to miss ICs)
                peaks = peaks[(peaks["continuity"] == 1) | (peaks["continuity"].isna())]

            # if less than 5 peaks we cannot check continuity, so we assume all peaks are continuous and not returning empty df

            # Creating the final dataframe with the IC indices
            final_ics = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
            final_ics["ic"] = peaks["index"]

        else:
            # if less than 2 peaks, returning empty dataframe
            final_ics = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
            self.ic_list_ = final_ics
            return self

        self.ic_list_ = final_ics

        return self