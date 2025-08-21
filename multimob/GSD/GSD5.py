import warnings
from typing_extensions import Self, Literal
import pandas as pd
import  numpy as np
from scipy.signal import welch, correlate, find_peaks
from mobgap.utils.array_handling import sliding_window_view
from multimob.GSD.utils.cwb import cwb
from mobgap.data_transform import (
    chain_transformers,
    ButterworthFilter
)


class KerenGSD:
    """
    Implementation of the Gait Sequence Detection (GSD) algorithm for wrist-worn accelerometer data, 
    based on Keren et al. (2021) [1]_. The algorithm detects walking bouts from 3-axis acceleration 
    by applying a sequence of preprocessing, spectral, and regularity checks.

    Two optimised variants are provided:
    - **"wrist" (fixed thresholds)**: uses fixed fine tuned threshold in people with multimorbidity.  
    - **"wrist_adaptive" (adaptive thresholds)**: thresholds are set adaptively based on the percentile of 
      the detrended acceleration distribution.  

    The granularity of the implementation is 1 second.

    Steps of the algorithm:
    1. **Data Preprocessing**: compute Euclidean norm, apply low-pass filtering, detrend.  
    2. **Peak Detection**: detect peaks above threshold.  
    3. **Windowing**: segment into 6 s windows with 5 s overlap.  
    4. **Standard Deviation**: check variability in each window.  
    5. **Power Spectral Density (PSD)**: verify dominant frequency lies in walking range (0.5–3 Hz).  
    6. **Autocorrelation**: assess step/stride regularity (>0.15).  
    7. **Combining Conditions**: retain windows meeting all conditions, merge into walking bouts.  

    Notes
    -----
    - The algorithm discards the first and last 3 seconds of data due to windowing.
    - Designed for 100–128 Hz sampling rates. Resampling is required if a different rate is used.  
    - The adaptive version may improve generalisation across heterogeneous datasets.  

    References
    ----------
    [1] Keren K, Busse M, Fritz NE, Muratori LM, Gazit E, Hillel I, Scheinowitz M, 
    Gurevich T, Inbar N, Omer N, Hausdorff JM, Quinn L. 
    Quantification of Daily-Living Gait Quantity and Quality Using a Wrist-Worn Accelerometer 
    in Huntington's Disease. *Front Neurol*. 2021;12:719442. 
    doi: 10.3389/fneur.2021.719442
    """

    def __init__(self, *, version: Literal["wrist", "wrist_adaptive"] = "wrist", cwb: bool=True):
        """
        Initialise the GSD algorithm.

        Parameters
        ----------
        version : {"wrist", "wrist_adaptive"}, default="wrist"
            The variant of the algorithm to use:
            - "wrist": fixed thresholds optimised for people with multimorbidity.
            - "wrist_adaptive": thresholds adaptively set based on signal distribution.
        cwb : bool, default=True
            Whether to merge micro walking bouts into continuous walking bouts (CWB).
        """

        self.version = version
        self.cwb = cwb

        if self.version == "wrist":
            self.threshold = 0.08 * 9.81
            self.threshold_sd = 0.07 * 9.81
            self.window_size = 6
            self.overlap = 5
        elif self.version == "wrist_adaptive":
            self.threshold_percentile = 84
            self.threshold_sd = 0.07 * 9.81
            self.window_size = 6
            self.overlap = 5

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float = 100) -> Self:
        """
        Detect gait sequences in wrist-worn accelerometer data.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame with three columns corresponding to acceleration axes (x, y, z).
        sampling_rate_hz : float, optional
            Sampling rate of the input data in Hz.

        Returns
        -------
        self : KerenGSD
            The instance with detected gait sequences stored in the `gs_list_` attribute
            as a DataFrame with "start" and "end" indices (in samples).
        """

        self.sampling_rate_hz = sampling_rate_hz
        self.data = data

        # check if the signal is long enough for windowing, if not return empty array
        if len(self.data) < self.window_size * self.sampling_rate_hz:
            warnings.warn("The signal is too short for windowing. Returning empty gait sequence list.", stacklevel=1)
            self.gs_list_ = pd.DataFrame(columns=["start", "end"])
            self.gs_list_.index.name = 'gs_id'
            return self


        # 1. calculating the eucledean norm of the data
        acc_norm = np.linalg.norm(self.data.iloc[:, 0:3], axis=1)


        # 2. low pass butterworth filter
        cutoff = 15
        # class instance
        filter_chain = [("butter", ButterworthFilter(order=4, cutoff_freq_hz=cutoff, filter_type='lowpass'))]
        acc_filt = chain_transformers(acc_norm, filter_chain, sampling_rate_hz=self.sampling_rate_hz)


        # 3. removing DC component
        acc_detrend = acc_filt - np.mean(acc_filt)

        # Setting the threshold for the adaptive version
        if self.version in ["wrist_adaptive"]:
            self.threshold = np.percentile(acc_detrend, self.threshold_percentile)
            print(self.threshold)

        # 4. creating binary signal with data above threshold (peaks only)
        # finding peaks withing the areas with >0.1g
        peaks, _ = find_peaks(acc_detrend, height=self.threshold)

        if len(peaks) == 0:
            self.gs_list_ = pd.DataFrame(columns=["start", "end"])
            self.gs_list_.index.name = 'gs_id'
            return self

        binary_peaks_thresh = np.zeros_like(acc_detrend, dtype=int)
        binary_peaks_thresh[peaks] = 1


        # 5. dividing signal into windows with a duration of 6 seconds, with 5-second overlap
        window_size_samples = self.window_size * self.sampling_rate_hz
        overlap_samples = self.overlap * self.sampling_rate_hz
        windows = sliding_window_view(acc_detrend, window_size_samples, overlap_samples)

        # checking the presence of at least one peak in the center second of each window (using step 4)
        binary_peak_thresh_windows = sliding_window_view(binary_peaks_thresh, window_size_samples, overlap_samples)
        # defining start and end of center second
        center = int(self.window_size*self.sampling_rate_hz / 2)
        center_range = int(self.sampling_rate_hz / 2)
        center_seconds_start = center-center_range
        center_seconds_end = center+center_range
        # check presence of peaks in the center seconds of each window
        binary_peak_presence = np.any(binary_peak_thresh_windows[:, center_seconds_start:center_seconds_end],
                                      axis=1).astype(int)


        # 6. calculating the standard deviation in each window
        std = np.std(windows, axis=1)

        # creating binary signal with data above threshold_sd, both thresholds are similar but are included as
        # two separate numbers to allow for fine tunning in the future
        binary_std_thresh = (std > self.threshold_sd).astype(int)


        # 7. compute Power Spectral Density (PSD) using Welch's method
        freq_results = []

        for window in windows:
            f, Pxx = welch(window, fs=self.sampling_rate_hz, window='hamming', nperseg=len(window))

            # finding the frequency with the highest power
            max_power_idx = np.argmax(Pxx)  # index of max power
            max_freq = f[max_power_idx]  # corresponding frequency

            freq_results.append(max_freq)

        # converting to pd DataFrame
        psd_df = pd.DataFrame({'Max_Freq': freq_results})

        # creating binary signal based on
        binary_psd_thresh = np.array(((psd_df['Max_Freq'] > 0.5) & (psd_df['Max_Freq'] < 3)).astype(int))


        # 8. Perform autocorrelation analysis for regularity
        # here we apply regularity check to both steps and strides as original paper does not specify which one
        regularity_check_results = []

        for window in windows:
            # Compute autocorrelation
            autocorr = correlate(window, window, mode='full')
            autocorr = autocorr[autocorr.size // 2:]  # keeping positive lags

            # Normalise (so value at zero lag is 1)
            autocorr = autocorr / autocorr[0]

            # Find peaks in autocorrelation
            peaks, _ = find_peaks(autocorr, height=0)

            if len(peaks) >= 2:
                # paper mentions that first peak is step regularity and second peak is stride regularity (only first and second peaks are mentioned)
                step_peak_idx = peaks[0]
                stride_peak_idx = peaks[1]

                step_regularity = autocorr[step_peak_idx]
                stride_regularity = autocorr[stride_peak_idx]
            else:
                step_regularity, stride_regularity = np.nan, np.nan  # cases with missing peaks

            # checking if both step and stride regularity are greater than 0.15
            if step_regularity > 0.15 and stride_regularity > 0.15:
                regularity_check_results.append(1)
            else:
                regularity_check_results.append(0)

        # Convert results to a boolean array
        binary_regularity_thresh = np.array(regularity_check_results)

        # 9. Combining all conditions and selecting central second windows which meet all conditions
        # stacking all condition arrays
        stacked_arrays = np.vstack((binary_peak_presence, binary_std_thresh, binary_psd_thresh, binary_regularity_thresh))
        # check if all conditions are met in each window
        result = np.all(stacked_arrays == 1, axis=0).astype(int)

        # Selecting middle seconds of windows which meet all conditions
        # calculating the start and end indexes of each window
        num_windows = len(windows)
        step_size = window_size_samples - overlap_samples
        start_indexes = np.arange(0, num_windows * step_size, step_size)
        # calculating the start and end indexes of the mid second of each window
        mid_second_start = start_indexes + center_seconds_start
        mid_second_end = start_indexes + center_seconds_end
        # creating pd.DataFrame with starts and end when all conditions are met
        filtered_starts = mid_second_start[result == 1]
        filtered_ends = mid_second_end[result == 1]
        # creating the dataframe
        seconds_walking = pd.DataFrame({
            'start': filtered_starts,
            'end': filtered_ends
        })

        # merging consequtive seconds
        merge_points = seconds_walking['start'] == seconds_walking['end'].shift()
        group = (~merge_points).cumsum()
        gs = seconds_walking.groupby(group).agg({'start': 'first', 'end': 'last'}).reset_index(drop=True)

        # setting the index name
        gs.index.name = 'gs_id'
        # Clipping start and end to be within limits of file
        gs[['start', 'end']] = np.clip(gs[['start', 'end']], 0, len(self.data))

        # Creating Continuous Walking Bouts from micro walking bouts
        if self.cwb:
            gs = cwb(gs, max_break_seconds=3, sampling_rate=self.sampling_rate_hz)

        self.gs_list_ = gs

        return self