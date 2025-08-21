from typing_extensions import Self, Literal
from multimob.GSD.utils.merge_bouts import merge_bouts
from multimob.GSD.utils.cwb import cwb
import pandas as pd
import  numpy as np
from mobgap.data_transform import (
    chain_transformers,
    GaussianFilter,
    ButterworthFilter
)


class MacLeanGSD:
    """
    Implementation of the Gait Sequence Detection (GSD) algorithm by MacLean et al. (2023) [1]_.
    This implementation is adapted and fine tuned specifically for wrist-worn accelerometer data.
    It identifies walking bouts by applying filtering, thresholding, and bout-merging
    steps to 3-axis accelerometer signals.

    The wrist implementation follows the structure of the original algorithm 
    but uses fixed thresholds tuned for wrist signals. The main steps are:

    1. Preprocessing: A 4th-order zero-lag low-pass Butterworth filter 
       (cutoff 0.25 Hz) is applied to remove high-frequency noise. 
       The filtered signal is subtracted from the raw data to preserve 
       movement-related frequencies.

    2. Magnitude calculation: The vector norm (3D Euclidean norm) of the 
       acceleration is computed to produce a single magnitude signal.

    3. Activity detection: A binary threshold of 0.11g * 9.81 is applied 
       to the magnitude signal. Points above threshold are marked active (1), 
       below threshold inactive (0). This binary activity signal is smoothed 
       with a Gaussian-weighted moving average.

    4. Bout segmentation: Start and end points of candidate walking bouts 
       are identified. Gaps shorter than a threshold (0.1 s * sampling rate) 
       are merged with neighboring bouts.

    5. Walking classification: Each bout is validated by comparing the 
       smoothed magnitude against a walking threshold of 0.5g * 9.81. 
       If fewer than 5% of points exceed this threshold, the bout is 
       classified as walking.

    6. Post-processing: Bouts shorter than 2 seconds are discarded. 
       Optionally, short breaks (< 3 s) between bouts can be merged to form 
       continuous walking bouts.

    Attributes
    ----------
    gs_list_ : pandas.DataFrame
        DataFrame containing the start and end times of detected gait 
        sequences. Each sequence is labeled with a unique `gs_id` index.

    Notes
    -----
    - This release supports only the fixed-threshold wrist-worn version.
    - Filters and thresholds are designed for 100 Hz data. Resampling is 
      required for other sampling rates.
    - Orientation correction steps from the original MATLAB implementation 
      are omitted for a wrist sensor implementation.

    References
    ----------
    [1] MacLean, M. K., Rehman, R. Z. U., Kerse, N., Taylor, L., Rochester, L.,
        & Del Din, S. (2023). Walking Bout Detection for People Living in Long 
        Residential Care: A Computationally Efficient Algorithm for a 3-Axis 
        Accelerometer on the Lower Back. Sensors, 23(21), 8973. 
        https://doi.org/10.3390/s23218973
    """

    def __init__(self, *, sampling_rate_hz: float = 100, version: Literal["wrist"] = "wrist", cwb: bool = True) -> None:

        """
        Initialize the class.

        Parameters
        ----------
        sampling_rate_hz : float
        version : str, optional
            The version of the algorithm to use. For this release we only support a version for wrist worn monitors.
        """

        self.sampling_rate_hz = sampling_rate_hz
        self.version = version
        self.cwb = cwb

        self.threshold_binary = 0.11 * 9.81
        self.gap_threshold = 0.4
        self.gap_index = 0.1 * self.sampling_rate_hz
        self.walk_threshold = 0.5 * 9.81
        self.walk_index = 0.05


    def detect(self, data: pd.DataFrame) -> Self:
        """
        Detect gait sequences in the provided data.

        Parameters
        ----------
        data : pandas.DataFrame
            Input acceleration data with the first three columns as x, y, z axes.

        Returns
        -------
        Self
            Returns the instance with detected gait sequences stored in the `gs_list_` attribute.
        """

        self.data = data

        # selecting only the accelerometer data
        acc = self.data.iloc[:, 0:3]

        # Performing a low pass butterworth filter
        cutoff = 0.25
        # class instance
        filter_chain = [("butter", ButterworthFilter(order=4, cutoff_freq_hz=cutoff, filter_type='lowpass'))]
        acc_filt = chain_transformers(acc, filter_chain, sampling_rate_hz=self.sampling_rate_hz)

        # subtracting the filtered from the original signal
        minusacc = acc - acc_filt

        # calculating the norm of the acceleration
        magnitude = np.linalg.norm(minusacc, axis=1)

        # finding datapoint above the threshold (1) or below (0)
        magnitude_thresh = (magnitude > self.threshold_binary).astype(int)

        # if first or last value is 1 then inserting 0 at the beginning or end. This is to detect the start and end of the first and last bouts
        if magnitude_thresh[0] == 1:
            magnitude_thresh = np.insert(magnitude_thresh, 0, 0)
        if magnitude_thresh[-1] == 1:
            magnitude_thresh = np.append(magnitude_thresh, 0)

        # finding the start and end of the gait sequence on the binary signal
        # 0 = no change, 1 = start of bout, -1 = end of bout
        mtdiff = np.diff(magnitude_thresh)
        start = np.where(mtdiff == 1)[0] + 1
        end = np.where(mtdiff == -1)[0] + 1

        # if end and start are empty then we shortcut since no gait was detected
        if start.size == 0 or end.size == 0:
            self.gs_list_ = pd.DataFrame(columns=["start", "end"])
            self.gs_list_.index.name = 'gs_id'
            return self

        # if the first end is before the first start we start the walking bout from the beginning of the file
        if end[0] < start[0]:
            start = np.insert(start, 0, 0)
        # if the last start is after the last end we end the walking bout at the end of the file
        if end[-1] < start[-1]:
            end = np.append(end, len(mtdiff))

        # smoothing binary data, sigma has been selected from comparisons with matlab which does not have a sigma parameter (it is calculated internally)
        filter_chain = [("gaussian_1", GaussianFilter(sigma_s=10 / self.sampling_rate_hz))]
        smoothed_data = np.asarray(chain_transformers(magnitude_thresh.astype(float), filter_chain, sampling_rate_hz=self.sampling_rate_hz))

        # gapindex includes end of each bout and start of the next
        # and a column to store the sum of the smoothed data below the threshold
        gapindex = np.column_stack((end[:-1], start[1:], np.zeros(len(start) - 1, dtype=int)))

        # iterating through each gap to make checks
        gapindex[:, 2] = np.array([
            np.sum(smoothed_data[start:end] < self.gap_threshold)
            for start, end in zip(gapindex[:, 0], gapindex[:, 1])
        ])

        # identifying invalid gaps (i.e., gaps which should be included as walking bouts)
        # using the threshold indicating the number of data points below which a gap is invalid
        invalid_gaps = gapindex[:, 2] < self.gap_index

        # extracting all ends of invalid gaps (to merge them with walking bouts)
        boutends_notinactive = gapindex[invalid_gaps, 1]

        # extracting all starts where the preceding gap is invalid (to merge them with walking bouts)
        boutstarts_notinactive = gapindex[invalid_gaps, 0]

        valid_wb_indices = []

        # we need to merge the detected bouts starts and ends with the gaps which include activity
        merged_starts, merged_ends = merge_bouts(start, end, boutstarts_notinactive, boutends_notinactive)

        # gaussian filter in the euclidian norm signal
        filter_chain = [("gaussian_1", GaussianFilter(sigma_s=2 / self.sampling_rate_hz))]
        magnitude_smooth = np.asarray(chain_transformers(magnitude, filter_chain, sampling_rate_hz=self.sampling_rate_hz))

        # having all the walking bouts (originating from the detected starts and ends merged with the invalid gaps)
        # we perform a check to see if the signal exceeds a threshold for a specific number of data points

        # in order to reduce computational resources we calculate the walk_index_values value for each walking bout outside of the loop
        # walk_index_values is the limit based on the walk_index
        walk_index_values = np.round((merged_ends - merged_starts) * self.walk_index).astype(int)

        for i in range(len(merged_starts)):
            # extracting the smoothed magnitude data for each walking bout and
            # determining if the data exceeds the threshold
            dt_thresh_bout = (magnitude_smooth[merged_starts[i]:merged_ends[i]] > self.walk_threshold).astype(int)

            # if the data exceeds the threshold for less than walk_index_values points (5% of the bout length), we include the bout
            # using precomputed walk_index_values values rather than calculating them on the fly in each loop
            if np.sum(dt_thresh_bout) < walk_index_values[i]:
                valid_wb_indices.append(i)

        # extracting valid walking bout starts and ends after the threshold check
        walk_start_checked = np.array(merged_starts)[valid_wb_indices]
        walk_end_checked = np.array(merged_ends)[valid_wb_indices]

        # a second check includes removing bouts shorter than 2 seconds
        bout_duration = walk_end_checked - walk_start_checked
        # minimum bout length is 2 seconds
        min_bout_length = 2 * self.sampling_rate_hz

        # filtering out short bouts
        valid_bouts = bout_duration > min_bout_length

        # final start and end of non-walking bouts
        final_wb_start = walk_start_checked[valid_bouts]
        final_wb_end = walk_end_checked[valid_bouts]

        # creating a dataframe with the final start and end of the walking bouts
        gs = pd.DataFrame({"start": final_wb_start, "end": final_wb_end})
        # setting the index name
        gs.index.name = 'gs_id'
        # Clipping start and end to be within limits of file
        gs[['start', 'end']] = np.clip(gs[['start', 'end']], 0, len(self.data))

        # Creating Continuous Walking Bouts from micro walking bouts
        if self.cwb:
            gs = cwb(gs, max_break_seconds=3, sampling_rate=self.sampling_rate_hz)

        self.gs_list_ = gs

        return self