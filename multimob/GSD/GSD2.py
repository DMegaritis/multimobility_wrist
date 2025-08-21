from typing_extensions import Self
import pandas as pd
import numpy as np
from multimob.GSD.utils.gravity_remove_butter import gravity_motion_butterworth
from multimob.GSD.utils.cwb import cwb
from mobgap.data_transform import (
    Resample,
    chain_transformers,
    ButterworthFilter
)

class HickeyGSD:
    """
    Implementation of the Gait Sequence Detection algorithm by Hickey et al. (2017) [1], adapted and fine-tuned for wrist-worn accelerometer data.

    The algorithm detects periods of walking by analyzing the norm of acceleration signals.
    It assumes higher variability in the acceleration norm during movement compared to still periods.
    The present version (wrist) uses the norm of the acceleration with a similar rationale than the lower back.
    However, the upright threshold is used as maximum activity threshold for the wrist rather than an upright threshold.
    This threshold is derived from identifying the 95th percentile of the signal from walking bouts in a sample of 108 participants with diverse conditions.



    [1] Hickey, A., Del Din, S., Rochester, L., & Godfrey, A. (2017).
    Detecting free-living steps and walking bouts: validating an algorithm for macro gait analysis.
    Physiological measurement, 38(1), N1-N15.

    Notes:
    - Acceleration is preprocessed by removing gravity and calculating the norm of all three axes.
    - Thresholds for stillness and activity are based on empirical data from a diverse sample.
    - The algorithm operates on resampled data (default 100 Hz), hence it can be considered sample rate agnostic.
    - Continuous walking bouts can optionally be merged if separated by short breaks (≤3s).
    """

    def __init__(self, *, version: str = "wrist", cwb: bool = True):
        """
         Initialise the HickeyGSD class for wrist-worn sensors.

         Parameters
         ----------
         version : str
             Only 'wrist' is supported in this release.
         cwb : bool
             If True, merges micro-walking bouts into continuous walking bouts.
         """

        if version != "wrist":
            raise ValueError("Only version='wrist' is supported in this release.")

        self.version = version
        self.cwb = cwb

        thresholdupright = 9.5
        thresholdstill = 0.1

        self.data = None
        self.gs_list_ = None
        self.ThresholdStill = thresholdstill
        self.ThresholdUpright = thresholdupright

    def preprocess(self, data, *, sampling_rate_hz: float = 100, target_sampling_rate_hz: float = 100) -> Self:
        """
        Preprocess wrist acceleration data.

        Steps:
        1. Remove gravity from all three axes.
        2. Compute the norm of the acceleration vector.
        3. Resample to the target sampling rate (default 100 Hz).

        Parameters
        ----------
        data : pd.DataFrame
            Input acceleration data with three axes.
        sampling_rate_hz : float
            Original sampling rate of the data.
        target_sampling_rate_hz : float, optional
            Sampling rate to resample to (default 100 Hz).

        Returns
        -------
        Self
            Returns the instance with preprocessed data stored in `imu_preprocessed`.
        """

        self.data = data
        self.sampling_rate_hz =sampling_rate_hz
        self.target_sampling_rate_hz = target_sampling_rate_hz
        acc = self.data.iloc[:, 0:3]

        # removing gravity from the 3 axes using custom function
        acc_nograv = gravity_motion_butterworth(acc, sampling_rate_hz)

        # calculating norm of the acceleration
        acc_norm = np.linalg.norm(acc_nograv, axis=1)
        # converting to pandas DataFrame
        acc_norm = pd.DataFrame(acc_norm, columns=['acc_norm'])

        # Target sample rate is 100 which is similar to the sensor.
        # I added a check to see if the sensor sample rate is 100 and if it is then we don't resample
        if self.sampling_rate_hz != 100:
            filter_chain = [("resampling", Resample(self.target_sampling_rate_hz))]
            acc_norm = chain_transformers(acc_norm, filter_chain, sampling_rate_hz=self.sampling_rate_hz)

        self.imu_preprocessed = acc_norm

        return self


    def detect_wrist(self) -> Self:
        """
        Detect walking bouts from wrist acceleration data.

        Steps:
        1. Center the norm of the acceleration.
        2. Apply low-pass filtering (17 Hz cutoff).
        3. Divide signal into 0.1s windows, calculate SD and mean.
        4. Identify movement windows based on thresholds.
        5. Merge consecutive bouts separated by ≤2.25s.
        6. Remove bouts shorter than 0.5s (require at least 2 strides).
        7. Optionally, merge into continuous walking bouts (CWB) if `self.cwb=True`.

        Returns
        -------
        Self
            Returns the instance with detected gait sequences stored in `gs_list_`.
        """

        if self.version in ["original", "improved"]:
            raise ValueError(
                "The detect_wrist() function is intended for wrist-worn sensors. "
                f"Current version='{self.version}' is for lower-back sensors. "
                "Use detect() instead."
            )

        data = self.imu_preprocessed

        # centering the norm acceleration
        acc_norm_mean = data.mean()
        acc_norm_centered = data - acc_norm_mean

        # Defining the window size which is 0.1s
        n = self.target_sampling_rate_hz / (self.target_sampling_rate_hz / 10)

        # Calculating the number of 0.1s windows present in the data
        win_num = int(len(acc_norm_centered) // n)

        # Performing a low pass butterworth filter on the data
        cutoff = 17
        # class instance
        filter_chain = [("butter", ButterworthFilter(order=2, cutoff_freq_hz=cutoff, filter_type='lowpass'))]

        # application to all corrected axes
        acc_filt = np.asarray(chain_transformers(acc_norm_centered, filter_chain, sampling_rate_hz=self.sampling_rate_hz))

        # SD and mean calculation for all axes every 0.1s
        std_acc = np.zeros(win_num)
        mean_acc = np.zeros(win_num)

        for i in range(win_num):
            start_idx = int(i * n)
            end_idx = int((i + 1) * n)

            std_acc[i] = np.std(acc_filt[start_idx:end_idx])
            mean_acc[i] = np.mean(data[start_idx:end_idx])

        # Initialize the result array with zeros
        i_array_move_st_si = np.zeros(win_num)

        # Apply the conditions to each window
        for i in range(win_num):
            if std_acc[i] >= self.ThresholdStill and mean_acc[i] <= self.ThresholdUpright:
                i_array_move_st_si[i] = 1

        # if i_array_move_st_si is all ones then the function should return a dataframe with the start and end of the signal!
        if i_array_move_st_si.sum() == win_num:
            self.gs_list_ = pd.DataFrame([[0, len(acc_norm_centered)]], columns=["start", "end"])
            # Add an index "gs_id" that starts from 0
            self.gs_list_.index.name = 'gs_id'
            return self

        # if i_array_move_st_si is all zeros then there is no walking and the function should return an empty dataframe
        if i_array_move_st_si.sum() == 0:
            self.gs_list_ = pd.DataFrame(columns=["start", "end"])
            # Add an index "gs_id" that starts from 0
            self.gs_list_.index.name = 'gs_id'
            return self

        #Calculating starts and ends of walking
        # first and last elements should be 0 to identify transitions
        i_array_move_st_si[0] = 0
        i_array_move_st_si[-1] = 0

        # difference in array elements indicate start (1) and stop (-1)
        diffs = np.diff(i_array_move_st_si)
        bout_start_move_st_si = np.where(diffs == 1)[0] + 1
        bout_stop_move_st_si = np.where(diffs == -1)[0] + 1

        # Combine start and stop bouts into one array
        bout_array_move_st_si = np.column_stack((bout_start_move_st_si, bout_stop_move_st_si))

        # Initialize Difference Arrays
        betweenbbout_array_move_st_si = np.zeros(len(bout_array_move_st_si), dtype=int)

        # Set the first value of DifferenceArrayAMoveStSi to be similar to the first value of BoutArrayMoveStSi
        # the reason is that this represents the difference from the beginning of the signal to the first bout
        betweenbbout_array_move_st_si[0] = bout_array_move_st_si[0,0]

        # Calculate the bout lengths
        boutlength_array_move_st_si = bout_stop_move_st_si - bout_start_move_st_si

        # Working out the differences between consecutive bouts
        for i in range(1, len(bout_array_move_st_si)):
            betweenbbout_array_move_st_si[i] = abs(bout_stop_move_st_si[i-1] - bout_start_move_st_si[i])

        # Combine all the variables into one array, including start, stop, difference between bouts and bout length
        difference_array_move_st_si = np.column_stack((
            bout_start_move_st_si,
            bout_stop_move_st_si,
            betweenbbout_array_move_st_si,
            boutlength_array_move_st_si
        ))

        # Here we merge bouts which are 2.25s or less apart. Rationale is that two consequtive ICs
        # are expected to be from 0.25 to 2.25s appart so if two bouts have a smaller break than 2.25s then the break is walking
        # Using 22.5 due to scaling of windows to 0.1s so 2.25 seconds is 22.5 values
        i = 1  # Start from the second bout since we are merging with the previous one
        while i < len(difference_array_move_st_si):
            if difference_array_move_st_si[i, 2] <= 22.5:
                # Merge current bout with the last one (index i-1)
                difference_array_move_st_si[i - 1, 1] = difference_array_move_st_si[i, 1]  # Update the stop time of the last bout
                difference_array_move_st_si[i - 1, 3] += difference_array_move_st_si[i, 3]  # Combine bout lengths

                # Remove the current row after merging
                difference_array_move_st_si = np.delete(difference_array_move_st_si, i, axis=0)
            else:
                i += 1  # Move to the next row if no merge is needed

        # According to consensus (Mob-D) a stride cannot be lower than 0.2s and if we need at least 2 strides to form a bout
        # we need to remove bouts that are shorter than 0.5s. This is in accordance with the original publication as well
        # Using 5 due to scaling of windows to 0.1s so half a second is 5 values
        difference_array_move_st_si = difference_array_move_st_si[difference_array_move_st_si[:, 3] > 5]

        # Removing the 3rd column indicating the "pause" between bouts
        difference_array_move_st_si = np.delete(difference_array_move_st_si, 2, axis=1)

        # Converting back to samples
        difference_array_move_st_si = (difference_array_move_st_si * n).astype(int)

        # Create a pandas dataframe with the start and end of the gait sequences
        gs_list_ = pd.DataFrame(difference_array_move_st_si[:, [0, 1]], columns=["start", "end"])

        # Add an index "gs_id" that starts from 0
        gs_list_.index.name = 'gs_id'
        # Clipping start and end to be within limits of file
        gs_list_[['start', 'end']] = np.clip(gs_list_[['start', 'end']], 0, len(self.data))

        # Creating Continuous Walking Bouts from micro walking bouts
        if self.cwb:
            gs_list_ = cwb(gs_list_, max_break_seconds=3, sampling_rate=self.target_sampling_rate_hz)

        self.gs_list_ = gs_list_

        return self