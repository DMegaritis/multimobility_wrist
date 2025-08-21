import numpy as np
import pandas as pd
import warnings
from typing import Literal
from multimob.GSD.utils.gravity_remove_butter import gravity_motion_butterworth
from typing_extensions import Self
from multimob.ICD.utils.auto_cov_unbiased import auto_cov_unbiased
from multimob.ICD.utils.dtwDasGupta import dtwdasgupta
from multimob.ICD.utils.peakfind import peakfind
from scipy.signal import detrend, find_peaks
from mobgap.data_transform import ButterworthFilter, chain_transformers
from numpy.lib.stride_tricks import sliding_window_view

class MicoAmigoIC:
    """
    Detect initial contacts (ICs) in gait using the Mico-Amigo algorithm [1] adapted and fine tuned for wrist-worn accelerometer data.

    The algorithm identifies periodicity in acceleration signals, selects a representative
    template segment, and detects points of maximal similarity to determine initial contacts.

    Attributes
    ----------
    ic_list_ : pd.DataFrame
        Detected initial contact indices, with `step_id` as the index and column "ic" indicating
        the sample indices of ICs.

    Notes
    -----
    - Parameters are designed to be sample-rate agnostic, but 100 Hz is recommended.
    - Performance is consistent for sample rates between 100–128 Hz.
    - Gravity removal is applied before calculating the acceleration norm for wrist-worn sensors.

    References
    ----------
    .. [1] Micó-Amigo, M. E., Kingma, I., Ainsworth, E., Walgaard, S., Niessen, M.,
       van Lummel, R. C., & van Dieën, J. H. (2016). A novel accelerometry-based algorithm
       for the detection of step durations over short episodes of gait in healthy elderly.
       Journal of NeuroEngineering and Rehabilitation, 13(1), 38.
    """

    ic_list_: pd.DataFrame

    def __init__ (self, *, version: Literal["wrist"] = "wrist") -> None:
        """
        Initialise the Mico-Amigo IC detection algorithm.

        Parameters
        ----------
        version : Literal["wrist"], default="wrist"
            Currently only the 'wrist' version is supported, optimised for wrist-worn
            accelerometers in populations with multimorbidity.

        Attributes
        ----------
        factorlimit : int
            Factor to limit template size during template selection.
        num_additional_templates : int
            Number of additional templates used to extend the search region.
        peakdistance : float
            Minimum distance between peaks as a fraction of the template size.
        shiftfactor : float
            Fractional shift applied when aligning templates.
        peakdistance_coef : float
            Scaling coefficient for peak detection during final IC estimation.
        event_offset : int
            Offset applied to detected ICs to adjust for template alignment.
        """

        self.version = version
        self.factorlimit = 2
        self.num_additional_templates = 2
        self.peakdistance = 1.1
        self.shiftfactor = 0.15
        self.peakdistance_coef = 1.0
        self.event_offset = 5

        # Initialising attributes to store processed data
        self.ExtraParameters_df = None
        self.templatesize = None
        self.acc_size = None


    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float) -> Self:
        """
        Detect initial contacts from raw accelerometer data.

        Parameters
        ----------
        data : pd.DataFrame
            Accelerometer data with 3 columns corresponding to x, y, z axes.
        sampling_rate_hz : float
            Sampling rate of the input accelerometer data in Hz.

        Returns
        -------
        Self
            The fitted instance of MicoAmigoIC. Detected initial contacts are stored in
            `self.ic_list_`.
        """

        self._data = data
        self._sampling_rate_hz = sampling_rate_hz

        # removing gravity from the 3 axes using custom function. We remove axis in the wrist version since the
        # original used the anteroposterior axis which does not include gravity component, hence it would be appropriate
        # to remove gravity component before caclulating the norm
        acc_all = data.iloc[:, 0:3]
        acc_nograv = gravity_motion_butterworth(acc_all, sampling_rate_hz)

        # calculating norm of the acceleration
        acc = np.linalg.norm(acc_nograv, axis=1)
        acc_size = len(acc)

        # Applying unbiased autocovariance
        ac = auto_cov_unbiased(acc)

        # Slicing the autocovariance to include the last half plus 10 samples
        mid = len(ac) // 2
        ac = ac[mid - 10:]

        # polynomial fit
        x = np.arange(1, len(ac) + 1)
        x = x.T
        y = ac
        p = np.polyfit(x, y, 6)

        # trend of signal
        f_y = np.polyval(p, x)

        # detrending
        residuals = ac - f_y
        possitive_detrend = detrend(residuals)

        # Using mobgap ButterworthFilter
        fc = 0.8*(self._sampling_rate_hz/100)

        # Mob-D filter
        filter_chain = [("Butter_high_pass", ButterworthFilter(order=2, cutoff_freq_hz=fc, filter_type='highpass'))]
        possitive_detrend_high = np.asarray(chain_transformers(possitive_detrend, filter_chain, sampling_rate_hz=self._sampling_rate_hz))

        # Fourier Transform
        l = len(possitive_detrend_high)
        # next power of 2 from length
        nfft = 2 ** int(np.ceil(np.log2(l)))
        # Fourier

        # Detrending for better performance
        # If I do not detrend in python the peak happens at 0
        detrended_signal = possitive_detrend_high - np.mean(possitive_detrend_high)

        y = np.fft.fft(detrended_signal, nfft) / l

        # Frequencies
        f = self._sampling_rate_hz / 2 * np.linspace(0, 1, nfft // 2 + 1)

        # Single sided power spectrum
        power_spectrum = 2 * np.abs(y[:nfft // 2 + 1])

        # Find dominant frequency
        index_ps_dominant_frequency = np.argmax(power_spectrum)
        dominant_frequency = f[index_ps_dominant_frequency]

        if dominant_frequency == 0:
            dominant_frequency = 1

        # Definition of the template size
        templatesize = round((1/dominant_frequency) * self._sampling_rate_hz)

        # template signal definition
        initiallimit = int(np.floor(self.factorlimit * templatesize))
        endlimit = acc_size - initiallimit

        # find peaks (in python given the same indexes but -1)
        peaks, _ = find_peaks(acc, distance = self.peakdistance * templatesize)

        # middlemaxima
        middlemaxima = peaks[(peaks > initiallimit) & (peaks < endlimit)] + 1

        # number of middle maxima
        nmiddlemaxima = len(middlemaxima)

        #shift
        shift = int(np.ceil(self.shiftfactor * templatesize))

        # Initialising variables to avoid warnings
        reference_section = None
        target_section = None

        # Ensambly of templates
        if nmiddlemaxima > 3:
            # initialise results section and new results section
            resultssection = []
            newresultssection = []

            for iRowJunks in range(nmiddlemaxima, 1, -1):
                for iJunks in range(1, iRowJunks - 1):
                    if iRowJunks == nmiddlemaxima:
                        # calculating indices of start and end
                        ref_start = middlemaxima[iJunks- 1] - shift
                        ref_end = middlemaxima[iJunks - 1] + templatesize
                        tgt_start = middlemaxima[iJunks] - shift
                        tgt_end = middlemaxima[iJunks] + templatesize

                        # Checking the bounds for reference_section if they are out of bounds
                        if ref_start < 0:
                            print("reference_section start index is out of bounds, setting to 0")
                            ref_start = 0
                        if ref_end > len(acc):
                            print("reference_section end index exceeds bounds, adjusting to max length")
                            ref_end = len(acc)

                        # Check bounds for target_section if they are out of bounds
                        if tgt_start < 0:
                            print("target_section start index is out of bounds, setting to 0")
                            tgt_start = 0
                        if tgt_end > len(acc):
                            print("target_section end index exceeds bounds, adjusting to max length")
                            tgt_end = len(acc)

                        # Extract sections
                        reference_section = acc[ref_start:ref_end]
                        target_section = acc[tgt_start:tgt_end]

                        # dtw
                        dtwsection = dtwdasgupta(reference_section, target_section)

                        # storing results
                        resultssection.append(dtwsection[:templatesize])

                    else:
                        if iJunks == 1 and iRowJunks < nmiddlemaxima - 1:
                            resultssection = newresultssection.copy()
                            newresultssection = []

                    if len(resultssection) > iJunks:
                        reference_section = resultssection[iJunks]

                    if iJunks + 1 < len(resultssection):
                        target_section = resultssection[iJunks + 1]

                    # dtw
                    if reference_section is not None:
                        dtwsection = dtwdasgupta(reference_section.T, target_section.T)
                    else:
                        # Handle the case where reference_section was not assigned
                        raise ValueError("reference_section was not initialized.")

                    # append
                    if len(newresultssection) < len(resultssection):  # Ensure it's not out of bounds
                        newresultssection.append(dtwsection[:templatesize + 1])  # Store up to templatesize + 1

            newresultssection_array = np.array(newresultssection)
            template = newresultssection_array[0]

        elif nmiddlemaxima == 3:
            # Handling the case where nmiddlemaxima equals 3
            ref_start = middlemaxima[0] - shift
            ref_end = middlemaxima[0] + templatesize
            tgt_start = middlemaxima[1] - shift
            tgt_end = middlemaxima[1] + templatesize

            # Check bounds for reference_section
            if ref_start < 0:
                print("reference_section start index is out of bounds, setting to 0")
                ref_start = 0
            if ref_end > len(acc):
                print("reference_section end index exceeds bounds, adjusting to max length")
                ref_end = len(acc)

            # Check bounds for target_section
            if tgt_start < 0:
                print("target_section start index is out of bounds, setting to 0")
                tgt_start = 0
            if tgt_end > len(acc):
                print("target_section end index exceeds bounds, adjusting to max length")
                tgt_end = len(acc)

            # Extract sections
            reference_section = acc[ref_start:ref_end]
            target_section = acc[tgt_start:tgt_end]

            # Perform DTW
            dtwsection = dtwdasgupta(reference_section.T, target_section.T)

            # Store the template result
            template = dtwsection[:templatesize]

        else:
            warnings.warn("The number of middle maxima is not supported", UserWarning)
            self.ic_list_ = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
            return self

        # Redimention of signal
        # the padding with follow the 'edge' method to avoid introducing artifacts

        # Padding the signal (resegmentedsig)
        len_pad_before = templatesize
        len_pad_after = templatesize*self.num_additional_templates
        resegmentedsig = np.pad(acc, (len_pad_before, len_pad_after), mode='edge')

        # Searching for similarities between templatesignal and resegmented
        # Ensure template size matches templatesize
        template = template[:templatesize]  # truncate if needed

        # Create a 2D array of sliding windows: shape (num_windows, templatesize)
        windows_matrix = sliding_window_view(resegmentedsig, window_shape=templatesize)

        # Means
        template_mean = np.mean(template)
        windows_mean = np.mean(windows_matrix, axis=1)

        # Standard deviations
        template_std = np.std(template)
        windows_std = np.std(windows_matrix, axis=1)

        # Covariance between template and each window
        cov = np.mean((windows_matrix - windows_mean[:, None]) * (template - template_mean), axis=1)

        # Compute correlation coefficients safely
        denominator = template_std * windows_std
        correlation = np.where(denominator == 0, 0, cov / denominator)

        # Calculate ranges
        rangesig = np.ptp(windows_matrix, axis=1)  # peak-to-peak (max - min)
        rangetemplate = np.ptp(template)

        # Scale correlation by range ratio
        r = correlation * rangesig / rangetemplate

        # Standard deviation of differences between template and windows
        sd = np.std(windows_matrix - template, axis=1)

        # Replacing NAs with 0 in R to, NAs come from the stable padded signal indicating minimum similarity
        r[np.isnan(r)] = 0

        # indices were the SD is 0 (maximal similarity)
        idx = np.where(sd == 0)[0]

        # indices were the SD is not 0
        if len(idx) > 0:
            temp = np.copy(sd)
            temp = np.delete(temp, idx) # Remove the indices where SD is 0
            sd[idx] = np.min(temp) # Set the SD where it was 0 to the minimum of the rest of the values

        # normalizing the SD and R
        sd = sd / np.nanmax(sd)
        r = r / np.nanmax(r)

        # calculating coefficient
        coef = (r / sd)

        # normalising coef
        coef = coef / np.nanmax(coef)

        # within the limits of the signal dimensions finding peaks
        peaks_rsd = peakfind(coef, int(np.floor(self.peakdistance_coef * templatesize)))

        hs_pre = peaks_rsd[:, 0] + shift - self.event_offset

        # Selection of events within the signal
        hs = hs_pre[(hs_pre > templatesize) & (hs_pre < acc_size + templatesize)] - templatesize

        # step differences
        steps = np.diff(hs)

        # adjusting last peak if last step is smaller than 0.6 * templatesize
        if steps[-1] < np.floor(0.6 * templatesize):
            hs = hs[:-1]


        self.templatesize = templatesize
        self.acc_size = acc_size

        final_ics = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
        final_ics["ic"] = hs.astype(int)

        self.ic_list_ = final_ics

        return self