import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def window(a, w=4, o=2, copy=False):
    """
    Create overlapping windows from a 1D array.

    Parameters
    ----------
    a : np.ndarray
        Input 1D array.
    w : int, optional
        Window size. Default is 4.
    o : int, optional
        Step size or overlap between windows. Default is 2.
    copy : bool, optional
        If True, returns a copy of the windows. Default is False.

    Returns
    -------
    np.ndarray
        Array of overlapping windows from the input array.
    """
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    return view.copy() if copy else view


def sum_partial_overlapping_windows(window_results, original_data, win_size, win_shift):
    """
    Sum values from partially overlapping windows over the original data.

    Parameters
    ----------
    window_results : np.ndarray
        Array containing the results for each window.
    original_data : np.ndarray
        Original data array to map the window results onto.
    win_size : int
        Size of each window.
    win_shift : int
        Shift between windows.

    Returns
    -------
    np.ndarray
        Summed values over the original data positions.
    """
    indices = window(np.arange(len(original_data)), win_size, win_shift)
    flatten_windows = np.broadcast_to(window_results, (win_size, len(indices))).T.flat
    return np.bincount(indices.flat, flatten_windows)


def remove_outliers(data: np.ndarray, lower_percentile, upper_percentile) -> np.ndarray:
    """
    Remove outliers from the data by zeroing elements outside specified percentiles.

    Parameters
    ----------
    data : np.ndarray
        2D array where rows represent different data series and columns represent time points or observations.

    Returns
    -------
    np.ndarray
        Modified array with outliers set to zero.
    """
    lower = np.percentile(data, lower_percentile, axis=1)
    upper = np.percentile(data, upper_percentile, axis=1)
    data.T[~((data.T > lower) & (data.T < upper))] = 0
    return data


def calc_activity_parameter(data: np.ndarray) -> np.ndarray:
    """
    Calculate an activity parameter that represents the ratio of active data within the series.

    Parameters
    ----------
    data : np.ndarray
        2D array where rows represent different data series and columns represent time points or observations.

    Returns
    -------
    np.ndarray
        Array containing the calculated activity parameter for each data series.
    """
    length = data.shape[1]
    max_val = np.nanmax(data, axis=1) * length
    max_val[max_val == 0] = 1  # Prevent division by zero for empty series
    win_sum = np.sum(data, axis=1)
    return (max_val - win_sum) / max_val


def resample_to_orginal_data_length(results: np.ndarray, len_original_data: int) -> np.ndarray:
    """
    Resample an array to match the length of the original data using nearest interpolation.

    Parameters
    ----------
    results : np.ndarray
        1D array of results to be resampled.
    len_original_data : int
        Length of the original data to match.

    Returns
    -------
    np.ndarray
        Resampled array matching the length of the original data.
    """
    interpolator = interp1d(np.arange(len(results)), results, kind='nearest')
    new_x = np.linspace(0, len(results) - 1, len_original_data)
    return interpolator(new_x)

def generate_gs_list(detected_walking: np.ndarray, sampling_rate=1):
    """
    Generate a list of walking bouts (start and end times) from the detected walking array.

    Parameters:
    detected_walking (np.ndarray): A binary array indicating detected walking (1) and non-walking (0) periods.
    sampling_rate (int, optional): The sampling rate of the data. Defaults to 1.

    Returns:
    list: A list of tuples where each tuple contains the start and end times of a walking bout.
    """
    cuts = np.where(np.diff(detected_walking) != 0)[0] + 1
    wbs = np.split(detected_walking, cuts)
    wb_list = [(int(c / sampling_rate), int((c + len(wb)) / sampling_rate)) for c, wb in zip([0] + list(cuts), wbs) if wb[0] == True]

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(wb_list, columns=['start', 'end'])
    df.index.name = 'gs_id'
    return df