import numpy as np

def peakfind(x, ws):
    """
    Find local maxima in a signal using a sliding window approach.

    Parameters
    ----------
    x : array-like
        Input signal.
    ws : int
        Window size (must be odd, at least 3).

    Returns
    -------
    maxout : ndarray
        Indices and values of local maxima.
    """

    # Ensure x is a numpy array and flatten to 1D
    x = np.array(x).flatten()

    # Ensure window size is odd and at least 3
    if ws % 2 == 0:
        ws += 1
    ws = max(ws, 3)

    # Padding to handle boundary issues (like MATLAB's NaN padding)
    npad = ws // 2
    x_padded = np.pad(x, (npad, npad), mode='constant', constant_values=np.nan)

    locmax = []

    # Sliding window approach to evaluate maxima in each window
    for i in range(npad, len(x_padded) - npad):
        window = x_padded[i - npad:i + npad + 1]
        center_value = window[npad]

        # Check if the center is a local maximum
        if center_value > np.nanmax(np.delete(window, npad)):
            locmax.append(i - npad)

    # Remove the first and last samples to avoid artificial extremes
    locmax = [idx for idx in locmax if 0 < idx < len(x)]

    # Convert maxima locations to numpy array
    locmax = np.array(locmax)

    # Extract values of maxima
    maxout = np.column_stack((locmax, x[locmax]))

    return maxout
