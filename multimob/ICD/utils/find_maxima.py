from mobgap.initial_contacts._utils import find_zero_crossings
import numpy as np


def _find_maxima(signal: np.ndarray) -> np.ndarray:
    """
    This function finds maxima between two zero crossings.
    """
    zero_crossings = find_zero_crossings(signal, "both", refine=False).astype("int64")

    if len(zero_crossings) == 0:
        return np.array([])

    # Determine where the zero-crossings transition from negative to positive
    neg_to_pos_bool = signal[zero_crossings] < 0

    # Adjust indices to define ranges between positive-to-negative and negative-to-positive crossings
    neg_to_pos_crossings = zero_crossings[neg_to_pos_bool] + 1
    pos_to_neg_crossings = zero_crossings[~neg_to_pos_bool] + 1

    # If the first crossing is positive-to-negative, discard it
    if not neg_to_pos_bool[0]:
        pos_to_neg_crossings = pos_to_neg_crossings[1:]

    # Find the local maxima in the signal between the negative-to-positive and positive-to-negative crossings
    maxima = np.array([
        np.argmax(signal[start:end]) + start
        for start, end in zip(neg_to_pos_crossings, pos_to_neg_crossings)
    ]).astype("int64")

    return maxima
