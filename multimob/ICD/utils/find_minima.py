from mobgap.initial_contacts._utils import find_zero_crossings
import numpy as np


def _find_minima(signal: np.ndarray) -> np.ndarray:
    """
    This function finds minima between two zero crossings.
    """
    zero_crossings = find_zero_crossings(signal, "both", refine=False).astype("int64")

    if len(zero_crossings) == 0:
        return np.array([])

    # Determine where the zero-crossings transition from positive to negative or vice versa
    crossing_signs = signal[zero_crossings] >= 0

    # Adjust indices to define ranges between positive-to-negative and negative-to-positive crossings
    pos_to_neg_crossings = zero_crossings[crossing_signs] + 1
    neg_to_pos_crossings = zero_crossings[~crossing_signs] + 1

    # If the first crossing is negative-to-positive, discard it
    if not crossing_signs[0]:
        neg_to_pos_crossings = neg_to_pos_crossings[1:]

    # Find the local minima in the signal between the positive-to-negative and negative-to-positive crossings
    minima = np.array([
        np.argmin(signal[start:end]) + start
        for start, end in zip(pos_to_neg_crossings, neg_to_pos_crossings)
    ]).astype("int64")

    return minima
