import numpy as np

def merge_bouts(wb_starts: np.ndarray, wb_ends: np.ndarray, non_wb_starts: np.ndarray, non_wb_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Merges walking bout start and end times, including intervals originally labeled
    as non-walking that should be considered part of walking.

    Parameters:
        wb_starts (array-like): Start times of walking bouts.
        wb_ends (array-like): End times of walking bouts.
        non_wb_starts (array-like): Start times of non-walking intervals to be included.
        non_wb_ends (array-like): End times of non-walking intervals to be included.

    Returns:
        merged_starts (numpy array): Start times of merged walking bouts.
        merged_ends (numpy array): End times of merged walking bouts.
    """
    merged_starts = []
    merged_ends = []

    i = 0
    while i < len(wb_starts):
        start = wb_starts[i]
        end = wb_ends[i]

        # Check for non-WB as gaps which need to be merged
        if i < len(wb_starts) - 1 and wb_ends[i] in non_wb_starts and wb_starts[i+1] in non_wb_ends:
            # Merge the current bout with the next
            end = wb_ends[i+1]
            #i += 1 # Skip the next bout

        merged_starts.append(start)
        merged_ends.append(end)
        i += 1

        j = 0
        while j < len(merged_starts) - 1:
            if merged_ends[j] > merged_starts[j+1]:
                merged_ends[j] = merged_ends[j+1]
                del merged_starts[j+1]
                del merged_ends[j+1]
            else:
                j += 1

    return np.array(merged_starts), np.array(merged_ends)