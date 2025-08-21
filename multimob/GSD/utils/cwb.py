import pandas as pd

def cwb(df, max_break_seconds=3, sampling_rate=100):
    """
    Creating a Continuous Walking Bout (CWB) from micro walking bouts.
    Effectively merges walking bouts when the gap between them is shorter than 3 seconds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['start', 'end'] and sorted by start time.
    max_break : int
        Maximum allowed gap (in seconds) between bouts to merge them.
    sampling_rate : float
        Sampling rate in Hz, used to convert seconds to samples.

    Returns
    -------
    pd.DataFrame
        Merged bouts with columns ['start', 'end']. If input is empty then we return an empty DataFrame with the same structure.
    """
    df = df.sort_values("start").reset_index(drop=True)

    if df.empty:
        empty_df = pd.DataFrame(columns=["start", "end"])
        empty_df.index.name = "gs_id"
        return empty_df

    max_break = max_break_seconds * sampling_rate

    merged = []
    current_start = df.loc[0, "start"]
    current_end = df.loc[0, "end"]

    for i in range(1, len(df)):
        gap = df.loc[i, "start"] - current_end

        if gap <= max_break:
            # Extend current bout
            current_end = max(current_end, df.loc[i, "end"])
        else:
            # Save current bout and start a new one
            merged.append({"start": current_start, "end": current_end})
            current_start = df.loc[i, "start"]
            current_end = df.loc[i, "end"]

    # Append the last bout
    merged.append({"start": current_start, "end": current_end})

    return pd.DataFrame(merged)