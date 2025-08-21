import pandas as pd
import numpy as np

def cart_to_spher(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Cartesian coordinates (x, y, z) in a DataFrame to spherical coordinates (azimuth, elevation, norm).

    Args:
    data (pd.DataFrame): A DataFrame with 3 columns representing x, y, z coordinates.

    Returns:
    pd.DataFrame: A DataFrame with the original data and new columns for azimuth, elevation, and norm.
    """

    # Ensure the DataFrame has 3 columns
    assert len(data.columns) == 3, "DataFrame must have exactly 3 columns"

    # Calculate the vector magnitude, azimuth, and elevation for each row
    r = np.linalg.norm(data.iloc[:, 0:3].values, axis=1)
    azimuth = np.arctan2(data.iloc[:, 1], data.iloc[:, 0])
    elevation = np.arcsin(data.iloc[:, 2] / r)

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    data = data.copy()

    # Add the calculated columns to the DataFrame
    data.loc[:, 'azimuth'] = azimuth
    data.loc[:, 'elevation'] = elevation
    data.loc[:, 'norm'] = r

    # Convert angles from radians to degrees
    #data['azimuth_deg'] = np.degrees(data['azimuth'])
    #data['elevation_deg'] = np.degrees(data['elevation'])

    # Return the updated DataFrame
    return data
