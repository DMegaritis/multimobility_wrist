from multimob.utils.data_loader import load_imu_data
from multimob.GSD.GSD5 import KerenGSD

"""
This is an example on how to use the Keren algo to detect gait events.
"""

imu_data = load_imu_data()

# Creating instance of the class and calling the preprocess and detect methods
GSDs = KerenGSD().detect(imu_data, sampling_rate_hz=100)

print(GSDs.gs_list_)