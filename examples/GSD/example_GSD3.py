from multimob.utils.data_loader import load_imu_data
from multimob.GSD.GSD3 import KheirkhahanGSD

"""
This is an example on how to use the Kheirkhahan algo to detect gait events while having a compatible output with my pipeline.
"""

imu_data = load_imu_data()

# Creating instance of the class and calling the preprocess and detect methods
GSDs = KheirkhahanGSD().detect(imu_data, sampling_rate_hz=100)

print(GSDs.gs_list_)