from multimob.utils.data_loader import load_imu_data
from multimob.GSD.GSD2 import HickeyGSD
import matplotlib.pyplot as plt

"""
This is an example on how to use the Hickey algo to detect gait events while having a compatible output with my pipeline.
"""

imu_data = load_imu_data()

# Calling the class with the preprocess and detect at once
GSDs = HickeyGSD().preprocess(imu_data, sampling_rate_hz=100).detect_wrist()

print(GSDs.gs_list_)