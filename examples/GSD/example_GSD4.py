from multimob.utils.data_loader import load_imu_data
from multimob.GSD.GSD4 import MacLeanGSD

"""
This is an example on how to use the MacLean algo to detect gait events while having a compatible output with my pipeline.
"""

imu_data = load_imu_data()

# Creating instance of the class and calling the preprocess and detect methods
GSDs = MacLeanGSD().detect(imu_data)

print(GSDs.gs_list_)