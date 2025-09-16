from multimob.ICD.ICD6 import GuIC
from multimob.utils.data_loader import load_imu_data

"""
This is an example on how to use the Gu algo to detect initial contacts.
"""

imu_data = load_imu_data()

# only one bout of walking
imu_data = imu_data[962:1427]

# Create an instance of the GuIC class
ICs = GuIC().detect(imu_data, sampling_rate_hz=100)

print(ICs.ic_list_)