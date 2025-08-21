from multimob.SL.SL1 import WeinbergSL
from multimob.utils.data_loader import load_imu_data, load_ICs

"""
This is an example on how to use the intensity based Weinberg stride length algorithm.
"""

imu_data = load_imu_data()
# only one bout of walking
imu_data = imu_data[962:1427]

reference_ic = load_ICs()

# calling the stride length algorithm
sl = WeinbergSL(version="wrist_adaptive").calculate(
    data=imu_data,
    initial_contacts=reference_ic,
    sampling_rate_hz=100
)

print(sl.stride_length_per_sec_)
print(sl.average_stride_length_)
