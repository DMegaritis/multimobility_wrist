from multimob.ICD.ICD2 import McCamleyIC
from multimob.utils.data_loader import load_imu_data

imu_data = load_imu_data()

# only one bout of walking
imu_data = imu_data[962:1427]

# Create an instance of the McCamley class
ICs = McCamleyIC().detect(imu_data, sampling_rate_hz=100)

print(ICs.ic_list_)
