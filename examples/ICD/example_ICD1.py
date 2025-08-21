from multimob.ICD.ICD1 import MicoAmigoIC
from multimob.utils.data_loader import load_imu_data

imu_data = load_imu_data()

# only one bout of walking
imu_data = imu_data[962:1427]

# Create an instance of the MicoAmigoIC class
ICs = MicoAmigoIC().detect(imu_data, sampling_rate_hz=100)
print(ICs.ic_list_)