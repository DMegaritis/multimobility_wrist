# Multimobility_Wrist

[![DOI](https://zenodo.org/badge/1037389095.svg)](https://doi.org/10.5281/zenodo.16926413)

Python implementations of signal-processing algorithms for validating mobility outcomes from wrist-worn sensors in people with multimorbidity.  
Includes gait detection, initial contact detection, and stride-length estimation methods.

This repository contains **algorithm implementations only**, along with examples and tests.  
A future repository will provide full pipelines and aggregation functions with minimal dependencies.

---

## 📖 Background

These algorithms were developed and validated as part of the project:

**Real-world gait in people with multiple long-term conditions**

- ISRCTN: ISRCTN25008143  
- ClinicalTrials.gov: NCT06473168  
- IRAS: 340676 | CPMS: 61049  

The work was funded by the Medical Research Council (MRC) Gap Fund award (UKRI/MR/B000091/1).

---

## ⚙️ Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/DMegaritis/multimobility_wrist.git
```

## ⚡ Usage

Example scripts are provided in the examples/ directory. Typical workflows include loading IMU data, detecting gait events, detecting initial contacts, and estimating stride length.

1. Load IMU data
```python
from multimob.utils.data_loader import load_imu_data

imu_data = load_imu_data()
```

2. Gait detection (GSD)
```python
from multimob.GSD.GSD2 import HickeyGSD

# Preprocess and detect gait events
GSDs = HickeyGSD().preprocess(imu_data, sampling_rate_hz=100).detect_wrist()
print(GSDs.gs_list_)
```

3. Initial contact detection (ICD)
```python
from multimob.ICD.ICD1 import MicoAmigoIC

# Optionally select a single bout of walking
imu_data = imu_data[962:1427]

ICs = MicoAmigoIC().detect(imu_data, sampling_rate_hz=100)
print(ICs.ic_list_)
```

4. Stride length estimation (SL)
```python
from multimob.SL.SL1 import WeinbergSL
from multimob.utils.data_loader import load_ICs

reference_ic = load_ICs()
sl = WeinbergSL(version="wrist_adaptive").calculate(
    data=imu_data,
    initial_contacts=reference_ic,
    sampling_rate_hz=100
)
print(sl.stride_length_per_sec_)
print(sl.average_stride_length_)
```

## 🧪 Tests
This repository includes a comprehensive suite of tests for all modules (GSD, ICD, SL). Tests cover edge cases, invalid inputs, and integration between modules.

**GSD (Gait Sequence Detection)**

-Validates that empty signals return no gait sequences.

-Checks that invalid parameters raise TypeError or ValueError.

-Confirms that the output DataFrame contains start and end columns with integer-like values.


**ICD (Initial Contact Detection)**

-Ensures that empty input produces no initial contacts.

-Checks that invalid parameters raise appropriate errors.

-Integrates with GSD outputs to verify that detected initial contacts lie within gait sequences.

**SL (Stride / Step Length)**

-Warns and returns NaN if there are insufficient initial contacts.

-Checks physical plausibility of stride lengths (non-negative, ≤ 3 m).

-Validates temporal aggregation matches the expected length from the data.

-Integrates with ICD and GSD outputs to ensure consistency.


Run all tests with:

```python
poetry run pytest
```

Verbose output:

```python
poetry run pytest -v
```

## 📄 Citation
If you use this code in your work, please cite the Zenodo record:

```python
@software{megaritis2025multimobility_wrist,
  author       = {Megaritis, Dimitrios},
  title        = {Multimobility_Wrist: Algorithms for Digital Mobility Assessment from Wrist-Worn Sensors},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16926413},
  url          = {https://doi.org/10.5281/zenodo.16926413}
}
```
Plain text:
Megaritis, D. (2025). *Multimobility_Wrist: Algorithms for Digital Mobility Assessment from Wrist-Worn Sensors*. Zenodo. https://doi.org/10.5281/zenodo.16926413


## 📢 Acknowledgment
This work was funded by the Medical Research Council (MRC) Gap Fund award (UKRI/MR/B000091/1).
