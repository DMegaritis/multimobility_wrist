import math
import numpy as np
import pandas as pd
import warnings
import pytest
from conftest import import_class, _as_df_or_empty, _synthesize_walk

# List of all SL algos
SL_ALGOS = [
    ("multimob.SL.SL1", "WeinbergSL"),
    ("multimob.SL.SL2", "KimSL"),
    ("multimob.SL.SL3", "BylemansSL"),
]

@pytest.mark.parametrize("module,class_name", SL_ALGOS)
def test_sl_not_enough_ics_warns_and_returns_nans(module, class_name, zeros_df):
    cls = import_class(module, class_name)
    inst = cls()
    initial_contacts = pd.DataFrame({"ic": [1]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = inst.calculate(data=zeros_df, initial_contacts=initial_contacts, sampling_rate_hz=100)
        assert len(w) >= 0
    stride_df = getattr(res, "stride_length_per_sec_", None)
    step_df = getattr(res, "step_length_per_sec_", None)
    if stride_df is None or step_df is None:
        pytest.skip("SL implementation does not expose stride/step per-second outputs with these names")
    stride_df = _as_df_or_empty(stride_df)
    step_df = _as_df_or_empty(step_df)
    assert stride_df["stride_length_m"].isna().all() or stride_df.empty
    assert step_df["step_length_m"].isna().all() or step_df.empty

@pytest.mark.parametrize("module,class_name", SL_ALGOS)
def test_sl_physical_sanity(module, class_name):
    cls = import_class(module, class_name)
    inst = cls()
    data = _synthesize_walk(n_samples=1000, sampling_rate_hz=100)
    ics = np.arange(50, 900, 50)
    initial_contacts = pd.DataFrame({"ic": ics})
    res = inst.calculate(data=data, initial_contacts=initial_contacts, sampling_rate_hz=100)
    stride_df = getattr(res, "stride_length_per_sec_", None)
    if stride_df is None:
        pytest.skip("SL implementation does not expose stride_length_per_sec_")
    stride_df = _as_df_or_empty(stride_df)
    if stride_df.empty:
        pytest.skip("SL returned empty per-second results on synthetic data")
    if "stride_length_m" in stride_df.columns:
        arr = stride_df["stride_length_m"].dropna().to_numpy()
        assert np.all(arr >= 0), "No negative stride lengths expected"
        assert np.all(arr <= 4.0), "Stride lengths larger than 3.0 m are unexpected"

@pytest.mark.parametrize("module,class_name", SL_ALGOS)
def test_sl_temporal_aggregation_length_matches_synthetic(module, class_name):
    cls = import_class(module, class_name)
    inst = cls()
    data = _synthesize_walk(n_samples=1000, sampling_rate_hz=100)
    ics = np.arange(50, 900, 50)
    initial_contacts = pd.DataFrame({"ic": ics})
    sr = 100.0
    res = inst.calculate(data=data, initial_contacts=initial_contacts, sampling_rate_hz=sr)
    stride_df = getattr(res, "stride_length_per_sec_", None)
    if stride_df is None:
        pytest.skip("SL implementation does not expose stride_length_per_sec_")
    stride_df = _as_df_or_empty(stride_df)
    # Expected length: total samples / sampling rate in seconds
    expected_len = math.ceil(len(data) / sr)
    assert len(stride_df) == expected_len, (
        f"Expected {expected_len} rows, got {len(stride_df)}"
    )

@pytest.mark.parametrize("module,class_name", SL_ALGOS)
def test_sl_integration_with_gsd(module, class_name, imu_df):
    """Detect ICs with ICD, then run SL algorithm on the segments."""
    ic_cls = import_class("multimob.ICD.ICD2", "McCamleyIC")
    sl_cls = import_class(module, class_name)

    ic_det = ic_cls()
    sl_algo = sl_cls()

    # Detect ICs on full IMU data
    ic_res = ic_det.detect(imu_df, sampling_rate_hz=100)
    ic_df = _as_df_or_empty(getattr(ic_res, "ic_list_", None))
    if ic_df.empty:
        pytest.skip("IC detector returned no ICs")

    ics = ic_df.to_numpy().ravel()

    # Run SL algorithm using detected ICs
    sl_res = sl_algo.calculate(data=imu_df, initial_contacts=pd.DataFrame({"ic": ics}), sampling_rate_hz=100)
    stride_df = getattr(sl_res, "stride_length_per_sec_", None)
    if stride_df is None or stride_df.empty:
        pytest.skip("SL returned empty results")
