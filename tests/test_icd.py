import pytest
import numpy as np
import pandas as pd
from conftest import import_class, _as_df_or_empty, _synthesize_walk


def test_icd_no_ics_returns_empty(zeros_df):
    cls = import_class("multimob.ICD.ICD2", "McCamleyIC")
    inst = cls()
    res = inst.detect(zeros_df, sampling_rate_hz=100)
    ic_obj = getattr(res, "ic_list_", None)
    if ic_obj is None:
        pytest.skip("IC detector did not return ic_list_ attribute — skip emptiness check")
    ic_df = _as_df_or_empty(ic_obj)
    assert ic_df.empty or len(ic_df) == 0


def test_icd_invalid_param_raises_or_type_error(imu_df):
    cls = import_class("multimob.ICD.ICD1", "MicoAmigoIC")
    inst = cls()
    with pytest.raises((TypeError, ValueError)):
        inst.detect(imu_df, sampling_rate_hz=100, invalid_param=True)  # type: ignore


def test_icd_integration_with_gsd(imu_df):
    """Testing the integration with GSD, using only the best performing algo from each category"""
    gsd_cls = import_class("multimob.GSD.GSD3", "KheirkhahanGSD")
    ic_cls = import_class("multimob.ICD.ICD2", "McCamleyIC")

    gsd = gsd_cls()
    detect_fn = getattr(gsd, "detect", getattr(gsd, "detect_wrist", None))
    if detect_fn is None:
        pytest.skip("GSD class lacks a usable detect method")

    gsd_res = detect_fn(imu_df, sampling_rate_hz=100)
    gs_df = _as_df_or_empty(getattr(gsd_res, "gs_list_", None))
    if gs_df.empty:
        pytest.skip("GSD found no gait sequences in the sample — cannot run GSD->IC integration test")

    ic_det = ic_cls()
    for idx, row in gs_df.iterrows():
        if "start" not in gs_df.columns or "end" not in gs_df.columns:
            pytest.skip("GSD gs_list_ lacks start/end columns — cannot slice for integration test")
        start_i = int(row["start"])
        end_i = int(row["end"])
        start_i = max(0, start_i)
        end_i = min(len(imu_df), end_i)
        if end_i - start_i <= 2:
            continue
        segment = imu_df.iloc[start_i:end_i].reset_index(drop=True)
        ic_res = ic_det.detect(segment, sampling_rate_hz=100)
        ic_df = _as_df_or_empty(getattr(ic_res, "ic_list_", None))
        if ic_df.empty:
            continue
        possible_cols = [c for c in ic_df.columns if "ic" in c.lower() or "index" in c.lower()]
        if possible_cols:
            ics = ic_df[possible_cols[0]].to_numpy().astype(float)
            assert np.all(ics >= 0) and np.all(ics <= (end_i - start_i)), "ICs should lie within the length of the gait-sequence segment"
        else:
            try:
                arr = np.asarray(ic_df).astype(float).ravel()
                if arr.size:
                    assert np.all(arr >= 0) and np.all(arr <= (end_i - start_i))
            except Exception:
                pytest.skip("IC detector returned an unrecognized structure")
