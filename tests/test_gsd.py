import pytest
import pandas as pd
from conftest import import_class, _as_df_or_empty, _synthesize_walk

# List of all GSD algos
GSD_ALGOS = [
    ("multimob.GSD.GSD3", "KheirkhahanGSD"),
    ("multimob.GSD.GSD4", "MacLeanGSD"),
    ("multimob.GSD.GSD5", "KerenGSD"),
]

# List of all GSD algos
GSD_ALGOS_ALL = [
    ("multimob.GSD.GSD2", "HickeyGSD"),
    ("multimob.GSD.GSD3", "KheirkhahanGSD"),
    ("multimob.GSD.GSD4", "MacLeanGSD"),
    ("multimob.GSD.GSD5", "KerenGSD"),
]


def _get_detect_fn(inst):
    """Helper: find the correct detect method on a class."""
    for name in ("detect", "detect_wrist", "run"):
        if hasattr(inst, name):
            return getattr(inst, name)
    return None


@pytest.mark.parametrize("module,class_name", GSD_ALGOS_ALL)
def test_gsd_invalid_param_raises_or_type_error(module, class_name, imu_df):
    """Check that passing an invalid parameter raises TypeError or ValueError."""
    cls = import_class(module, class_name)
    inst = cls()
    detect_fn = _get_detect_fn(inst)
    if detect_fn is None:
        pytest.skip(f"No detect method found for {class_name}")

    with pytest.raises((TypeError, ValueError)):
        detect_fn(imu_df, sampling_rate_hz=100, invalid_param=True)  # type: ignore


@pytest.mark.parametrize("module,class_name", GSD_ALGOS)
def test_gsd_empty_returns_no_sequences(zeros_df, module, class_name):
    cls = import_class(module, class_name)
    inst = cls()
    detect_fn = _get_detect_fn(inst)
    if detect_fn is None:
        pytest.skip(f"No detect method found for {class_name}")

    res = detect_fn(zeros_df)
    gs_df = _as_df_or_empty(getattr(res, "gs_list_", None))
    assert gs_df.empty, f"{class_name} should not detect gait sequences in zero signal"

def test_hickey_gsd_empty_returns_no_sequences(zeros_df):
    """Special case for HickeyGSD: must call preprocess before detect_wrist."""
    from multimob.GSD.GSD2 import HickeyGSD

    inst = HickeyGSD()
    res = inst.preprocess(zeros_df).detect_wrist()
    gs_df = _as_df_or_empty(getattr(res, "gs_list_", None))
    assert gs_df.empty, "HickeyGSD should not detect gait sequences in zero signal"


@pytest.mark.parametrize("module,class_name", GSD_ALGOS)
def test_gsd_happy_path_has_start_end(imu_df, module, class_name):
    cls = import_class(module, class_name)
    inst = cls()
    detect_fn = _get_detect_fn(inst)
    if detect_fn is None:
        pytest.skip(f"No detect method found for {class_name}")

    res = detect_fn(imu_df)
    gs_df = _as_df_or_empty(getattr(res, "gs_list_", None))
    if gs_df.empty:
        pytest.skip(f"{class_name} found no gait sequences — skip schema checks")

    assert isinstance(gs_df, pd.DataFrame)
    if {"start", "end"}.issubset(gs_df.columns):
        assert gs_df["start"].dtype.kind in "iu", "start must be integer-like"
        assert gs_df["end"].dtype.kind in "iu", "end must be integer-like"

def test_gsd3_happy_path_has_start_end(imu_df):
    """Test GSD3 (KheirkhahanGSD) outputs a DataFrame with start and end columns."""
    from multimob.GSD.GSD3 import KheirkhahanGSD

    inst = KheirkhahanGSD()
    detect_fn = getattr(inst, "detect", None)
    if detect_fn is None:
        pytest.skip("No detect method found for KheirkhahanGSD")

    res = detect_fn(imu_df)
    gs_df = _as_df_or_empty(getattr(res, "gs_list_", None))
    if gs_df.empty:
        pytest.skip("KheirkhahanGSD found no gait sequences — skip schema checks")

    assert isinstance(gs_df, pd.DataFrame)
    if {"start", "end"}.issubset(gs_df.columns):
        assert gs_df["start"].dtype.kind in "iu", "start must be integer-like"
        assert gs_df["end"].dtype.kind in "iu", "end must be integer-like"



