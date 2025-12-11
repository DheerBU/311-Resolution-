# tests/test_pipeline_utils.py
import pandas as pd
import numpy as np

from midterm_pipeline import basic_clean, add_simple_rolling_volume, build_feature_pipeline

def make_dummy_df():
    df = pd.DataFrame({
        "case_id": [1, 2],
        "open_dt": ["2024-01-01T10:00:00Z", "2024-01-02T12:00:00Z"],
        "closed_dt": ["2024-01-01T15:00:00Z", "2024-01-03T12:00:00Z"],
        "reason": ["Sanitation", "Street Lights"],
        "type": ["Request A", "Request B"],
        "source": ["Phone", "App"],
        "latitude": [42.35, 42.36],
        "longitude": [-71.06, -71.05],
        "subject": ["Trash pickup", "Streetlight out"],
    })
    return df

def test_basic_clean_duration():
    df = make_dummy_df()
    cleaned = basic_clean(df)
    assert "duration_hours" in cleaned.columns
    # first row: 5 hours, second row: 24 hours
    assert np.isclose(cleaned["duration_hours"].iloc[0], 5.0)
    assert np.isclose(cleaned["duration_hours"].iloc[1], 24.0)

def test_add_simple_rolling_volume():
    df = make_dummy_df()
    cleaned = basic_clean(df)
    with_volume = add_simple_rolling_volume(cleaned)
    assert "daily_vol_roll7" in with_volume.columns
    assert not with_volume["daily_vol_roll7"].isna().any()

def test_build_feature_pipeline_runs():
    df = make_dummy_df()
    cleaned = basic_clean(df)
    with_volume = add_simple_rolling_volume(cleaned)
    pre = build_feature_pipeline(with_volume)
    X = pre.fit_transform(with_volume)
    # we just check it runs and produces non-empty design matrix
    assert X.shape[0] == len(with_volume)
    assert X.shape[1] > 0
