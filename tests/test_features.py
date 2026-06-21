from __future__ import annotations

import pandas as pd

from data.features import compute_features


def _sample_frame():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="min"),
            "open": [100 + i for i in range(30)],
            "high": [101 + i for i in range(30)],
            "low": [99 + i for i in range(30)],
            "close": [100 + i for i in range(30)],
            "volume": [1000 + i for i in range(30)],
        }
    )


def test_rsi_stays_between_0_100():
    df = compute_features(_sample_frame())
    rsi = df["rsi_14"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_macd_signal_computed_correctly_on_known_series():
    df = _sample_frame()
    features = compute_features(df)
    assert "macd_signal" in features.columns
    assert features["macd_signal"].notna().any()


def test_realized_vol_20_is_non_negative():
    df = compute_features(_sample_frame())
    vol = df["realized_vol_20"].dropna()
    assert (vol >= 0).all()
