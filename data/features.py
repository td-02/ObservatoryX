from __future__ import annotations

import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd_signal(series: pd.Series) -> pd.Series:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return signal


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["returns"] = frame["close"].pct_change()
    frame["realized_vol_20"] = frame["returns"].rolling(20).std()
    frame["rsi_14"] = _rsi(frame["close"], 14)
    frame["macd_signal"] = _macd_signal(frame["close"])
    frame["intrabar_range"] = (frame["high"] - frame["low"]) / frame["close"]
    return frame
