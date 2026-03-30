from __future__ import annotations

import re

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift(1)).abs()
    low_close = (frame["low"] - frame["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def add_technical_features(
    frame: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 60,
    atr_period: int = 14,
    volume_window: int = 20,
    breakout_window: int = 20,
) -> pd.DataFrame:
    output = frame.copy()
    output["ema_fast"] = ema(output["close"], ema_fast)
    output["ema_slow"] = ema(output["close"], ema_slow)
    output["atr"] = atr(output, atr_period)
    output["atr_pct"] = output["atr"] / output["close"].replace(0, np.nan)
    output["volume_mean"] = output["vol"].rolling(volume_window, min_periods=volume_window).mean()
    output["relative_volume"] = output["vol"] / output["volume_mean"].replace(0, np.nan)
    output["bar_range"] = output["high"] - output["low"]
    output["close_location"] = (
        (output["close"] - output["low"]) / output["bar_range"].replace(0, np.nan)
    ).clip(lower=0.0, upper=1.0)
    output["returns_1"] = output["close"].pct_change(1)
    output["returns_3"] = output["close"].pct_change(3)
    output["returns_12"] = output["close"].pct_change(12)
    output["rolling_high"] = output["high"].shift(1).rolling(breakout_window, min_periods=breakout_window).max()
    output["rolling_low"] = output["low"].shift(1).rolling(breakout_window, min_periods=breakout_window).min()
    output["trend_strength"] = (
        (output["ema_fast"] - output["ema_slow"]) / output["atr"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    output["trend_strength"] = output["trend_strength"].fillna(0.0)
    output["relative_volume"] = output["relative_volume"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    output["atr_pct"] = output["atr_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    output["close_location"] = output["close_location"].fillna(0.5)
    return output


def _timeframe_to_timedelta(value: str | None) -> pd.Timedelta | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    match = re.fullmatch(r"(\d+)\s*([a-z]+)", text)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    unit_map = {
        "m": "min",
        "min": "min",
        "mins": "min",
        "minute": "min",
        "minutes": "min",
        "h": "h",
        "hr": "h",
        "hour": "h",
        "hours": "h",
        "d": "d",
        "day": "d",
        "days": "d",
    }
    normalized_unit = unit_map.get(unit)
    if normalized_unit is None:
        return None
    return pd.to_timedelta(amount, unit=normalized_unit)


def _infer_frame_interval(frame: pd.DataFrame) -> pd.Timedelta | None:
    if "bar" in frame.columns:
        non_null_bar = frame["bar"].dropna()
        if not non_null_bar.empty:
            explicit = _timeframe_to_timedelta(str(non_null_bar.iloc[0]))
            if explicit is not None:
                return explicit

    timestamps = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna().sort_values()
    if len(timestamps) < 2:
        return None
    diffs = timestamps.diff().dropna()
    if diffs.empty:
        return None
    median_diff = diffs.median()
    return median_diff if pd.notna(median_diff) and median_diff > pd.Timedelta(0) else None


def merge_higher_timeframe(
    lower_frame: pd.DataFrame,
    higher_frame: pd.DataFrame,
    prefix: str = "h1_",
) -> pd.DataFrame:
    lower = lower_frame.sort_values("timestamp").copy()
    higher = higher_frame.sort_values("timestamp").copy()
    higher["timestamp"] = pd.to_datetime(higher["timestamp"], utc=True)
    interval = _infer_frame_interval(higher)
    if interval is not None:
        # Match only fully closed informative candles, similar to Freqtrade's informative merge guidance.
        higher["timestamp"] = higher["timestamp"] + interval
    rename_map = {
        column: f"{prefix}{column}"
        for column in higher.columns
        if column not in {"timestamp", "inst_id", "bar"}
    }
    higher = higher.rename(columns=rename_map)
    merged = pd.merge_asof(lower, higher, on="timestamp", direction="backward")
    return merged
