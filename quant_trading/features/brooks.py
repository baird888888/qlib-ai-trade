from __future__ import annotations

import pandas as pd


def add_brooks_features(frame: pd.DataFrame, body_window: int = 20) -> pd.DataFrame:
    output = frame.copy()
    body = (output["close"] - output["open"]).abs()
    average_body = body.rolling(body_window, min_periods=body_window).mean()
    bullish_close_near_high = output["close_location"] > 0.65
    bearish_close_near_low = output["close_location"] < 0.35

    output["brooks_bull_trend_bar"] = (
        (output["close"] > output["open"])
        & bullish_close_near_high
        & (body > average_body.fillna(body))
    ).astype(int)

    pullback_bar = (output["close"] < output["open"]) & bearish_close_near_low
    pullback_recent = pullback_bar.shift(1).rolling(3, min_periods=1).max().fillna(0).astype(bool)
    inside_bar = (output["high"] < output["high"].shift(1)) & (output["low"] > output["low"].shift(1))
    inside_bar_fail = inside_bar.shift(1, fill_value=False) & (output["close"] > output["high"].shift(1))
    breakout_above_previous = output["close"] > output["high"].shift(1)
    second_entry_long = output["brooks_bull_trend_bar"].astype(bool) & pullback_recent & breakout_above_previous

    output["brooks_bull_signal"] = (
        output["brooks_bull_trend_bar"].astype(bool)
        & (inside_bar_fail | second_entry_long | breakout_above_previous)
    ).astype(int)
    return output
