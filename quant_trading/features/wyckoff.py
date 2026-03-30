from __future__ import annotations

import numpy as np
import pandas as pd


def add_wyckoff_features(
    frame: pd.DataFrame,
    range_window: int = 36,
    recent_window: int = 12,
) -> pd.DataFrame:
    output = frame.copy()
    support = output["low"].shift(1).rolling(range_window, min_periods=range_window).min()
    resistance = output["high"].shift(1).rolling(range_window, min_periods=range_window).max()
    range_width = resistance - support
    range_pct = range_width / output["close"].replace(0, np.nan)

    near_support = output["close"] <= (support + range_width * 0.35)
    near_resistance = output["close"] >= (resistance - range_width * 0.35)
    inside_range = (output["close"] >= support) & (output["close"] <= resistance)

    accumulation = (
        inside_range
        & near_support
        & range_pct.between(0.01, 0.12)
        & output["relative_volume"].between(0.7, 1.8)
    )
    spring = (
        (output["low"] < support * 0.9985)
        & (output["close"] > support)
        & (output["close_location"] > 0.55)
        & (output["relative_volume"] > 1.05)
    )
    upthrust = (
        (output["high"] > resistance * 1.0015)
        & (output["close"] < resistance)
        & (output["close_location"] < 0.45)
        & (output["relative_volume"] > 1.05)
    )

    recent_accumulation = accumulation.shift(1).rolling(recent_window, min_periods=1).max().fillna(0).astype(bool)
    recent_spring = spring.shift(1).rolling(recent_window, min_periods=1).max().fillna(0).astype(bool)
    recent_upthrust = upthrust.shift(1).rolling(recent_window, min_periods=1).max().fillna(0).astype(bool)

    breakout_long = (
        (output["close"] > resistance * 1.001)
        & (output["relative_volume"] > 1.1)
        & (recent_accumulation | recent_spring)
        & ~recent_upthrust
    )

    phase = np.select(
        [
            accumulation,
            breakout_long,
            near_resistance & inside_range,
            output["close"] < support.fillna(-np.inf),
        ],
        ["accumulation", "markup", "distribution", "markdown"],
        default="neutral",
    )

    output["wyckoff_support"] = support
    output["wyckoff_resistance"] = resistance
    output["wyckoff_accumulation"] = accumulation.astype(int)
    output["wyckoff_spring"] = spring.astype(int)
    output["wyckoff_upthrust"] = upthrust.astype(int)
    output["wyckoff_breakout_long"] = breakout_long.astype(int)
    output["wyckoff_phase"] = phase
    return output
