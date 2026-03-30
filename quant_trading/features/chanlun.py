from __future__ import annotations

import numpy as np
import pandas as pd


def _identify_fractals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    fractal_top = (frame["high"] > frame["high"].shift(1)) & (frame["high"] > frame["high"].shift(-1))
    fractal_bottom = (frame["low"] < frame["low"].shift(1)) & (frame["low"] < frame["low"].shift(-1))
    return fractal_top.fillna(False), fractal_bottom.fillna(False)


def _build_turning_points(
    frame: pd.DataFrame,
    fractal_top: pd.Series,
    fractal_bottom: pd.Series,
    min_separation: int,
) -> list[dict]:
    candidates: list[dict] = []
    for idx in range(len(frame)):
        if fractal_top.iloc[idx]:
            candidates.append({"idx": idx, "kind": "top", "price": float(frame["high"].iloc[idx])})
        elif fractal_bottom.iloc[idx]:
            candidates.append({"idx": idx, "kind": "bottom", "price": float(frame["low"].iloc[idx])})

    points: list[dict] = []
    for candidate in candidates:
        if not points:
            points.append(candidate)
            continue

        previous = points[-1]
        if candidate["kind"] == previous["kind"]:
            if candidate["kind"] == "top" and candidate["price"] >= previous["price"]:
                points[-1] = candidate
            elif candidate["kind"] == "bottom" and candidate["price"] <= previous["price"]:
                points[-1] = candidate
            continue

        if candidate["idx"] - previous["idx"] < min_separation:
            continue
        points.append(candidate)

    return points


def _build_strokes(frame: pd.DataFrame, points: list[dict]) -> list[dict]:
    strokes: list[dict] = []
    for left, right in zip(points, points[1:]):
        if left["kind"] == right["kind"]:
            continue
        segment = frame.iloc[left["idx"] : right["idx"] + 1]
        direction = 1 if left["kind"] == "bottom" and right["kind"] == "top" else -1
        strokes.append(
            {
                "start_idx": left["idx"],
                "end_idx": right["idx"],
                "direction": direction,
                "high": float(segment["high"].max()),
                "low": float(segment["low"].min()),
            }
        )
    return strokes


def add_chanlun_features(frame: pd.DataFrame, min_separation: int = 3) -> pd.DataFrame:
    """
    This is a simplified engineering approximation of Chan analysis:
    fractals -> alternating turning points -> strokes -> overlap-based center.
    """

    output = frame.copy()
    fractal_top, fractal_bottom = _identify_fractals(output)
    turning_points = _build_turning_points(output, fractal_top, fractal_bottom, min_separation)
    strokes = _build_strokes(output, turning_points)

    output["chan_fractal_top"] = fractal_top.astype(int)
    output["chan_fractal_bottom"] = fractal_bottom.astype(int)
    length = len(output)
    stroke_dir = np.zeros(length, dtype=np.int8)
    center_high = np.full(length, np.nan, dtype=np.float32)
    center_low = np.full(length, np.nan, dtype=np.float32)

    for stroke in strokes:
        stroke_dir[stroke["start_idx"] : stroke["end_idx"] + 1] = stroke["direction"]

    if len(strokes) >= 3:
        for stroke_idx in range(2, len(strokes)):
            recent_three = strokes[stroke_idx - 2 : stroke_idx + 1]
            recent_center_high = min(stroke["high"] for stroke in recent_three)
            recent_center_low = max(stroke["low"] for stroke in recent_three)
            if recent_center_high <= recent_center_low:
                continue

            segment_start = strokes[stroke_idx]["end_idx"]
            segment_end = strokes[stroke_idx + 1]["end_idx"] if stroke_idx + 1 < len(strokes) else length - 1
            center_high[segment_start : segment_end + 1] = recent_center_high
            center_low[segment_start : segment_end + 1] = recent_center_low

    output["chan_stroke_dir"] = stroke_dir
    output["chan_center_high"] = center_high
    output["chan_center_low"] = center_low

    rolling_mean = output["close"].rolling(10, min_periods=10).mean().fillna(output["close"])
    has_center = np.isfinite(center_high) & np.isfinite(center_low)
    trend_state = np.zeros(length, dtype=np.int8)

    bullish_momentum = (stroke_dir > 0) & (output["close"].to_numpy() > rolling_mean.to_numpy())
    bearish_momentum = (stroke_dir < 0) & (output["close"].to_numpy() < rolling_mean.to_numpy())
    bullish_center_break = has_center & (stroke_dir > 0) & (output["close"].to_numpy() > center_high)
    bearish_center_break = has_center & (stroke_dir < 0) & (output["close"].to_numpy() < center_low)

    trend_state[bullish_momentum] = 1
    trend_state[bearish_momentum] = -1
    trend_state[bullish_center_break] = 1
    trend_state[bearish_center_break] = -1
    output["chan_trend_state"] = trend_state

    recent_bottom = output["chan_fractal_bottom"].shift(1).rolling(9, min_periods=1).max().fillna(0).astype(bool)
    center_break = output["chan_center_high"].notna() & (output["close"] > output["chan_center_high"])
    momentum_follow = (output["chan_stroke_dir"] > 0) & recent_bottom & (output["close"] > rolling_mean.fillna(output["close"]))
    output["chan_long_confirmation"] = ((center_break | momentum_follow) & (output["chan_trend_state"] >= 0)).astype(int)
    return output
