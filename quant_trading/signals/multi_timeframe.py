from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trading.config import FeatureConfig, RiskConfig, SignalConfig
from quant_trading.features.brooks import add_brooks_features
from quant_trading.features.chanlun import add_chanlun_features
from quant_trading.features.technical import add_technical_features, merge_higher_timeframe
from quant_trading.features.wyckoff import add_wyckoff_features
from quant_trading.risk.kelly import kelly_fraction


def _enrich_frame(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    enriched = add_technical_features(
        frame,
        ema_fast=config.ema_fast,
        ema_slow=config.ema_slow,
        atr_period=config.atr_period,
        volume_window=config.volume_window,
        breakout_window=config.breakout_window,
    )
    enriched = add_chanlun_features(enriched, min_separation=config.chan_min_separation)
    enriched = add_wyckoff_features(
        enriched,
        range_window=config.wyckoff_range_window,
        recent_window=config.wyckoff_recent_window,
    )
    enriched = add_brooks_features(enriched, body_window=config.brooks_body_window)
    return enriched


def _numeric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float64")


def _binary_series(frame: pd.DataFrame, column: str, default: int = 0) -> pd.Series:
    return _numeric_series(frame, column, float(default)).fillna(default).astype(int)


def _range_reversion_components(frame: pd.DataFrame) -> dict[str, pd.Series]:
    close = _numeric_series(frame, "close", 0.0).clip(lower=1e-9)
    h1_close = _numeric_series(frame, "h1_close", 0.0).clip(lower=1e-9)

    support = pd.to_numeric(frame.get("wyckoff_support"), errors="coerce")
    resistance = pd.to_numeric(frame.get("wyckoff_resistance"), errors="coerce")
    h1_support = pd.to_numeric(frame.get("h1_wyckoff_support"), errors="coerce")
    h1_resistance = pd.to_numeric(frame.get("h1_wyckoff_resistance"), errors="coerce")

    range_width = (resistance - support).where((resistance - support) > close * 0.002)
    h1_range_width = (h1_resistance - h1_support).where((h1_resistance - h1_support) > h1_close * 0.002)

    support_position = ((close - support) / range_width).replace([np.inf, -np.inf], np.nan)
    support_position = support_position.clip(lower=0.0, upper=1.0).fillna(0.5)
    resistance_position = (1.0 - support_position).clip(lower=0.0, upper=1.0)

    distance_to_support_pct = ((close - support) / close).replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=0.20)
    distance_to_support_pct = distance_to_support_pct.fillna(0.02)
    distance_to_resistance_pct = ((resistance - close) / close).replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=0.20)
    distance_to_resistance_pct = distance_to_resistance_pct.fillna(0.02)

    range_width_pct = (range_width / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    h1_range_width_pct = (h1_range_width / h1_close).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ema_spread_pct = (
        (_numeric_series(frame, "ema_fast", 0.0) - _numeric_series(frame, "ema_slow", 0.0)).abs() / close
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    h1_ema_spread_pct = (
        (_numeric_series(frame, "h1_ema_fast", 0.0) - _numeric_series(frame, "h1_ema_slow", 0.0)).abs() / h1_close
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    inside_range = support.notna() & resistance.notna() & (close >= support) & (close <= resistance)
    inside_h1_range = h1_support.notna() & h1_resistance.notna() & (h1_close >= h1_support) & (h1_close <= h1_resistance)
    low_trend = _numeric_series(frame, "trend_strength", 0.0).abs() <= 0.55
    low_h1_trend = _numeric_series(frame, "h1_trend_strength", 0.0).abs() <= 0.75
    balanced_ema = ema_spread_pct <= 0.012
    balanced_h1_ema = h1_ema_spread_pct <= 0.018

    range_regime = (
        inside_range
        & inside_h1_range
        & range_width_pct.between(0.008, 0.14)
        & h1_range_width_pct.between(0.01, 0.18)
        & low_trend
        & low_h1_trend
        & balanced_ema
        & balanced_h1_ema
        & (_binary_series(frame, "simons_breakout_long") == 0)
    )
    support_rebound_score = (
        resistance_position * 0.45
        + _numeric_series(frame, "signal_confidence", 0.0).clip(lower=0.0, upper=1.0) * 0.30
        + _numeric_series(frame, "qlib_score_rank", 0.5).clip(lower=0.0, upper=1.0) * 0.20
        + (1.0 - (_numeric_series(frame, "trend_strength", 0.0).abs().clip(lower=0.0, upper=2.0) / 2.0)) * 0.05
    ).clip(lower=0.0, upper=1.0)

    return {
        "range_regime": range_regime.astype(int),
        "range_support_position": support_position,
        "range_resistance_position": resistance_position,
        "range_width_pct": range_width_pct,
        "range_h1_width_pct": h1_range_width_pct,
        "range_distance_to_support_pct": distance_to_support_pct,
        "range_distance_to_resistance_pct": distance_to_resistance_pct,
        "range_rebound_score": support_rebound_score,
    }


def build_multitimeframe_signal_frame(
    lower_frame: pd.DataFrame,
    higher_frame: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    signal_config: SignalConfig | None = None,
    risk_config: RiskConfig | None = None,
) -> pd.DataFrame:
    feature_config = feature_config or FeatureConfig()
    signal_config = signal_config or SignalConfig()
    risk_config = risk_config or RiskConfig()

    lower = _enrich_frame(lower_frame, feature_config)
    higher = _enrich_frame(higher_frame, feature_config)
    merged = merge_higher_timeframe(lower, higher, prefix="h1_").copy()

    breakout_distance = (
        (merged["close"] - merged["rolling_high"]) / merged["atr"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    volume_score = (merged["relative_volume"] - 1.0).clip(lower=0.0, upper=2.0)
    trend_score = merged["trend_strength"].clip(lower=0.0, upper=2.0)
    merged["simons_breakout_score"] = (
        breakout_distance.fillna(0.0).clip(lower=0.0, upper=3.0) * 0.5
        + volume_score.fillna(0.0) * 0.25
        + trend_score.fillna(0.0) * 0.25
    )
    merged["simons_breakout_long"] = (
        (merged["close"] > merged["rolling_high"].fillna(np.inf))
        & (merged["simons_breakout_score"] >= signal_config.simons_breakout_threshold)
    ).astype(int)

    merged["trend_filter_long"] = (
        (merged["h1_ema_fast"] > merged["h1_ema_slow"])
        & (merged["h1_close"] > merged["h1_ema_fast"])
        & (merged["h1_chan_trend_state"] >= 0)
    ).astype(int)

    recent_spring = (
        merged["wyckoff_spring"]
        .shift(1)
        .rolling(signal_config.recent_spring_window, min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )
    recent_accumulation = (
        merged["wyckoff_accumulation"]
        .shift(1)
        .rolling(signal_config.recent_spring_window, min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )
    merged["wyckoff_long_context"] = (
        (merged["wyckoff_breakout_long"].astype(bool) | recent_spring)
        & (recent_accumulation | recent_spring)
        & (merged["wyckoff_upthrust"] == 0)
    ).astype(int)

    confidence_components = (
        merged["trend_filter_long"] * 0.22
        + merged["wyckoff_long_context"] * 0.26
        + merged["brooks_bull_signal"] * 0.16
        + merged["chan_long_confirmation"] * 0.16
        + merged["simons_breakout_long"] * 0.20
    )
    breakout_bonus = (merged["simons_breakout_score"] / 2.0).clip(lower=0.0, upper=0.2)
    merged["signal_confidence"] = (confidence_components + breakout_bonus).clip(lower=0.0, upper=1.0)

    merged["entry_long_signal"] = (
        (merged["trend_filter_long"] == 1)
        & (merged["wyckoff_long_context"] == 1)
        & (merged["brooks_bull_signal"] == 1)
        & (merged["chan_long_confirmation"] == 1)
        & (merged["simons_breakout_long"] == 1)
        & (merged["signal_confidence"] >= signal_config.confidence_threshold)
    ).astype(int)

    merged["exit_long_signal"] = (
        (merged["wyckoff_upthrust"] == 1)
        | ((merged["close"] < merged["ema_fast"]) & (merged["close"] < merged["ema_slow"]))
        | ((merged["chan_trend_state"] < 0) & (merged["close"] < merged["ema_fast"]))
        | ((merged["trend_filter_long"] == 0) & (merged["returns_3"] < 0))
    ).astype(int)

    atr_stop = (merged["atr_pct"] * signal_config.atr_stop_multiple).clip(
        lower=signal_config.min_stop_loss_pct,
        upper=signal_config.max_stop_loss_pct,
    )
    structure_stop = (
        (merged["close"] - merged["wyckoff_support"]) / merged["close"].replace(0, np.nan)
    ).clip(lower=signal_config.min_stop_loss_pct, upper=signal_config.max_stop_loss_pct)
    merged["stop_loss_pct"] = np.where(structure_stop.notna(), np.minimum(atr_stop, structure_stop), atr_stop)
    merged["stop_loss_pct"] = (
        pd.Series(merged["stop_loss_pct"], index=merged.index)
        .clip(lower=signal_config.min_stop_loss_pct, upper=signal_config.max_stop_loss_pct)
        .fillna(0.02)
    )
    tp_scale = 1.0 + (merged["signal_confidence"] - 0.5).clip(lower=-0.2, upper=0.3)
    merged["take_profit_pct"] = (merged["stop_loss_pct"] * signal_config.reward_multiple * tp_scale).clip(
        lower=signal_config.min_stop_loss_pct * 2.0,
        upper=signal_config.max_stop_loss_pct * 3.0,
    )

    seeded_win_rate = (
        risk_config.default_win_rate + (merged["signal_confidence"] - 0.5) * 0.20
    ).clip(lower=0.35, upper=0.70)
    seeded_payoff = (
        risk_config.default_payoff_ratio + merged["signal_confidence"] * 0.90
    ).clip(lower=1.2, upper=3.0)
    merged["kelly_fraction"] = [
        kelly_fraction(
            win_rate=float(win_rate),
            payoff_ratio=float(payoff),
            floor=risk_config.kelly_floor,
            cap=risk_config.kelly_cap,
        )
        for win_rate, payoff in zip(seeded_win_rate, seeded_payoff)
    ]
    return merged


def apply_qlib_score_filter(
    signal_frame: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    confidence_weight: float = 0.35,
    risk_config: RiskConfig | None = None,
) -> pd.DataFrame:
    risk_config = risk_config or RiskConfig()

    output = signal_frame.copy()
    output["heuristic_entry_long_signal"] = output["entry_long_signal"].astype(int)
    output["heuristic_exit_long_signal"] = output["exit_long_signal"].astype(int)
    output["qlib_score"] = pd.to_numeric(output.get("qlib_score", 0.0), errors="coerce")
    output["qlib_score_rank"] = pd.to_numeric(output.get("qlib_score_rank", 0.5), errors="coerce").fillna(0.5)
    output["qlib_score_rank"] = output["qlib_score_rank"].clip(lower=0.0, upper=1.0)

    base_confidence = pd.to_numeric(output.get("signal_confidence", 0.0), errors="coerce").fillna(0.0)
    output["signal_confidence"] = (
        base_confidence * (1.0 - confidence_weight) + output["qlib_score_rank"] * confidence_weight
    ).clip(lower=0.0, upper=1.0)

    output["entry_long_signal"] = (
        (output["heuristic_entry_long_signal"] == 1)
        & output["qlib_score"].notna()
        & (output["qlib_score"] >= entry_threshold)
    ).astype(int)
    output["exit_long_signal"] = (
        (output["heuristic_exit_long_signal"] == 1)
        | (output["qlib_score"].notna() & (output["qlib_score"] <= exit_threshold))
    ).astype(int)

    kelly_scale = (0.75 + output["qlib_score_rank"] * 0.5).clip(lower=0.75, upper=1.25)
    output["kelly_fraction"] = (
        pd.to_numeric(output.get("kelly_fraction", 0.0), errors="coerce").fillna(risk_config.kelly_floor) * kelly_scale
    ).clip(lower=risk_config.kelly_floor, upper=risk_config.kelly_cap)
    output["qlib_entry_threshold"] = float(entry_threshold)
    output["qlib_exit_threshold"] = float(exit_threshold)
    return output


def apply_practical_signal_overrides(
    signal_frame: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    confidence_threshold: float = 0.55,
    exit_confidence_threshold: float = 0.35,
    min_confirmation_count: int = 2,
    require_wyckoff: bool = True,
    require_trend: bool = True,
    qlib_rank_threshold: float = 0.60,
    qlib_exit_rank_threshold: float = 0.45,
    confidence_weight: float = 0.40,
    stop_loss_scale: float = 1.0,
    take_profit_scale: float = 1.0,
    kelly_scale: float = 1.0,
    max_leverage: float = 3.0,
    initial_entry_fraction: float = 0.35,
    layer_stake_fraction: float = 0.20,
    max_entry_layers: int = 3,
    reduce_stake_fraction: float = 0.35,
    partial_take_profit_scale: float = 0.55,
    enable_range_reversion: bool = True,
    range_entry_zone: float = 0.30,
    range_exit_zone: float = 0.72,
    range_qlib_rank_threshold: float = 0.45,
    range_position_scale: float = 0.75,
    range_leverage_cap: float = 2.5,
    range_profit_capture: float = 0.85,
    risk_config: RiskConfig | None = None,
) -> pd.DataFrame:
    risk_config = risk_config or RiskConfig()

    output = signal_frame.copy()
    output["qlib_score"] = pd.to_numeric(output.get("qlib_score", 0.0), errors="coerce")
    output["qlib_score_rank"] = pd.to_numeric(output.get("qlib_score_rank", 0.5), errors="coerce").fillna(0.5)
    output["qlib_score_rank"] = output["qlib_score_rank"].clip(lower=0.0, upper=1.0)

    raw_confidence = pd.to_numeric(output.get("signal_confidence", 0.0), errors="coerce").fillna(0.0)
    output["raw_signal_confidence"] = raw_confidence
    output["signal_confidence"] = (
        raw_confidence * (1.0 - confidence_weight) + output["qlib_score_rank"] * confidence_weight
    ).clip(lower=0.0, upper=1.0)
    for column, series in _range_reversion_components(output).items():
        output[column] = series

    confirmation_count = (
        _binary_series(output, "brooks_bull_signal")
        + _binary_series(output, "chan_long_confirmation")
        + _binary_series(output, "simons_breakout_long")
    )
    output["confirmation_count"] = confirmation_count

    structural_filter = pd.Series(True, index=output.index)
    if require_trend:
        structural_filter &= _binary_series(output, "trend_filter_long") == 1
    if require_wyckoff:
        structural_filter &= _binary_series(output, "wyckoff_long_context") == 1

    trend_entry_gate = (
        structural_filter
        & (confirmation_count >= int(min_confirmation_count))
        & (output["signal_confidence"] >= float(confidence_threshold))
        & output["qlib_score"].notna()
        & (output["qlib_score"] >= float(entry_threshold))
        & (output["qlib_score_rank"] >= float(qlib_rank_threshold))
    )
    trend_exit_gate = (
        _binary_series(output, "heuristic_exit_long_signal", default=-1).eq(1)
        | _binary_series(output, "exit_long_signal").eq(1)
        | (output["qlib_score"].notna() & (output["qlib_score"] <= float(exit_threshold)))
        | (output["qlib_score_rank"] <= float(qlib_exit_rank_threshold))
        | ((_numeric_series(output, "returns_3", 0.0) < 0) & (output["signal_confidence"] <= float(exit_confidence_threshold)))
    )

    range_confidence_floor = max(float(confidence_threshold) - 0.12, 0.22)
    safe_range_entry_zone = min(max(float(range_entry_zone), 0.10), 0.50)
    safe_range_exit_zone = min(max(float(range_exit_zone), 0.50), 0.95)
    safe_range_rank_threshold = min(max(float(range_qlib_rank_threshold), 0.20), 0.80)
    range_regime = _binary_series(output, "range_regime").eq(1)
    support_position = _numeric_series(output, "range_support_position", 0.5)
    resistance_position = _numeric_series(output, "range_resistance_position", 0.5)
    bar_close_strength = _numeric_series(output, "close_location", 0.5)
    range_rebound_score = _numeric_series(output, "range_rebound_score", 0.0)
    accumulation_context = (
        _binary_series(output, "wyckoff_accumulation").eq(1)
        | _binary_series(output, "wyckoff_spring").eq(1)
        | (bar_close_strength >= 0.42)
    )
    range_entry_gate = (
        bool(enable_range_reversion)
        & range_regime
        & (support_position <= safe_range_entry_zone)
        & (output["signal_confidence"] >= range_confidence_floor)
        & (output["qlib_score_rank"] >= safe_range_rank_threshold)
        & accumulation_context
        & (_binary_series(output, "wyckoff_upthrust") == 0)
        & (_numeric_series(output, "relative_volume", 1.0).between(0.55, 2.25))
    )
    range_exit_gate = (
        bool(enable_range_reversion)
        & range_regime
        & (
            (support_position >= safe_range_exit_zone)
            | _binary_series(output, "wyckoff_upthrust").eq(1)
            | (output["qlib_score_rank"] <= max(safe_range_rank_threshold - 0.20, 0.18))
            | (
                (_numeric_series(output, "returns_1", 0.0) > 0)
                & (bar_close_strength < 0.45)
                & (support_position >= max(safe_range_exit_zone - 0.10, 0.55))
            )
        )
    )

    output["heuristic_entry_long_signal"] = _binary_series(output, "entry_long_signal")
    output["heuristic_exit_long_signal"] = _binary_series(output, "exit_long_signal")
    output["trend_entry_long_signal"] = trend_entry_gate.astype(int)
    output["trend_exit_long_signal"] = trend_exit_gate.astype(int)
    output["range_entry_long_signal"] = range_entry_gate.astype(int)
    output["range_exit_long_signal"] = range_exit_gate.astype(int)
    output["entry_long_signal"] = (trend_entry_gate | range_entry_gate).astype(int)
    output["exit_long_signal"] = (trend_exit_gate | range_exit_gate).astype(int)

    base_stop = pd.to_numeric(output.get("stop_loss_pct", 0.02), errors="coerce").fillna(0.02)
    base_take_profit = pd.to_numeric(output.get("take_profit_pct", 0.04), errors="coerce").fillna(0.04)
    output["stop_loss_pct"] = (base_stop * stop_loss_scale).clip(lower=0.004, upper=0.08)
    output["take_profit_pct"] = (base_take_profit * take_profit_scale).clip(lower=0.01, upper=0.25)
    output["kelly_fraction"] = (
        pd.to_numeric(output.get("kelly_fraction", 0.0), errors="coerce").fillna(0.0) * kelly_scale
    ).clip(lower=risk_config.kelly_floor, upper=risk_config.kelly_cap)

    confirmation_norm = (output["confirmation_count"] / 3.0).clip(lower=0.0, upper=1.0)
    trend_strength = _numeric_series(output, "trend_strength", 0.0).clip(lower=-2.0, upper=2.0)
    trend_strength_norm = ((trend_strength + 0.5) / 2.5).clip(lower=0.0, upper=1.0)
    relative_volume = _numeric_series(output, "relative_volume", 1.0).clip(lower=0.0, upper=3.0)
    relative_volume_norm = (relative_volume / 3.0).clip(lower=0.0, upper=1.0)
    atr_pct = _numeric_series(output, "atr_pct", 0.02).clip(lower=0.004, upper=0.08)

    trend_execution_quality = (
        output["signal_confidence"] * 0.40
        + output["qlib_score_rank"] * 0.35
        + confirmation_norm * 0.15
        + trend_strength_norm * 0.10
    ).clip(lower=0.0, upper=1.0)
    range_execution_quality = (
        output["signal_confidence"] * 0.30
        + output["qlib_score_rank"] * 0.25
        + range_rebound_score * 0.35
        + (1.0 - trend_strength_norm) * 0.10
    ).clip(lower=0.0, upper=1.0)
    execution_quality = pd.Series(
        np.where(range_regime, np.maximum(range_execution_quality, trend_execution_quality * 0.85), trend_execution_quality),
        index=output.index,
        dtype="float64",
    )

    range_stop_loss = (
        (_numeric_series(output, "range_distance_to_support_pct", 0.02) + atr_pct * 0.25)
        .clip(lower=0.004, upper=0.03)
    )
    range_take_profit = (
        (_numeric_series(output, "range_distance_to_resistance_pct", 0.02) * float(range_profit_capture))
        .clip(lower=0.008, upper=0.10)
    )
    output["stop_loss_pct"] = pd.Series(
        np.where(range_regime, np.minimum(output["stop_loss_pct"], range_stop_loss), output["stop_loss_pct"]),
        index=output.index,
        dtype="float64",
    ).clip(lower=0.004, upper=0.08)
    output["take_profit_pct"] = pd.Series(
        np.where(
            range_regime,
            np.maximum(np.minimum(output["take_profit_pct"], range_take_profit), output["stop_loss_pct"] * 1.15),
            output["take_profit_pct"],
        ),
        index=output.index,
        dtype="float64",
    ).clip(lower=0.008, upper=0.25)
    volatility_scale = (0.02 / atr_pct.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    volatility_scale = volatility_scale.clip(lower=0.65, upper=1.25)
    leverage_span = max(float(max_leverage) - 1.0, 0.0)
    leverage_core = ((execution_quality - 0.45) / 0.55).clip(lower=0.0, upper=1.0)
    output["target_leverage"] = (
        1.0 + leverage_span * leverage_core * volatility_scale
    ).clip(lower=1.0, upper=max(float(max_leverage), 1.0))
    output["target_leverage"] = pd.Series(
        np.where(range_regime, np.minimum(output["target_leverage"], max(float(range_leverage_cap), 1.0)), output["target_leverage"]),
        index=output.index,
        dtype="float64",
    ).clip(lower=1.0, upper=max(float(max_leverage), float(range_leverage_cap), 1.0))

    output["entry_stake_fraction"] = (
        float(initial_entry_fraction) + (execution_quality - 0.5).clip(lower=0.0, upper=0.4) * 0.30
    ).clip(lower=0.18, upper=0.65)
    output["entry_stake_fraction"] = pd.Series(
        np.where(range_regime, output["entry_stake_fraction"] * float(range_position_scale), output["entry_stake_fraction"]),
        index=output.index,
        dtype="float64",
    ).clip(lower=0.12, upper=0.65)
    output["layer_stake_fraction"] = (
        float(layer_stake_fraction)
        + confirmation_norm * 0.10
        + (output["qlib_score_rank"] - 0.5).clip(lower=0.0, upper=0.4) * 0.15
    ).clip(lower=0.10, upper=0.50)
    output["layer_stake_fraction"] = pd.Series(
        np.where(range_regime, output["layer_stake_fraction"] * min(float(range_position_scale), 0.80), output["layer_stake_fraction"]),
        index=output.index,
        dtype="float64",
    ).clip(lower=0.08, upper=0.50)
    output["reduce_stake_fraction"] = (
        float(reduce_stake_fraction) + (1.0 - execution_quality) * 0.10
    ).clip(lower=0.20, upper=0.60)
    output["reduce_stake_fraction"] = pd.Series(
        np.where(range_regime, np.maximum(output["reduce_stake_fraction"], 0.45), output["reduce_stake_fraction"]),
        index=output.index,
        dtype="float64",
    ).clip(lower=0.20, upper=0.70)
    output["partial_take_profit_pct"] = (
        output["take_profit_pct"] * float(partial_take_profit_scale)
    ).clip(lower=0.008, upper=0.12)
    output["max_entry_layers"] = pd.Series(
        np.where(
            execution_quality >= 0.82,
            max(int(max_entry_layers), 3),
            np.where(execution_quality >= 0.62, max(int(max_entry_layers) - 1, 2), 1),
        ),
        index=output.index,
    ).astype(int)
    output["max_entry_layers"] = pd.Series(
        np.where(range_regime, np.minimum(output["max_entry_layers"], 2), output["max_entry_layers"]),
        index=output.index,
    ).astype(int)

    pullback_reload = (
        (_numeric_series(output, "returns_1", 0.0) < 0)
        & (_numeric_series(output, "close", 0.0) > _numeric_series(output, "ema_fast", 0.0))
        & (_numeric_series(output, "returns_3", 0.0) > -0.02)
    )
    breakout_reload = (
        _binary_series(output, "simons_breakout_long").eq(1)
        & (_numeric_series(output, "returns_1", 0.0) > 0)
    )
    trend_add_signal = (
        structural_filter
        & (output["entry_long_signal"] == 0)
        & (output["signal_confidence"] >= float(confidence_threshold) + 0.05)
        & (output["qlib_score_rank"] >= float(qlib_rank_threshold))
        & (pullback_reload | breakout_reload)
    )
    range_add_signal = (
        bool(enable_range_reversion)
        & range_regime
        & (output["entry_long_signal"] == 0)
        & (support_position <= max(safe_range_entry_zone * 0.85, 0.18))
        & (output["signal_confidence"] >= range_confidence_floor)
        & (output["qlib_score_rank"] >= safe_range_rank_threshold)
        & (_numeric_series(output, "returns_1", 0.0) <= 0)
    )
    output["add_position_signal"] = (trend_add_signal | range_add_signal).astype(int)

    trend_reduce_signal = (
        (output["exit_long_signal"] == 0)
        & (
            (output["signal_confidence"] <= max(float(exit_confidence_threshold) + 0.05, 0.35))
            | (output["qlib_score_rank"] <= max(float(qlib_exit_rank_threshold) + 0.08, 0.35))
            | (
                (_numeric_series(output, "close", 0.0) < _numeric_series(output, "ema_fast", 0.0))
                & (_numeric_series(output, "returns_1", 0.0) < 0)
            )
        )
    )
    range_reduce_signal = (
        bool(enable_range_reversion)
        & range_regime
        & (output["exit_long_signal"] == 0)
        & (
            (support_position >= max(safe_range_exit_zone - 0.12, 0.55))
            | (_numeric_series(output, "returns_1", 0.0) > 0.01)
            | (_binary_series(output, "wyckoff_upthrust") == 1)
        )
    )
    output["reduce_position_signal"] = (trend_reduce_signal | range_reduce_signal).astype(int)

    trend_rotation_score = (
        output["signal_confidence"] * 0.45
        + output["qlib_score_rank"] * 0.35
        + trend_strength_norm * 0.10
        + relative_volume_norm * 0.10
    ).clip(lower=0.0, upper=1.0)
    range_rotation_score = (
        output["signal_confidence"] * 0.28
        + output["qlib_score_rank"] * 0.22
        + range_rebound_score * 0.35
        + (1.0 - trend_strength_norm) * 0.10
        + relative_volume_norm * 0.05
    ).clip(lower=0.0, upper=1.0)
    output["rotation_score"] = pd.Series(
        np.where(range_regime, range_rotation_score, trend_rotation_score),
        index=output.index,
        dtype="float64",
    ).clip(lower=0.0, upper=1.0)
    output["rotation_rank"] = 0
    output["rotation_weight"] = 0.0
    output["rotation_entry_long_signal"] = output["entry_long_signal"].astype(int)

    output["qlib_entry_threshold"] = float(entry_threshold)
    output["qlib_exit_threshold"] = float(exit_threshold)
    return output


def apply_rotation_overlay(
    symbol_frames: dict[str, pd.DataFrame],
    top_n: int = 2,
    min_rotation_score: float = 0.55,
) -> dict[str, pd.DataFrame]:
    if not symbol_frames:
        return {}

    safe_top_n = max(int(top_n), 1)
    min_score = float(min_rotation_score)
    prepared_frames: dict[str, pd.DataFrame] = {}
    ranking_blocks: list[pd.DataFrame] = []

    for symbol, frame in symbol_frames.items():
        local = frame.copy()
        local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
        local["symbol"] = symbol
        local["rotation_score"] = _numeric_series(local, "rotation_score", 0.0).clip(lower=0.0, upper=1.0)
        local["entry_long_signal"] = _binary_series(local, "entry_long_signal")
        ranking_blocks.append(local[["timestamp", "symbol", "rotation_score", "entry_long_signal"]].copy())
        prepared_frames[symbol] = local

    ranking_frame = pd.concat(ranking_blocks, ignore_index=True)
    ranking_frame["rotation_eligible"] = (
        (ranking_frame["entry_long_signal"] == 1)
        & (ranking_frame["rotation_score"] >= min_score)
    ).astype(int)
    ranking_frame["rotation_rank"] = safe_top_n + 99
    eligible_mask = ranking_frame["rotation_eligible"] == 1
    if eligible_mask.any():
        ranking_frame.loc[eligible_mask, "rotation_rank"] = (
            ranking_frame.loc[eligible_mask]
            .groupby("timestamp")["rotation_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

    selected_mask = eligible_mask & (ranking_frame["rotation_rank"] <= safe_top_n)
    ranking_frame["rotation_weight"] = 0.0
    if selected_mask.any():
        selected_scores = ranking_frame.loc[selected_mask, "rotation_score"].clip(lower=min_score)
        selected_sum = selected_scores.groupby(ranking_frame.loc[selected_mask, "timestamp"]).transform("sum")
        ranking_frame.loc[selected_mask, "rotation_weight"] = (selected_scores / selected_sum).fillna(0.0)

    ranking_frame["rotation_entry_long_signal"] = selected_mask.astype(int)
    overlay_lookup = ranking_frame[
        ["timestamp", "symbol", "rotation_rank", "rotation_weight", "rotation_entry_long_signal"]
    ]

    adjusted_frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in prepared_frames.items():
        overlay = overlay_lookup.loc[overlay_lookup["symbol"] == symbol].drop(columns=["symbol"])
        local = frame.merge(overlay, on="timestamp", how="left", suffixes=("", "_overlay"))
        local["rotation_rank"] = pd.to_numeric(
            local.get("rotation_rank_overlay", local.get("rotation_rank", 0)),
            errors="coerce",
        ).fillna(safe_top_n + 99).astype(int)
        local["rotation_weight"] = pd.to_numeric(
            local.get("rotation_weight_overlay", local.get("rotation_weight", 0.0)),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0, upper=1.0)
        local["rotation_entry_long_signal"] = _binary_series(
            local,
            "rotation_entry_long_signal_overlay"
            if "rotation_entry_long_signal_overlay" in local.columns
            else "rotation_entry_long_signal",
            default=0,
        )
        local["entry_stake_fraction"] = (
            _numeric_series(local, "entry_stake_fraction", 0.25) * local["rotation_weight"]
        ).clip(lower=0.0, upper=0.80)
        local["layer_stake_fraction"] = (
            _numeric_series(local, "layer_stake_fraction", 0.15) * local["rotation_weight"]
        ).clip(lower=0.0, upper=0.60)
        local["target_leverage"] = (
            _numeric_series(local, "target_leverage", 1.0) * (0.80 + local["rotation_weight"] * 0.20)
        ).clip(lower=1.0)
        local["reduce_position_signal"] = (
            _binary_series(local, "reduce_position_signal")
            | (
                (_binary_series(local, "entry_long_signal") == 1)
                & (local["rotation_entry_long_signal"] == 0)
                & (_numeric_series(local, "rotation_score", 0.0) < min_score + 0.05)
            )
        ).astype(int)
        local["entry_long_signal"] = local["rotation_entry_long_signal"].astype(int)
        local["add_position_signal"] = (
            _binary_series(local, "add_position_signal") & (local["rotation_entry_long_signal"] == 1)
        ).astype(int)
        local = local.drop(
            columns=[
                column
                for column in [
                    "rotation_rank_overlay",
                    "rotation_weight_overlay",
                    "rotation_entry_long_signal_overlay",
                ]
                if column in local.columns
            ]
        )
        adjusted_frames[symbol] = local

    return adjusted_frames
