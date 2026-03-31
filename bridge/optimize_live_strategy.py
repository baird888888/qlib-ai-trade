from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from quant_trading.backtest.engine import run_backtest, run_portfolio_backtest
from quant_trading.config import BacktestConfig, RiskConfig
from quant_trading.signals.multi_timeframe import apply_practical_signal_overrides, apply_rotation_overlay
from bridge.rank_utils import reference_scores_with_fallback, score_percentiles

RUNTIME_ROOT = PROJECT_ROOT / "bridge" / "runtime"
FEATURE_DIR = RUNTIME_ROOT / "features"
REPORT_DIR = RUNTIME_ROOT / "reports"

SUMMARY_COLUMNS = [
    "return_pct",
    "return_ann_pct",
    "cagr_pct",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown_pct",
    "win_rate_pct",
    "payoff_ratio",
    "trades",
]


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp


def _slug(symbol: str) -> str:
    return symbol.lower().replace("-", "_").replace("/", "_")


def _load_training_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def _safe_float(value: object) -> float:
    try:
        casted = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(casted) or np.isinf(casted):
        return 0.0
    return casted


def _slice_frame(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    mask = (timestamp >= start) & (timestamp <= end)
    return frame.loc[mask].reset_index(drop=True)


def _load_feature_frames(symbols: list[str]) -> dict[str, pd.DataFrame]:
    return {
        symbol: pd.read_parquet(FEATURE_DIR / f"{_slug(symbol)}_qlib_signals.parquet")
        for symbol in symbols
    }


def _score_quantiles(scores: pd.Series, entry_quantile: float, exit_quantile: float) -> tuple[float, float]:
    clean_scores = pd.to_numeric(scores, errors="coerce").dropna()
    if clean_scores.empty:
        return 0.0, 0.0
    return float(clean_scores.quantile(entry_quantile)), float(clean_scores.quantile(exit_quantile))


def _global_validation_scores(frames: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    blocks = [
        pd.to_numeric(_slice_frame(frame, start, end)["qlib_score"], errors="coerce").dropna()
        for frame in frames.values()
    ]
    if not blocks:
        return pd.Series(dtype="float64")
    return pd.concat(blocks, ignore_index=True)


def _symbol_thresholds(
    validation_frame: pd.DataFrame,
    entry_quantile: float,
    exit_quantile: float,
    fallback_scores: pd.Series,
) -> tuple[float, float]:
    scores = pd.to_numeric(validation_frame["qlib_score"], errors="coerce").dropna()
    if scores.empty:
        return _score_quantiles(fallback_scores, entry_quantile, exit_quantile)
    return _score_quantiles(scores, entry_quantile, exit_quantile)


def _base_parameter_space() -> dict[str, list]:
    return {
        "entry_quantile": [0.45, 0.55, 0.65, 0.75, 0.85],
        "exit_quantile": [0.15, 0.25, 0.35, 0.45, 0.55],
        "confidence_threshold": [0.30, 0.35, 0.45, 0.55, 0.65],
        "exit_confidence_threshold": [0.20, 0.30, 0.40, 0.50],
        "min_confirmation_count": [1, 2],
        "enable_range_reversion": [False, True],
        "qlib_rank_threshold": [0.45, 0.55, 0.65, 0.75],
        "qlib_exit_rank_threshold": [0.15, 0.25, 0.35, 0.45],
        "confidence_weight": [0.30, 0.40, 0.50],
        "stop_loss_scale": [0.75, 0.90, 1.0, 1.15],
        "take_profit_scale": [1.0, 1.2, 1.4],
        "kelly_scale": [0.60, 0.75, 0.90, 1.0, 1.10, 1.25],
        "max_leverage": [2.0, 3.0, 4.0, 5.0, 6.0],
        "initial_entry_fraction": [0.25, 0.35, 0.45, 0.55, 0.65],
        "layer_stake_fraction": [0.10, 0.15, 0.20, 0.30],
        "max_entry_layers": [1, 2, 3],
        "reduce_stake_fraction": [0.25, 0.35, 0.45, 0.55],
        "partial_take_profit_scale": [0.35, 0.45, 0.55, 0.65],
        "rotation_top_n": [1, 2, 3],
        "rotation_min_score": [0.45, 0.55, 0.65, 0.75],
        "range_entry_zone": [0.20, 0.30, 0.40],
        "range_exit_zone": [0.60, 0.72, 0.82],
        "range_qlib_rank_threshold": [0.35, 0.45, 0.55],
        "range_position_scale": [0.60, 0.75, 0.90],
        "range_leverage_cap": [2.0, 3.0, 4.0],
        "range_profit_capture": [0.65, 0.80, 0.95],
    }


def _nearest_option_index(options: list, target: object) -> int:
    if not options:
        return 0
    try:
        return options.index(target)
    except ValueError:
        pass

    first = options[0]
    if isinstance(first, (int, float)) and isinstance(target, (int, float)):
        return min(range(len(options)), key=lambda idx: abs(float(options[idx]) - float(target)))
    return 0


def _parameter_space(anchor_params: dict | None = None, local_radius: int = 1) -> dict[str, list]:
    base_space = _base_parameter_space()
    if not anchor_params:
        return base_space

    refined_space: dict[str, list] = {}
    for key, options in base_space.items():
        if key not in anchor_params:
            refined_space[key] = options
            continue
        anchor_index = _nearest_option_index(options, anchor_params[key])
        start = max(anchor_index - max(int(local_radius), 0), 0)
        stop = min(anchor_index + max(int(local_radius), 0) + 1, len(options))
        refined_space[key] = options[start:stop]
    return refined_space


def _apply_fixed_params(space: dict[str, list], fixed_params: dict[str, object] | None) -> dict[str, list]:
    if not fixed_params:
        return space
    constrained = {key: list(values) for key, values in space.items()}
    for key, value in fixed_params.items():
        constrained[key] = [value]
    return constrained


def _load_json_arg(raw_json: str | None, raw_file: str | None) -> dict[str, object] | None:
    if raw_json:
        return json.loads(raw_json)
    if raw_file:
        path = Path(raw_file)
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _sample_parameter_sets(max_trials: int, seed: int, space: dict[str, list]) -> list[dict]:
    keys = list(space)
    rng = np.random.default_rng(seed)
    total_combinations = 1
    for key in keys:
        total_combinations *= len(space[key])

    if total_combinations <= max_trials:
        return [dict(zip(keys, values)) for values in product(*(space[key] for key in keys))]

    selected: list[dict] = []
    seen: set[tuple] = set()
    while len(selected) < max_trials:
        values = tuple(space[key][int(rng.integers(0, len(space[key])))] for key in keys)
        if values in seen:
            continue
        seen.add(values)
        selected.append(dict(zip(keys, values)))
    return selected


def _empty_summary() -> dict[str, float | int]:
    return {
        "return_pct": 0.0,
        "return_ann_pct": 0.0,
        "cagr_pct": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "payoff_ratio": 0.0,
        "trades": 0,
    }


def _coerce_metric_frame(frame: pd.DataFrame, extra_numeric: list[str] | None = None) -> pd.DataFrame:
    numeric_columns = list(SUMMARY_COLUMNS)
    if extra_numeric:
        numeric_columns.extend(extra_numeric)
    coerced = frame.copy()
    for column in numeric_columns:
        if column not in coerced.columns:
            continue
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return coerced


def _aggregate_single_period_objective(
    rows: list[dict],
    portfolio_summary: dict[str, float | int],
    stability_metrics: dict[str, float | int] | None = None,
) -> dict[str, float]:
    frame = _coerce_metric_frame(pd.DataFrame(rows))
    if frame.empty:
        frame = pd.DataFrame([{**{"symbol": "N/A"}, **_empty_summary()}])
    avg_ann = float(frame["return_ann_pct"].mean())
    avg_cagr = float(frame["cagr_pct"].mean())
    worst_dd = float(frame["max_drawdown_pct"].max())
    total_trades = int(frame["trades"].sum())
    positive_symbols = int((frame["return_pct"] > 0).sum())
    avg_payoff = float(frame["payoff_ratio"].mean())
    portfolio_ann = _safe_float(portfolio_summary.get("return_ann_pct", 0.0))
    portfolio_cagr = _safe_float(portfolio_summary.get("cagr_pct", 0.0))
    portfolio_dd = _safe_float(portfolio_summary.get("max_drawdown_pct", 0.0))
    portfolio_trades = int(_safe_float(portfolio_summary.get("trades", 0)))

    objective = portfolio_ann * 1.2 + portfolio_cagr * 0.7 + positive_symbols * 2.0 + avg_payoff
    if portfolio_dd > 30.0:
        objective -= (portfolio_dd - 30.0) * 25.0
    if portfolio_trades < 50:
        objective -= (50 - portfolio_trades) * 0.8
    output = {
        "objective": float(objective),
        "avg_ann_pct": avg_ann,
        "avg_cagr_pct": avg_cagr,
        "worst_dd_pct": worst_dd,
        "total_trades": total_trades,
        "portfolio_return_ann_pct": portfolio_ann,
        "portfolio_cagr_pct": portfolio_cagr,
        "portfolio_max_drawdown_pct": portfolio_dd,
        "portfolio_trades": portfolio_trades,
        "positive_symbols": positive_symbols,
        "avg_payoff_ratio": avg_payoff,
    }
    if stability_metrics:
        output.update(stability_metrics)
    return output


def _aggregate_portfolio_summaries(
    portfolios: list[dict],
    stability_metrics: dict[str, float | int] | None = None,
) -> dict[str, float | int]:
    if not portfolios:
        return _empty_summary()
    frame = _coerce_metric_frame(pd.DataFrame(portfolios), extra_numeric=["fold"])
    summary = {
        "return_pct": float(frame["return_pct"].mean()),
        "return_ann_pct": float(frame["return_ann_pct"].mean()),
        "cagr_pct": float(frame["cagr_pct"].mean()),
        "sharpe": float(frame["sharpe"].mean()),
        "sortino": float(frame["sortino"].mean()),
        "calmar": float(frame["calmar"].mean()),
        "max_drawdown_pct": float(frame["max_drawdown_pct"].max()),
        "win_rate_pct": float(frame["win_rate_pct"].mean()),
        "payoff_ratio": float(frame["payoff_ratio"].mean()),
        "trades": int(frame["trades"].sum()),
        "folds": int(len(frame)),
        "positive_fold_ratio": float((frame["return_pct"] > 0).mean()),
    }
    if stability_metrics:
        summary.update(stability_metrics)
    return summary


def _period_return_metrics(equity_curve: pd.DataFrame, frequency: str, label: str) -> dict[str, float | int]:
    if equity_curve.empty:
        return {
            f"{label}_periods": 0,
            f"{label}_positive_ratio": 0.0,
            f"{label}_non_negative_ratio": 0.0,
            f"{label}_avg_return_pct": 0.0,
            f"{label}_median_return_pct": 0.0,
            f"{label}_std_return_pct": 0.0,
            f"{label}_worst_return_pct": 0.0,
            f"{label}_best_return_pct": 0.0,
        }

    curve = equity_curve.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], utc=True)
    curve = curve.dropna(subset=["timestamp"]).sort_values("timestamp")
    if curve.empty:
        return {
            f"{label}_periods": 0,
            f"{label}_positive_ratio": 0.0,
            f"{label}_non_negative_ratio": 0.0,
            f"{label}_avg_return_pct": 0.0,
            f"{label}_median_return_pct": 0.0,
            f"{label}_std_return_pct": 0.0,
            f"{label}_worst_return_pct": 0.0,
            f"{label}_best_return_pct": 0.0,
        }

    equity = pd.to_numeric(curve["equity"], errors="coerce")
    series = pd.Series(equity.to_numpy(), index=curve["timestamp"]).dropna()
    period_equity = series.resample(frequency).last().dropna()
    period_returns = period_equity.pct_change().dropna() * 100.0
    if period_returns.empty:
        return {
            f"{label}_periods": 0,
            f"{label}_positive_ratio": 0.0,
            f"{label}_non_negative_ratio": 0.0,
            f"{label}_avg_return_pct": 0.0,
            f"{label}_median_return_pct": 0.0,
            f"{label}_std_return_pct": 0.0,
            f"{label}_worst_return_pct": 0.0,
            f"{label}_best_return_pct": 0.0,
        }

    return {
        f"{label}_periods": int(len(period_returns)),
        f"{label}_positive_ratio": float((period_returns > 0).mean()),
        f"{label}_non_negative_ratio": float((period_returns >= 0).mean()),
        f"{label}_avg_return_pct": float(period_returns.mean()),
        f"{label}_median_return_pct": float(period_returns.median()),
        f"{label}_std_return_pct": float(period_returns.std(ddof=0)) if len(period_returns) > 1 else 0.0,
        f"{label}_worst_return_pct": float(period_returns.min()),
        f"{label}_best_return_pct": float(period_returns.max()),
    }


def _equity_stability_metrics(equity_curve: pd.DataFrame) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    metrics.update(_period_return_metrics(equity_curve, "W", "weekly"))
    metrics.update(_period_return_metrics(equity_curve, "ME", "monthly"))
    metrics.update(_period_return_metrics(equity_curve, "YE", "yearly"))
    return metrics


def _stitch_fold_equity_curves(
    fold_equity_curves: list[pd.DataFrame],
    initial_cash: float,
) -> pd.DataFrame:
    stitched_parts: list[pd.DataFrame] = []
    running_equity = float(initial_cash)

    for curve in fold_equity_curves:
        if curve is None or curve.empty:
            continue
        local = curve.copy()
        local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
        local = local.sort_values("timestamp").dropna(subset=["timestamp"])
        if local.empty:
            continue
        local_equity = pd.to_numeric(local["equity"], errors="coerce")
        first_equity = float(local_equity.iloc[0]) if not local_equity.empty else 0.0
        if first_equity <= 0:
            continue
        scale = running_equity / first_equity
        local["equity"] = local_equity * scale
        if "free_cash" in local.columns:
            local["free_cash"] = pd.to_numeric(local["free_cash"], errors="coerce") * scale
        if "used_margin" in local.columns:
            local["used_margin"] = pd.to_numeric(local["used_margin"], errors="coerce") * scale
        if "gross_notional" in local.columns:
            local["gross_notional"] = pd.to_numeric(local["gross_notional"], errors="coerce") * scale
        stitched_parts.append(local)
        running_equity = float(pd.to_numeric(local["equity"], errors="coerce").iloc[-1])

    if not stitched_parts:
        return pd.DataFrame()

    stitched = pd.concat(stitched_parts, ignore_index=True)
    stitched["timestamp"] = pd.to_datetime(stitched["timestamp"], utc=True)
    stitched = stitched.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return stitched


def _aggregate_symbol_rows(symbol_rows: list[dict]) -> list[dict]:
    if not symbol_rows:
        return []
    frame = _coerce_metric_frame(pd.DataFrame(symbol_rows), extra_numeric=["fold"])
    positive_fold_counts = (
        frame.assign(positive_fold=(frame["return_pct"] > 0).astype(int))
        .groupby("symbol", as_index=False)["positive_fold"]
        .sum()
        .rename(columns={"positive_fold": "positive_folds"})
    )
    grouped = (
        frame.groupby("symbol", as_index=False)
        .agg(
            return_pct=("return_pct", "mean"),
            return_ann_pct=("return_ann_pct", "mean"),
            cagr_pct=("cagr_pct", "mean"),
            sharpe=("sharpe", "mean"),
            sortino=("sortino", "mean"),
            calmar=("calmar", "mean"),
            max_drawdown_pct=("max_drawdown_pct", "max"),
            win_rate_pct=("win_rate_pct", "mean"),
            payoff_ratio=("payoff_ratio", "mean"),
            trades=("trades", "sum"),
            folds=("fold", "nunique"),
        )
        .merge(positive_fold_counts, on="symbol", how="left")
    )
    grouped["positive_folds"] = grouped["positive_folds"].fillna(0).astype(int)
    grouped["trades"] = grouped["trades"].round().astype(int)
    grouped["folds"] = grouped["folds"].round().astype(int)
    return grouped.to_dict(orient="records")


def _aggregate_walk_forward_objective(
    portfolios: list[dict],
    symbol_rows: list[dict],
    stability_metrics: dict[str, float | int] | None = None,
) -> dict[str, float]:
    if not portfolios:
        empty = _aggregate_single_period_objective([], _empty_summary())
        empty.update(
            {
                "folds": 0,
                "positive_fold_ratio": 0.0,
                "recent_return_ann_pct": 0.0,
                "return_ann_std": 0.0,
                "return_pct_std": 0.0,
                "worst_fold_return_ann_pct": 0.0,
                "avg_drawdown_pct": 0.0,
                "profitable_symbol_ratio": 0.0,
            }
        )
        return empty

    portfolio_frame = _coerce_metric_frame(pd.DataFrame(portfolios), extra_numeric=["fold"])
    symbol_frame = _coerce_metric_frame(pd.DataFrame(symbol_rows), extra_numeric=["fold"])

    mean_ann = float(portfolio_frame["return_ann_pct"].mean())
    median_ann = float(portfolio_frame["return_ann_pct"].median())
    min_ann = float(portfolio_frame["return_ann_pct"].min())
    recent_ann = float(portfolio_frame["return_ann_pct"].iloc[-1])
    mean_cagr = float(portfolio_frame["cagr_pct"].mean())
    worst_dd = float(portfolio_frame["max_drawdown_pct"].max())
    avg_dd = float(portfolio_frame["max_drawdown_pct"].mean())
    total_trades = int(portfolio_frame["trades"].sum())
    positive_fold_ratio = float((portfolio_frame["return_pct"] > 0).mean())
    avg_payoff = float(portfolio_frame["payoff_ratio"].mean())
    ann_std = float(portfolio_frame["return_ann_pct"].std(ddof=0)) if len(portfolio_frame) > 1 else 0.0
    return_std = float(portfolio_frame["return_pct"].std(ddof=0)) if len(portfolio_frame) > 1 else 0.0
    if not symbol_frame.empty:
        symbol_mean_returns = symbol_frame.groupby("symbol", as_index=False)["return_pct"].mean()
        positive_symbol_count = int((symbol_mean_returns["return_pct"] > 0).sum())
        profitable_symbol_ratio = float((symbol_mean_returns["return_pct"] > 0).mean())
    else:
        positive_symbol_count = 0
        profitable_symbol_ratio = 0.0

    clipped_return_fraction = (portfolio_frame["return_pct"] / 100.0).clip(lower=-0.95)
    growth_scores = np.log1p(clipped_return_fraction)
    mean_growth = float(growth_scores.mean())
    median_growth = float(growth_scores.median())
    min_growth = float(growth_scores.min())
    recent_growth = float(growth_scores.iloc[-1])
    growth_std = float(growth_scores.std(ddof=0)) if len(growth_scores) > 1 else 0.0

    stability_metrics = stability_metrics or {}
    weekly_positive_ratio = _safe_float(stability_metrics.get("weekly_positive_ratio", 0.0))
    monthly_positive_ratio = _safe_float(stability_metrics.get("monthly_positive_ratio", 0.0))
    yearly_positive_ratio = _safe_float(stability_metrics.get("yearly_positive_ratio", 0.0))
    weekly_worst_return_pct = _safe_float(stability_metrics.get("weekly_worst_return_pct", 0.0))
    monthly_worst_return_pct = _safe_float(stability_metrics.get("monthly_worst_return_pct", 0.0))
    yearly_worst_return_pct = _safe_float(stability_metrics.get("yearly_worst_return_pct", 0.0))

    objective = (
        mean_growth * 120.0
        + median_growth * 85.0
        + min_growth * 45.0
        + recent_growth * 30.0
        + avg_payoff * 1.50
        + positive_fold_ratio * 10.0
        + weekly_positive_ratio * 10.0
        + monthly_positive_ratio * 18.0
        + yearly_positive_ratio * 16.0
    )
    objective -= growth_std * 35.0
    objective -= ann_std * 0.12
    objective -= return_std * 0.08
    objective -= max(avg_dd - 12.0, 0.0) * 0.65
    objective -= max(-weekly_worst_return_pct, 0.0) * 0.08
    objective -= max(-monthly_worst_return_pct, 0.0) * 0.18
    objective -= max(-yearly_worst_return_pct, 0.0) * 0.35

    if worst_dd > 30.0:
        objective -= (worst_dd - 30.0) * 35.0
    if positive_fold_ratio < 0.65:
        objective -= (0.65 - positive_fold_ratio) * 30.0
    if weekly_positive_ratio < 0.55:
        objective -= (0.55 - weekly_positive_ratio) * 25.0
    if monthly_positive_ratio < 0.60:
        objective -= (0.60 - monthly_positive_ratio) * 35.0
    if yearly_positive_ratio < 0.80:
        objective -= (0.80 - yearly_positive_ratio) * 40.0

    minimum_trade_budget = max(int(len(portfolio_frame) * 18), 1)
    if total_trades < minimum_trade_budget:
        objective -= (minimum_trade_budget - total_trades) * 0.30

    output = {
        "objective": float(objective),
        "avg_ann_pct": mean_ann,
        "avg_cagr_pct": mean_cagr,
        "worst_dd_pct": worst_dd,
        "total_trades": total_trades,
        "portfolio_return_ann_pct": mean_ann,
        "portfolio_cagr_pct": mean_cagr,
        "portfolio_max_drawdown_pct": worst_dd,
        "portfolio_trades": total_trades,
        "positive_symbols": positive_symbol_count,
        "avg_payoff_ratio": avg_payoff,
        "folds": int(len(portfolio_frame)),
        "positive_fold_ratio": positive_fold_ratio,
        "recent_return_ann_pct": recent_ann,
        "return_ann_std": ann_std,
        "return_pct_std": return_std,
        "worst_fold_return_ann_pct": min_ann,
        "avg_drawdown_pct": avg_dd,
        "profitable_symbol_ratio": profitable_symbol_ratio,
        "avg_log_growth": mean_growth,
        "log_growth_std": growth_std,
    }
    if stability_metrics:
        output.update(stability_metrics)
    return output


def _build_holdout_split(
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    holdout_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if holdout_days <= 0:
        raise ValueError("--holdout-days must be positive.")
    holdout_start = test_end - pd.Timedelta(days=holdout_days)
    search_end = holdout_start - pd.Timedelta(seconds=1)
    if holdout_start <= test_start or search_end <= test_start:
        raise ValueError(
            "The requested holdout window leaves no room for walk-forward search. Reduce --holdout-days."
        )
    return search_end, holdout_start


def _build_walk_forward_windows(
    search_start: pd.Timestamp,
    search_end: pd.Timestamp,
    calibration_window_days: int,
    validation_window_days: int,
    step_days: int,
) -> list[WalkForwardWindow]:
    if calibration_window_days <= 0 or validation_window_days <= 0 or step_days <= 0:
        raise ValueError("Walk-forward window sizes must be positive.")

    epsilon = pd.Timedelta(seconds=1)
    evaluation_span = pd.Timedelta(days=validation_window_days)
    step_span = pd.Timedelta(days=step_days)
    calibration_span = pd.Timedelta(days=calibration_window_days)
    minimum_tail_span = pd.Timedelta(days=max(validation_window_days // 2, 45))

    windows: list[WalkForwardWindow] = []
    evaluation_start = search_start
    while evaluation_start <= search_end:
        nominal_end = evaluation_start + evaluation_span - epsilon
        if nominal_end >= search_end:
            evaluation_end = search_end
        else:
            remaining_after_nominal = search_end - nominal_end
            evaluation_end = search_end if remaining_after_nominal < minimum_tail_span else nominal_end

        calibration_end = evaluation_start - epsilon
        calibration_start = calibration_end - calibration_span + epsilon
        windows.append(
            WalkForwardWindow(
                calibration_start=calibration_start,
                calibration_end=calibration_end,
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
            )
        )

        if evaluation_end >= search_end:
            break
        evaluation_start = evaluation_start + step_span

    if len(windows) < 3:
        raise ValueError(
            "Walk-forward search produced fewer than 3 folds. Increase history or reduce the holdout / window sizes."
        )
    return windows


def _serialize_window(window: WalkForwardWindow) -> dict[str, list[str]]:
    return {
        "calibration": [window.calibration_start.isoformat(), window.calibration_end.isoformat()],
        "evaluation": [window.evaluation_start.isoformat(), window.evaluation_end.isoformat()],
    }


def _build_variant_frames(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    params: dict,
    risk_config: RiskConfig,
    fallback_scores: pd.Series,
) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        full_frame = frames[symbol]
        validation_frame = _slice_frame(full_frame, validation_start, validation_end)
        target_frame = _slice_frame(full_frame, start, end).copy()
        reference_scores = reference_scores_with_fallback(validation_frame.get("qlib_score"), fallback_scores)
        target_frame["qlib_score_rank"] = score_percentiles(target_frame.get("qlib_score"), reference_scores)
        entry_threshold, exit_threshold = _symbol_thresholds(
            validation_frame,
            entry_quantile=params["entry_quantile"],
            exit_quantile=params["exit_quantile"],
            fallback_scores=fallback_scores,
        )
        variants[symbol] = apply_practical_signal_overrides(
            target_frame,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            confidence_threshold=params.get("confidence_threshold", 0.55),
            exit_confidence_threshold=params.get("exit_confidence_threshold", 0.35),
            min_confirmation_count=params.get("min_confirmation_count", 2),
            qlib_rank_threshold=params.get("qlib_rank_threshold", 0.60),
            qlib_exit_rank_threshold=params.get("qlib_exit_rank_threshold", 0.45),
            confidence_weight=params.get("confidence_weight", 0.40),
            stop_loss_scale=params.get("stop_loss_scale", 1.0),
            take_profit_scale=params.get("take_profit_scale", 1.0),
            kelly_scale=params.get("kelly_scale", 1.0),
            max_leverage=params.get("max_leverage", 3.0),
            initial_entry_fraction=params.get("initial_entry_fraction", 0.35),
            layer_stake_fraction=params.get("layer_stake_fraction", 0.20),
            max_entry_layers=params.get("max_entry_layers", 3),
            reduce_stake_fraction=params.get("reduce_stake_fraction", 0.35),
            partial_take_profit_scale=params.get("partial_take_profit_scale", 0.55),
            range_entry_zone=params.get("range_entry_zone", 0.30),
            range_exit_zone=params.get("range_exit_zone", 0.72),
            range_qlib_rank_threshold=params.get("range_qlib_rank_threshold", 0.45),
            range_position_scale=params.get("range_position_scale", 0.75),
            range_leverage_cap=params.get("range_leverage_cap", 2.5),
            range_profit_capture=params.get("range_profit_capture", 0.85),
            risk_config=risk_config,
        )
        variants[symbol]["symbol"] = symbol
    return apply_rotation_overlay(
        variants,
        top_n=params["rotation_top_n"],
        min_rotation_score=params["rotation_min_score"],
    )


def _rows_from_variant_frames(
    variant_frames: dict[str, pd.DataFrame],
    symbols: list[str],
    backtest_config: BacktestConfig,
    risk_config: RiskConfig,
) -> list[dict]:
    rows: list[dict] = []
    local_backtest = BacktestConfig(
        initial_cash=backtest_config.initial_cash,
        spread=backtest_config.spread,
        commission=backtest_config.commission,
        margin=backtest_config.margin,
        exclusive_orders=backtest_config.exclusive_orders,
    )
    for symbol in symbols:
        backtest = run_backtest(variant_frames[symbol], backtest_config=local_backtest, risk_config=risk_config)
        rows.append({"symbol": symbol, **backtest["summary"]})
    return rows


def _slice_frames(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    return {symbol: _slice_frame(frames[symbol], start, end) for symbol in symbols}


def _evaluate_frames(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    backtest_config: BacktestConfig,
    risk_config: RiskConfig,
    include_rows: bool = True,
) -> dict[str, object]:
    rows = (
        _rows_from_variant_frames(
            variant_frames=frames,
            symbols=symbols,
            backtest_config=backtest_config,
            risk_config=risk_config,
        )
        if include_rows
        else []
    )
    portfolio_backtest = run_portfolio_backtest(
        frames,
        backtest_config=backtest_config,
        risk_config=risk_config,
    )
    return {
        "rows": rows,
        "portfolio": portfolio_backtest["summary"],
        "equity_curve": portfolio_backtest["equity_curve"],
        "stability": _equity_stability_metrics(portfolio_backtest["equity_curve"]),
    }


def _current_strategy_rows(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    backtest_config: BacktestConfig,
    risk_config: RiskConfig,
) -> dict[str, object]:
    rows: list[dict] = []
    for symbol in symbols:
        frame = _slice_frame(frames[symbol], start, end)
        backtest = run_backtest(frame, backtest_config=backtest_config, risk_config=risk_config)
        rows.append({"symbol": symbol, **backtest["summary"]})
    portfolio_backtest = run_portfolio_backtest(
        _slice_frames(frames, symbols, start, end),
        backtest_config=backtest_config,
        risk_config=risk_config,
    )
    return {
        "rows": rows,
        "portfolio": portfolio_backtest["summary"],
        "equity_curve": portfolio_backtest["equity_curve"],
        "stability": _equity_stability_metrics(portfolio_backtest["equity_curve"]),
    }


def _evaluate_walk_forward_candidate(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    windows: list[WalkForwardWindow],
    params: dict,
    backtest_config: BacktestConfig,
    risk_config: RiskConfig,
    include_rows: bool = True,
) -> dict[str, object]:
    fold_entries: list[dict] = []
    portfolio_summaries: list[dict] = []
    symbol_rows: list[dict] = []
    fold_equity_curves: list[pd.DataFrame] = []

    for fold_id, window in enumerate(windows, start=1):
        fallback_scores = _global_validation_scores(frames, window.calibration_start, window.calibration_end)
        variant_frames = _build_variant_frames(
            frames=frames,
            symbols=symbols,
            start=window.evaluation_start,
            end=window.evaluation_end,
            validation_start=window.calibration_start,
            validation_end=window.calibration_end,
            params=params,
            risk_config=risk_config,
            fallback_scores=fallback_scores,
        )
        evaluation = _evaluate_frames(
            frames=variant_frames,
            symbols=symbols,
            backtest_config=backtest_config,
            risk_config=risk_config,
            include_rows=include_rows,
        )
        portfolio_summary = evaluation["portfolio"]
        fold_entries.append(
            {
                "fold": fold_id,
                "window": _serialize_window(window),
                "portfolio": portfolio_summary,
                "stability": evaluation["stability"],
                "rows": evaluation["rows"],
            }
        )
        portfolio_summaries.append({"fold": fold_id, **portfolio_summary})
        fold_equity_curves.append(evaluation["equity_curve"])
        if include_rows:
            for row in evaluation["rows"]:
                symbol_rows.append({"fold": fold_id, **row})

    stitched_equity_curve = _stitch_fold_equity_curves(
        fold_equity_curves,
        initial_cash=backtest_config.initial_cash,
    )
    stability = _equity_stability_metrics(stitched_equity_curve)

    return {
        "aggregate": _aggregate_walk_forward_objective(portfolio_summaries, symbol_rows, stability),
        "portfolio": _aggregate_portfolio_summaries(portfolio_summaries, stability),
        "rows": _aggregate_symbol_rows(symbol_rows) if include_rows else [],
        "stability": stability,
        "folds": fold_entries,
    }


def _evaluate_walk_forward_current_strategy(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    windows: list[WalkForwardWindow],
    backtest_config: BacktestConfig,
    risk_config: RiskConfig,
) -> dict[str, object]:
    fold_entries: list[dict] = []
    portfolio_summaries: list[dict] = []
    symbol_rows: list[dict] = []
    fold_equity_curves: list[pd.DataFrame] = []

    for fold_id, window in enumerate(windows, start=1):
        evaluation = _current_strategy_rows(
            frames,
            symbols,
            window.evaluation_start,
            window.evaluation_end,
            backtest_config,
            risk_config,
        )
        portfolio_summary = evaluation["portfolio"]
        fold_entries.append(
            {
                "fold": fold_id,
                "window": _serialize_window(window),
                "portfolio": portfolio_summary,
                "stability": evaluation["stability"],
                "rows": evaluation["rows"],
            }
        )
        portfolio_summaries.append({"fold": fold_id, **portfolio_summary})
        fold_equity_curves.append(evaluation["equity_curve"])
        for row in evaluation["rows"]:
            symbol_rows.append({"fold": fold_id, **row})

    stitched_equity_curve = _stitch_fold_equity_curves(
        fold_equity_curves,
        initial_cash=backtest_config.initial_cash,
    )
    stability = _equity_stability_metrics(stitched_equity_curve)

    return {
        "aggregate": _aggregate_walk_forward_objective(portfolio_summaries, symbol_rows, stability),
        "portfolio": _aggregate_portfolio_summaries(portfolio_summaries, stability),
        "rows": _aggregate_symbol_rows(symbol_rows),
        "stability": stability,
        "folds": fold_entries,
    }


def _build_walk_forward_variant_frames(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    windows: list[WalkForwardWindow],
    params: dict,
    risk_config: RiskConfig,
) -> dict[str, pd.DataFrame]:
    stitched: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
    for window in windows:
        fallback_scores = _global_validation_scores(frames, window.calibration_start, window.calibration_end)
        segment_frames = _build_variant_frames(
            frames=frames,
            symbols=symbols,
            start=window.evaluation_start,
            end=window.evaluation_end,
            validation_start=window.calibration_start,
            validation_end=window.calibration_end,
            params=params,
            risk_config=risk_config,
            fallback_scores=fallback_scores,
        )
        for symbol in symbols:
            stitched[symbol].append(segment_frames[symbol])

    merged_frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if not stitched[symbol]:
            merged_frames[symbol] = frames[symbol].iloc[0:0].copy()
            continue
        merged = pd.concat(stitched[symbol], ignore_index=True)
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        merged_frames[symbol] = merged
    return merged_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Search for a more robust live-trading parameter set.")
    parser.add_argument("--training-summary", default=str(REPORT_DIR / "qlib_training_summary.json"))
    parser.add_argument("--max-trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-summary", default=str(REPORT_DIR / "live_strategy_search_summary.json"))
    parser.add_argument("--anchor-params-json", default=None)
    parser.add_argument("--anchor-params-file", default=None)
    parser.add_argument("--fixed-params-json", default=None)
    parser.add_argument("--fixed-params-file", default=None)
    parser.add_argument("--refine-around-current-best", action="store_true")
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--holdout-days", type=int, default=365)
    parser.add_argument("--validation-window-days", type=int, default=180)
    parser.add_argument("--calibration-window-days", type=int, default=365)
    parser.add_argument("--fold-step-days", type=int, default=180)
    args = parser.parse_args()

    summary = _load_training_summary(Path(args.training_summary))
    symbols = list(summary["symbols"])
    frames = _load_feature_frames(symbols)
    segments = summary["segments"]
    test_start = _timestamp(segments["test"][0])
    test_end = _timestamp(segments["test"][1])
    search_end, holdout_start = _build_holdout_split(test_start, test_end, args.holdout_days)
    walk_forward_windows = _build_walk_forward_windows(
        search_start=test_start,
        search_end=search_end,
        calibration_window_days=args.calibration_window_days,
        validation_window_days=args.validation_window_days,
        step_days=args.fold_step_days,
    )
    holdout_window = WalkForwardWindow(
        calibration_start=holdout_start - pd.Timedelta(days=args.calibration_window_days),
        calibration_end=holdout_start - pd.Timedelta(seconds=1),
        evaluation_start=holdout_start,
        evaluation_end=test_end,
    )

    backtest_config = BacktestConfig()
    risk_config = RiskConfig()

    anchor_params: dict | None = None
    if args.refine_around_current_best:
        anchor_params = _load_json_arg(args.anchor_params_json, args.anchor_params_file)
        if anchor_params:
            pass
        else:
            anchor_path = Path(args.anchor_summary)
            if anchor_path.exists():
                anchor_summary = _load_training_summary(anchor_path)
                anchor_params = anchor_summary.get("best_params")
        if not anchor_params:
            raise RuntimeError(
                "--refine-around-current-best requested, but no anchor params were available."
            )
    fixed_params = _load_json_arg(args.fixed_params_json, args.fixed_params_file)
    parameter_space = _parameter_space(anchor_params=anchor_params, local_radius=args.local_radius)
    parameter_space = _apply_fixed_params(parameter_space, fixed_params)

    baseline_validation = _evaluate_walk_forward_current_strategy(
        frames,
        symbols,
        walk_forward_windows,
        backtest_config,
        risk_config,
    )
    baseline_holdout_eval = _current_strategy_rows(
        frames,
        symbols,
        holdout_start,
        test_end,
        backtest_config,
        risk_config,
    )
    baseline_full_test_eval = _current_strategy_rows(
        frames,
        symbols,
        test_start,
        test_end,
        backtest_config,
        risk_config,
    )

    trial_rows: list[dict] = []
    best_params: dict | None = None
    best_validation_objective = float("-inf")
    best_validation: dict[str, object] | None = None

    for trial_id, params in enumerate(_sample_parameter_sets(args.max_trials, args.seed, parameter_space), start=1):
        validation_result = _evaluate_walk_forward_candidate(
            frames=frames,
            symbols=symbols,
            windows=walk_forward_windows,
            params=params,
            backtest_config=backtest_config,
            risk_config=risk_config,
            include_rows=False,
        )
        aggregate = validation_result["aggregate"]
        trial_rows.append({"trial": trial_id, **params, **aggregate})
        if aggregate["objective"] > best_validation_objective:
            best_validation_objective = aggregate["objective"]
            best_params = params
            best_validation = validation_result

    if best_params is None or best_validation is None:
        raise RuntimeError("No parameter candidates were evaluated.")

    best_validation = _evaluate_walk_forward_candidate(
        frames=frames,
        symbols=symbols,
        windows=walk_forward_windows,
        params=best_params,
        backtest_config=backtest_config,
        risk_config=risk_config,
        include_rows=True,
    )

    best_holdout_frames = _build_variant_frames(
        frames=frames,
        symbols=symbols,
        start=holdout_window.evaluation_start,
        end=holdout_window.evaluation_end,
        validation_start=holdout_window.calibration_start,
        validation_end=holdout_window.calibration_end,
        params=best_params,
        risk_config=risk_config,
        fallback_scores=_global_validation_scores(frames, holdout_window.calibration_start, holdout_window.calibration_end),
    )
    best_holdout_eval = _evaluate_frames(
        frames=best_holdout_frames,
        symbols=symbols,
        backtest_config=backtest_config,
        risk_config=risk_config,
    )
    best_full_test_frames = _build_walk_forward_variant_frames(
        frames=frames,
        symbols=symbols,
        windows=[*walk_forward_windows, holdout_window],
        params=best_params,
        risk_config=risk_config,
    )
    best_full_test_eval = _evaluate_frames(
        frames=best_full_test_frames,
        symbols=symbols,
        backtest_config=backtest_config,
        risk_config=risk_config,
    )

    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "symbols": symbols,
        "search_trials": args.max_trials,
        "search_method": "walk_forward_holdout",
        "search_config": {
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "search_end": search_end.isoformat(),
            "holdout_start": holdout_start.isoformat(),
            "holdout_days": args.holdout_days,
            "validation_window_days": args.validation_window_days,
            "calibration_window_days": args.calibration_window_days,
            "fold_step_days": args.fold_step_days,
            "refine_around_current_best": bool(args.refine_around_current_best),
            "local_radius": int(args.local_radius),
            "anchor_summary": str(Path(args.anchor_summary)),
            "anchor_params_json": args.anchor_params_json,
            "anchor_params_file": args.anchor_params_file,
            "fixed_params_json": args.fixed_params_json,
            "fixed_params_file": args.fixed_params_file,
            "fold_count": len(walk_forward_windows),
            "fold_windows": [_serialize_window(window) for window in walk_forward_windows],
        },
        "baseline_validation": baseline_validation,
        "baseline_holdout": {
            "aggregate": _aggregate_single_period_objective(
                baseline_holdout_eval["rows"],
                baseline_holdout_eval["portfolio"],
                baseline_holdout_eval["stability"],
            ),
            "portfolio": baseline_holdout_eval["portfolio"],
            "stability": baseline_holdout_eval["stability"],
            "rows": baseline_holdout_eval["rows"],
        },
        "baseline_test": {
            "aggregate": _aggregate_single_period_objective(
                baseline_full_test_eval["rows"],
                baseline_full_test_eval["portfolio"],
                baseline_full_test_eval["stability"],
            ),
            "portfolio": baseline_full_test_eval["portfolio"],
            "stability": baseline_full_test_eval["stability"],
            "rows": baseline_full_test_eval["rows"],
        },
        "best_params": best_params,
        "best_validation": best_validation,
        "best_holdout": {
            "aggregate": _aggregate_single_period_objective(
                best_holdout_eval["rows"],
                best_holdout_eval["portfolio"],
                best_holdout_eval["stability"],
            ),
            "portfolio": best_holdout_eval["portfolio"],
            "stability": best_holdout_eval["stability"],
            "rows": best_holdout_eval["rows"],
            "window": _serialize_window(holdout_window),
        },
        "best_test": {
            "aggregate": _aggregate_single_period_objective(
                best_full_test_eval["rows"],
                best_full_test_eval["portfolio"],
                best_full_test_eval["stability"],
            ),
            "portfolio": best_full_test_eval["portfolio"],
            "stability": best_full_test_eval["stability"],
            "rows": best_full_test_eval["rows"],
        },
    }

    trial_frame = pd.DataFrame(trial_rows).sort_values("objective", ascending=False).reset_index(drop=True)
    trial_frame.to_csv(REPORT_DIR / "live_strategy_search_trials.csv", index=False)
    (REPORT_DIR / "live_strategy_search_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
