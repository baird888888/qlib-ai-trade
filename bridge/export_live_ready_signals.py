from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from bridge.signal_store import to_external_signal_frame, write_signal_frame
from bridge.rank_utils import reference_scores_with_fallback, score_percentiles
from quant_trading.backtest.engine import run_backtest, run_portfolio_backtest
from quant_trading.config import BacktestConfig, RiskConfig
from quant_trading.signals.multi_timeframe import apply_practical_signal_overrides, apply_rotation_overlay

RUNTIME_ROOT = PROJECT_ROOT / "bridge" / "runtime"
FEATURE_DIR = RUNTIME_ROOT / "features"
REPORT_DIR = RUNTIME_ROOT / "reports"

LIVE_READY_PROFILE = {
    "profile_name": "practical_live_v3_search_best",
    "entry_quantile": 0.45,
    "exit_quantile": 0.20,
    "confidence_threshold": 0.35,
    "exit_confidence_threshold": 0.30,
    "min_confirmation_count": 1,
    "qlib_rank_threshold": 0.55,
    "qlib_exit_rank_threshold": 0.30,
    "stop_loss_scale": 1.00,
    "take_profit_scale": 1.20,
    "kelly_scale": 1.00,
    "max_leverage": 4.0,
    "initial_entry_fraction": 0.45,
    "layer_stake_fraction": 0.30,
    "max_entry_layers": 3,
    "reduce_stake_fraction": 0.35,
    "partial_take_profit_scale": 0.55,
    "rotation_top_n": 1,
    "rotation_min_score": 0.55,
}


def _slug(symbol: str) -> str:
    return symbol.lower().replace("-", "_").replace("/", "_")


def _load_summary() -> dict:
    return json.loads((REPORT_DIR / "qlib_training_summary.json").read_text(encoding="utf-8"))


def _slice_frame(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    return frame.loc[(timestamp >= start_ts) & (timestamp <= end_ts)].reset_index(drop=True)


def _quantile_thresholds(scores: pd.Series) -> tuple[float, float]:
    clean_scores = pd.to_numeric(scores, errors="coerce").dropna()
    if clean_scores.empty:
        return 0.0, 0.0
    return (
        float(clean_scores.quantile(LIVE_READY_PROFILE["entry_quantile"])),
        float(clean_scores.quantile(LIVE_READY_PROFILE["exit_quantile"])),
    )


def _thresholds_from_validation(
    validation_frame: pd.DataFrame,
    fallback_scores: pd.Series,
) -> tuple[tuple[float, float], str]:
    scores = pd.to_numeric(validation_frame.get("qlib_score"), errors="coerce").dropna()
    if scores.empty:
        return _quantile_thresholds(fallback_scores), "global_validation_fallback"
    return _quantile_thresholds(scores), "symbol_validation"


def main() -> None:
    summary = _load_summary()
    symbols = list(summary["symbols"])
    segments = summary["segments"]
    risk_config = RiskConfig()
    backtest_config = BacktestConfig()

    frames: dict[str, pd.DataFrame] = {
        symbol: pd.read_parquet(FEATURE_DIR / f"{_slug(symbol)}_qlib_signals.parquet")
        for symbol in symbols
    }
    validation_score_blocks = [
        pd.to_numeric(
            _slice_frame(frame, segments["valid"][0], segments["valid"][1])["qlib_score"],
            errors="coerce",
        ).dropna()
        for frame in frames.values()
    ]
    fallback_scores = (
        pd.concat(validation_score_blocks, ignore_index=True)
        if validation_score_blocks
        else pd.Series(dtype="float64")
    )

    threshold_sources: dict[str, str] = {}
    live_ready_frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = frames[symbol]
        validation_frame = _slice_frame(frame, segments["valid"][0], segments["valid"][1])
        entry_exit_thresholds, threshold_source = _thresholds_from_validation(validation_frame, fallback_scores)
        entry_threshold, exit_threshold = entry_exit_thresholds
        threshold_sources[symbol] = threshold_source
        local_reference_scores = reference_scores_with_fallback(validation_frame.get("qlib_score"), fallback_scores)
        prepared_frame = frame.copy()
        prepared_frame["qlib_score_rank"] = score_percentiles(prepared_frame.get("qlib_score"), local_reference_scores)

        live_ready_frame = apply_practical_signal_overrides(
            prepared_frame,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            confidence_threshold=LIVE_READY_PROFILE["confidence_threshold"],
            exit_confidence_threshold=LIVE_READY_PROFILE["exit_confidence_threshold"],
            min_confirmation_count=LIVE_READY_PROFILE["min_confirmation_count"],
            qlib_rank_threshold=LIVE_READY_PROFILE["qlib_rank_threshold"],
            qlib_exit_rank_threshold=LIVE_READY_PROFILE["qlib_exit_rank_threshold"],
            stop_loss_scale=LIVE_READY_PROFILE["stop_loss_scale"],
            take_profit_scale=LIVE_READY_PROFILE["take_profit_scale"],
            kelly_scale=LIVE_READY_PROFILE["kelly_scale"],
            max_leverage=LIVE_READY_PROFILE["max_leverage"],
            initial_entry_fraction=LIVE_READY_PROFILE["initial_entry_fraction"],
            layer_stake_fraction=LIVE_READY_PROFILE["layer_stake_fraction"],
            max_entry_layers=LIVE_READY_PROFILE["max_entry_layers"],
            reduce_stake_fraction=LIVE_READY_PROFILE["reduce_stake_fraction"],
            partial_take_profit_scale=LIVE_READY_PROFILE["partial_take_profit_scale"],
            risk_config=risk_config,
        )
        live_ready_frame["symbol"] = symbol
        live_ready_frame["entry_threshold"] = entry_threshold
        live_ready_frame["exit_threshold"] = exit_threshold
        live_ready_frames[symbol] = live_ready_frame

    live_ready_frames = apply_rotation_overlay(
        live_ready_frames,
        top_n=LIVE_READY_PROFILE["rotation_top_n"],
        min_rotation_score=LIVE_READY_PROFILE["rotation_min_score"],
    )

    external_frames: list[pd.DataFrame] = []
    report_rows: list[dict] = []
    live_ready_test_frames: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        live_ready_frame = live_ready_frames[symbol]
        live_ready_frame.to_parquet(FEATURE_DIR / f"{_slug(symbol)}_live_ready_signals.parquet", index=False)
        external_frames.append(to_external_signal_frame(live_ready_frame, symbol))

        live_ready_test = _slice_frame(live_ready_frame, segments["test"][0], segments["test"][1])
        live_ready_test_frames[symbol] = live_ready_test
        backtest = run_backtest(live_ready_test, backtest_config=backtest_config, risk_config=risk_config)
        entry_rows = live_ready_test.loc[live_ready_test["entry_long_signal"] == 1]
        report_rows.append(
            {
                "symbol": symbol,
                "threshold_source": threshold_sources[symbol],
                "entry_threshold": float(live_ready_frame["entry_threshold"].iloc[-1]) if not live_ready_frame.empty else 0.0,
                "exit_threshold": float(live_ready_frame["exit_threshold"].iloc[-1]) if not live_ready_frame.empty else 0.0,
                "avg_target_leverage": float(pd.to_numeric(entry_rows.get("target_leverage", 1.0), errors="coerce").mean() or 0.0),
                "avg_rotation_weight": float(pd.to_numeric(entry_rows.get("rotation_weight", 0.0), errors="coerce").mean() or 0.0),
                **backtest["summary"],
            }
        )

    portfolio_backtest = run_portfolio_backtest(
        live_ready_test_frames,
        backtest_config=backtest_config,
        risk_config=risk_config,
    )
    write_signal_frame(pd.concat(external_frames, ignore_index=True))
    report_frame = pd.DataFrame(report_rows)
    report_frame.to_csv(REPORT_DIR / "live_ready_backtest_summary.csv", index=False)
    if not portfolio_backtest["trades"].empty:
        portfolio_backtest["trades"].to_csv(REPORT_DIR / "live_ready_portfolio_trades.csv", index=False)
    if not portfolio_backtest["equity_curve"].empty:
        portfolio_backtest["equity_curve"].to_csv(REPORT_DIR / "live_ready_portfolio_equity.csv", index=False)
    (REPORT_DIR / "live_ready_portfolio_summary.json").write_text(
        json.dumps(portfolio_backtest["summary"], indent=2),
        encoding="utf-8",
    )
    (REPORT_DIR / "live_ready_profile.json").write_text(
        json.dumps(
            {
                "profile": LIVE_READY_PROFILE,
                "symbols": symbols,
                "results": report_rows,
                "portfolio": portfolio_backtest["summary"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "profile": LIVE_READY_PROFILE,
                "results": report_rows,
                "portfolio": portfolio_backtest["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
