from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from bridge.optimize_live_strategy import (  # noqa: E402
    REPORT_DIR,
    _build_variant_frames,
    _global_validation_scores,
    _load_feature_frames,
    _load_training_summary,
    _timestamp,
)
from quant_trading.backtest.engine import run_portfolio_backtest  # noqa: E402
from quant_trading.config import BacktestConfig, RiskConfig  # noqa: E402


def _load_search_summary() -> dict:
    return json.loads((REPORT_DIR / "live_strategy_search_summary.json").read_text(encoding="utf-8"))


def main() -> None:
    training_summary = _load_training_summary(REPORT_DIR / "qlib_training_summary.json")
    search_summary = _load_search_summary()

    symbols = list(search_summary["symbols"])
    segments = training_summary["segments"]
    validation_start = _timestamp(segments["valid"][0])
    validation_end = _timestamp(segments["valid"][1])
    test_start = _timestamp(segments["test"][0])
    test_end = _timestamp(segments["test"][1])

    frames = _load_feature_frames(symbols)
    fallback_scores = _global_validation_scores(frames, validation_start, validation_end)

    best_test_frames = _build_variant_frames(
        frames=frames,
        symbols=symbols,
        start=test_start,
        end=test_end,
        validation_start=validation_start,
        validation_end=validation_end,
        params=search_summary["best_params"],
        risk_config=RiskConfig(),
        fallback_scores=fallback_scores,
    )

    portfolio_backtest = run_portfolio_backtest(
        best_test_frames,
        backtest_config=BacktestConfig(),
        risk_config=RiskConfig(),
    )

    equity_curve = portfolio_backtest["equity_curve"].copy()
    trades = portfolio_backtest["trades"].copy()
    summary = portfolio_backtest["summary"]

    equity_csv_path = REPORT_DIR / "best_test_portfolio_equity_norepaint.csv"
    trades_csv_path = REPORT_DIR / "best_test_portfolio_trades_norepaint.csv"
    png_path = REPORT_DIR / "best_test_portfolio_equity_curve_norepaint.png"
    summary_path = REPORT_DIR / "best_test_portfolio_summary_norepaint.json"

    if not equity_curve.empty:
        equity_curve["timestamp"] = pd.to_datetime(equity_curve["timestamp"], utc=True)
        equity_curve.to_csv(equity_csv_path, index=False)

        plt.figure(figsize=(14, 7))
        plt.plot(equity_curve["timestamp"], equity_curve["equity"], color="#0f766e", linewidth=1.5)
        plt.axhline(BacktestConfig().initial_cash, color="#94a3b8", linestyle="--", linewidth=1.0)
        plt.title("Best Test Portfolio Equity (No Repaint, 0.05% Fee, 0.02% Slippage)")
        plt.xlabel("Timestamp (UTC)")
        plt.ylabel("Equity")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()

    if not trades.empty:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
        trades.to_csv(trades_csv_path, index=False)

    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source_search_summary": str(REPORT_DIR / "live_strategy_search_summary.json"),
        "source_training_summary": str(REPORT_DIR / "qlib_training_summary.json"),
        "best_params": search_summary["best_params"],
        "portfolio_summary": summary,
        "symbols": symbols,
        "artifacts": {
            "equity_csv": str(equity_csv_path),
            "trades_csv": str(trades_csv_path),
            "equity_png": str(png_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
