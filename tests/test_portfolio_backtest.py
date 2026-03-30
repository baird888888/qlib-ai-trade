from __future__ import annotations

import unittest

import pandas as pd

from quant_trading.backtest.engine import run_portfolio_backtest
from quant_trading.config import BacktestConfig, RiskConfig
from quant_trading.features.technical import merge_higher_timeframe
from quant_trading.signals.multi_timeframe import apply_rotation_overlay


def _signal_frame(
    prices: list[float],
    *,
    opens: list[float] | None = None,
    entry_index: int = 0,
    exit_index: int | None = None,
    rotation_score: float = 0.8,
    entry_stake_fraction: float = 1.0,
    layer_stake_fraction: float = 0.3,
) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(prices), freq="5min", tz="UTC")
    opens = opens or prices
    entry_flags = [0] * len(prices)
    exit_flags = [0] * len(prices)
    entry_flags[entry_index] = 1
    if exit_index is not None:
        exit_flags[exit_index] = 1
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": prices,
            "low": prices,
            "close": prices,
            "entry_long_signal": entry_flags,
            "exit_long_signal": exit_flags,
            "add_position_signal": [0] * len(prices),
            "reduce_position_signal": [0] * len(prices),
            "stop_loss_pct": [0.02] * len(prices),
            "take_profit_pct": [0.50] * len(prices),
            "partial_take_profit_pct": [0.25] * len(prices),
            "kelly_fraction": [0.25] * len(prices),
            "target_leverage": [1.0] * len(prices),
            "entry_stake_fraction": [entry_stake_fraction] * len(prices),
            "layer_stake_fraction": [layer_stake_fraction] * len(prices),
            "reduce_stake_fraction": [0.33] * len(prices),
            "max_entry_layers": [1] * len(prices),
            "rotation_score": [rotation_score] * len(prices),
        }
    )


class PortfolioBacktestTests(unittest.TestCase):
    def test_run_portfolio_backtest_shares_one_cash_pool(self) -> None:
        frames = {
            "AAA-USDT": _signal_frame([100.0, 100.0, 110.0, 110.0], exit_index=2, entry_stake_fraction=1.0),
            "BBB-USDT": _signal_frame([100.0, 100.0, 110.0, 110.0], exit_index=2, entry_stake_fraction=1.0),
        }

        result = run_portfolio_backtest(
            frames,
            backtest_config=BacktestConfig(initial_cash=100_000.0, commission=0.0, spread=0.0),
            risk_config=RiskConfig(),
        )

        self.assertEqual(result["summary"]["trades"], 1)
        self.assertAlmostEqual(float(result["summary"]["return_pct"]), 10.0, places=6)
        self.assertEqual(result["trades"]["symbol"].tolist(), ["AAA-USDT"])
        self.assertLessEqual(float(result["equity_curve"]["used_margin"].max()), 100_000.0)

    def test_run_portfolio_backtest_executes_on_next_bar(self) -> None:
        result = run_portfolio_backtest(
            {
                "AAA-USDT": _signal_frame(
                    [100.0, 120.0, 120.0, 120.0],
                    opens=[100.0, 120.0, 120.0, 120.0],
                    entry_index=0,
                    exit_index=1,
                    entry_stake_fraction=1.0,
                )
            },
            backtest_config=BacktestConfig(initial_cash=100_000.0, commission=0.0, spread=0.0),
            risk_config=RiskConfig(),
        )

        self.assertEqual(result["summary"]["trades"], 1)
        self.assertAlmostEqual(float(result["summary"]["return_pct"]), 0.0, places=6)
        self.assertAlmostEqual(float(result["trades"]["entry_price"].iloc[0]), 120.0, places=6)
        self.assertAlmostEqual(float(result["trades"]["exit_price"].iloc[0]), 120.0, places=6)

    def test_run_portfolio_backtest_applies_slippage_to_execution_prices(self) -> None:
        result = run_portfolio_backtest(
            {
                "AAA-USDT": _signal_frame(
                    [100.0, 100.0, 100.0, 100.0],
                    opens=[100.0, 100.0, 100.0, 100.0],
                    entry_index=0,
                    exit_index=1,
                    entry_stake_fraction=1.0,
                )
            },
            backtest_config=BacktestConfig(initial_cash=100_000.0, commission=0.0, spread=0.01),
            risk_config=RiskConfig(),
        )

        self.assertEqual(result["summary"]["trades"], 1)
        self.assertAlmostEqual(float(result["trades"]["entry_price"].iloc[0]), 101.0, places=6)
        self.assertAlmostEqual(float(result["trades"]["exit_price"].iloc[0]), 99.0, places=6)
        self.assertLess(float(result["summary"]["return_pct"]), 0.0)

    def test_apply_rotation_overlay_blocks_unselected_entries(self) -> None:
        frames = {
            "AAA-USDT": _signal_frame([100.0], rotation_score=0.95, entry_stake_fraction=0.5, layer_stake_fraction=0.3),
            "BBB-USDT": _signal_frame([100.0], rotation_score=0.75, entry_stake_fraction=0.5, layer_stake_fraction=0.3),
        }

        rotated = apply_rotation_overlay(frames, top_n=1, min_rotation_score=0.55)
        leader = rotated["AAA-USDT"].iloc[0]
        follower = rotated["BBB-USDT"].iloc[0]

        self.assertEqual(int(leader["entry_long_signal"]), 1)
        self.assertAlmostEqual(float(leader["rotation_weight"]), 1.0, places=6)
        self.assertEqual(int(follower["entry_long_signal"]), 0)
        self.assertAlmostEqual(float(follower["rotation_weight"]), 0.0, places=6)
        self.assertAlmostEqual(float(follower["entry_stake_fraction"]), 0.0, places=6)
        self.assertEqual(int(follower["add_position_signal"]), 0)

    def test_merge_higher_timeframe_waits_for_informative_close(self) -> None:
        lower = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01 10:00:00", periods=14, freq="5min", tz="UTC"),
                "close": range(14),
            }
        )
        higher = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:00:00+00:00", "2024-01-01 11:00:00+00:00"],
                    utc=True,
                ),
                "marker": [1.0, 2.0],
                "bar": ["1H", "1H"],
            }
        )

        merged = merge_higher_timeframe(lower, higher, prefix="h1_")

        self.assertTrue(pd.isna(merged.loc[merged["timestamp"] == pd.Timestamp("2024-01-01 10:55:00+00:00"), "h1_marker"]).all())
        self.assertEqual(
            float(merged.loc[merged["timestamp"] == pd.Timestamp("2024-01-01 11:00:00+00:00"), "h1_marker"].iloc[0]),
            1.0,
        )
        self.assertEqual(
            float(merged.loc[merged["timestamp"] == pd.Timestamp("2024-01-01 11:05:00+00:00"), "h1_marker"].iloc[0]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
