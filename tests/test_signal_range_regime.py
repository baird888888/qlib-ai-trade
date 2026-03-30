from __future__ import annotations

import unittest

import pandas as pd

from quant_trading.signals.multi_timeframe import apply_practical_signal_overrides


def _base_range_frame(close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [close],
            "ema_fast": [100.0],
            "ema_slow": [100.4],
            "h1_close": [100.0],
            "h1_ema_fast": [100.0],
            "h1_ema_slow": [100.5],
            "trend_strength": [0.10],
            "h1_trend_strength": [0.15],
            "wyckoff_support": [98.0],
            "wyckoff_resistance": [110.0],
            "h1_wyckoff_support": [96.0],
            "h1_wyckoff_resistance": [112.0],
            "signal_confidence": [0.42],
            "qlib_score": [0.002],
            "qlib_score_rank": [0.60],
            "relative_volume": [1.0],
            "atr_pct": [0.012],
            "returns_1": [-0.002],
            "returns_3": [-0.004],
            "close_location": [0.62],
            "entry_long_signal": [0],
            "exit_long_signal": [0],
            "brooks_bull_signal": [0],
            "chan_long_confirmation": [0],
            "simons_breakout_long": [0],
            "trend_filter_long": [0],
            "wyckoff_long_context": [0],
            "wyckoff_accumulation": [1],
            "wyckoff_spring": [0],
            "wyckoff_upthrust": [0],
            "stop_loss_pct": [0.02],
            "take_profit_pct": [0.05],
            "kelly_fraction": [0.10],
        }
    )


class RangeRegimeSignalTests(unittest.TestCase):
    def test_range_regime_can_open_long_near_support_without_trend_entry(self) -> None:
        frame = _base_range_frame(close=100.0)

        result = apply_practical_signal_overrides(
            frame,
            entry_threshold=0.001,
            exit_threshold=-0.001,
            confidence_threshold=0.45,
            qlib_rank_threshold=0.60,
            range_entry_zone=0.30,
            range_qlib_rank_threshold=0.45,
            range_leverage_cap=2.0,
        )
        row = result.iloc[0]

        self.assertEqual(int(row["trend_entry_long_signal"]), 0)
        self.assertEqual(int(row["range_regime"]), 1)
        self.assertEqual(int(row["range_entry_long_signal"]), 1)
        self.assertEqual(int(row["entry_long_signal"]), 1)
        self.assertLessEqual(float(row["target_leverage"]), 2.0)

    def test_range_regime_exits_near_resistance(self) -> None:
        frame = _base_range_frame(close=109.0)
        frame["returns_1"] = [0.01]

        result = apply_practical_signal_overrides(
            frame,
            entry_threshold=0.001,
            exit_threshold=-0.001,
            range_exit_zone=0.72,
            range_qlib_rank_threshold=0.45,
        )
        row = result.iloc[0]

        self.assertEqual(int(row["range_regime"]), 1)
        self.assertEqual(int(row["range_exit_long_signal"]), 1)
        self.assertEqual(int(row["exit_long_signal"]), 1)


if __name__ == "__main__":
    unittest.main()
