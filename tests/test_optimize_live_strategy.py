from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from bridge.optimize_live_strategy import (
    _apply_fixed_params,
    _build_holdout_split,
    _build_walk_forward_windows,
    _equity_stability_metrics,
    _load_json_arg,
    _stitch_fold_equity_curves,
)


class OptimizeLiveStrategyTests(unittest.TestCase):
    def test_build_holdout_split_keeps_search_and_holdout_disjoint(self) -> None:
        test_start = pd.Timestamp("2020-01-01 00:00:00+00:00")
        test_end = pd.Timestamp("2021-12-31 00:00:00+00:00")

        search_end, holdout_start = _build_holdout_split(test_start, test_end, holdout_days=120)

        self.assertLess(search_end, holdout_start)
        self.assertEqual(search_end + pd.Timedelta(seconds=1), holdout_start)
        self.assertGreater(holdout_start, test_start)

    def test_build_walk_forward_windows_cover_search_range_without_overlap(self) -> None:
        search_start = pd.Timestamp("2020-01-01 00:00:00+00:00")
        search_end = pd.Timestamp("2021-08-15 00:00:00+00:00")

        windows = _build_walk_forward_windows(
            search_start=search_start,
            search_end=search_end,
            calibration_window_days=180,
            validation_window_days=120,
            step_days=120,
        )

        self.assertGreaterEqual(len(windows), 3)
        self.assertEqual(windows[0].evaluation_start, search_start)
        self.assertEqual(windows[-1].evaluation_end, search_end)

        for previous, current in zip(windows, windows[1:]):
            self.assertLess(previous.evaluation_end, current.evaluation_start)
            self.assertEqual(previous.evaluation_end + pd.Timedelta(seconds=1), current.evaluation_start)

        for window in windows:
            self.assertLess(window.calibration_end, window.evaluation_start)
            self.assertEqual(window.calibration_end + pd.Timedelta(seconds=1), window.evaluation_start)

    def test_stitch_fold_equity_curves_rebases_sequentially(self) -> None:
        fold_one = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01 00:00:00+00:00", "2024-01-02 00:00:00+00:00"],
                    utc=True,
                ),
                "equity": [100_000.0, 110_000.0],
            }
        )
        fold_two = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-03 00:00:00+00:00", "2024-01-04 00:00:00+00:00"],
                    utc=True,
                ),
                "equity": [100_000.0, 105_000.0],
            }
        )

        stitched = _stitch_fold_equity_curves([fold_one, fold_two], initial_cash=100_000.0)

        self.assertAlmostEqual(float(stitched["equity"].iloc[0]), 100_000.0, places=6)
        self.assertAlmostEqual(float(stitched["equity"].iloc[1]), 110_000.0, places=6)
        self.assertAlmostEqual(float(stitched["equity"].iloc[2]), 110_000.0, places=6)
        self.assertAlmostEqual(float(stitched["equity"].iloc[3]), 115_500.0, places=6)

    def test_equity_stability_metrics_measure_positive_period_ratios(self) -> None:
        equity_curve = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00+00:00",
                        "2024-01-08 00:00:00+00:00",
                        "2024-01-15 00:00:00+00:00",
                        "2024-02-01 00:00:00+00:00",
                        "2024-03-01 00:00:00+00:00",
                        "2025-01-01 00:00:00+00:00",
                    ],
                    utc=True,
                ),
                "equity": [100.0, 110.0, 105.0, 120.0, 118.0, 150.0],
            }
        )

        metrics = _equity_stability_metrics(equity_curve)

        self.assertGreater(metrics["weekly_periods"], 0)
        self.assertGreater(metrics["monthly_periods"], 0)
        self.assertGreater(metrics["yearly_periods"], 0)
        self.assertGreater(metrics["weekly_positive_ratio"], 0.0)
        self.assertLess(metrics["weekly_positive_ratio"], 1.0)
        self.assertLess(metrics["weekly_worst_return_pct"], 0.0)

    def test_apply_fixed_params_constrains_selected_keys(self) -> None:
        space = {
            "enable_range_reversion": [False, True],
            "max_leverage": [3.0, 4.0, 5.0],
        }

        constrained = _apply_fixed_params(
            space,
            {"enable_range_reversion": False, "max_leverage": 5.0},
        )

        self.assertEqual(constrained["enable_range_reversion"], [False])
        self.assertEqual(constrained["max_leverage"], [5.0])

    def test_load_json_arg_prefers_inline_and_supports_file(self) -> None:
        self.assertEqual(_load_json_arg('{"a": 1}', None), {"a": 1})

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "params.json"
            path.write_text('{"b": 2}', encoding="utf-8")
            self.assertEqual(_load_json_arg(None, str(path)), {"b": 2})


if __name__ == "__main__":
    unittest.main()
