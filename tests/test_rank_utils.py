from __future__ import annotations

import unittest

import pandas as pd

from bridge.rank_utils import reference_scores_with_fallback, score_percentiles


class RankUtilsTests(unittest.TestCase):
    def test_score_percentiles_use_reference_only(self) -> None:
        scores = pd.Series([1.0, 2.0, 3.0])
        reference = pd.Series([10.0, 20.0, 30.0, 40.0])

        result = score_percentiles(scores, reference)

        self.assertAlmostEqual(float(result.iloc[0]), 0.0, places=6)
        self.assertAlmostEqual(float(result.iloc[1]), 0.0, places=6)
        self.assertAlmostEqual(float(result.iloc[2]), 0.0, places=6)

    def test_reference_scores_with_fallback_prefers_local_scores(self) -> None:
        local = pd.Series([0.1, 0.2, 0.3])
        fallback = pd.Series([0.9, 1.0])

        result = reference_scores_with_fallback(local, fallback)

        self.assertEqual(result.tolist(), [0.1, 0.2, 0.3])

    def test_reference_scores_with_fallback_uses_fallback_when_local_empty(self) -> None:
        local = pd.Series([None, float("nan")])
        fallback = pd.Series([0.9, 1.0])

        result = reference_scores_with_fallback(local, fallback)

        self.assertEqual(result.tolist(), [0.9, 1.0])


if __name__ == "__main__":
    unittest.main()
