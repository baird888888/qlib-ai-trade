from __future__ import annotations

import numpy as np
import pandas as pd


def score_percentiles(
    scores: pd.Series,
    reference_scores: pd.Series,
    *,
    default: float = 0.5,
) -> pd.Series:
    score_series = pd.to_numeric(scores, errors="coerce")
    clean_reference = pd.to_numeric(reference_scores, errors="coerce").dropna().to_numpy(dtype="float64")
    if clean_reference.size == 0:
        return pd.Series(default, index=score_series.index, dtype="float64")

    sorted_reference = np.sort(clean_reference)
    raw_scores = score_series.to_numpy(dtype="float64", copy=True)
    output = np.full(score_series.shape[0], float(default), dtype="float64")
    valid_mask = np.isfinite(raw_scores)
    if valid_mask.any():
        output[valid_mask] = np.searchsorted(sorted_reference, raw_scores[valid_mask], side="right") / sorted_reference.size
    return pd.Series(output, index=score_series.index, dtype="float64").clip(lower=0.0, upper=1.0)


def reference_scores_with_fallback(local_scores: pd.Series, fallback_scores: pd.Series) -> pd.Series:
    clean_local = pd.to_numeric(local_scores, errors="coerce").dropna()
    if not clean_local.empty:
        return clean_local.reset_index(drop=True)
    return pd.to_numeric(fallback_scores, errors="coerce").dropna().reset_index(drop=True)
