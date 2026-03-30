from __future__ import annotations

from pathlib import Path

import pandas as pd

RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
DEFAULT_SIGNAL_PATH = RUNTIME_DIR / "freqtrade_signals.parquet"

SIGNAL_COLUMNS = [
    "date",
    "symbol",
    "canonical_pair",
    "freqtrade_pair",
    "entry_long_signal",
    "exit_long_signal",
    "add_position_signal",
    "reduce_position_signal",
    "signal_confidence",
    "qlib_score",
    "qlib_score_rank",
    "stop_loss_pct",
    "take_profit_pct",
    "partial_take_profit_pct",
    "kelly_fraction",
    "target_leverage",
    "entry_stake_fraction",
    "layer_stake_fraction",
    "reduce_stake_fraction",
    "max_entry_layers",
    "rotation_score",
    "rotation_rank",
    "rotation_weight",
    "rotation_entry_long_signal",
]


def canonical_pair(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    if ":" in text:
        text = text.split(":", 1)[0]
    text = text.replace("/", "-").replace("_", "-")
    while "--" in text:
        text = text.replace("--", "-")
    return text


def as_freqtrade_pair(symbol: str) -> str:
    canonical = canonical_pair(symbol)
    if not canonical:
        return ""
    parts = canonical.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return canonical


def _empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_COLUMNS)


def read_signal_frame(path: Path | str = DEFAULT_SIGNAL_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return _empty_signal_frame()

    frame = pd.read_parquet(path)
    if frame.empty:
        return _empty_signal_frame()

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["canonical_pair"] = frame["canonical_pair"].map(canonical_pair)
    frame["freqtrade_pair"] = frame["canonical_pair"].map(as_freqtrade_pair)
    return frame.sort_values(["canonical_pair", "date"]).reset_index(drop=True)


def write_signal_frame(frame: pd.DataFrame, path: Path | str = DEFAULT_SIGNAL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        _empty_signal_frame().to_parquet(path, index=False)
        return path

    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], utc=True)
    output["canonical_pair"] = output["symbol"].map(canonical_pair)
    output["freqtrade_pair"] = output["canonical_pair"].map(as_freqtrade_pair)
    for column in SIGNAL_COLUMNS:
        if column not in output.columns:
            output[column] = (
                0.0
                if column in {
                    "signal_confidence",
                    "qlib_score",
                    "qlib_score_rank",
                    "stop_loss_pct",
                    "take_profit_pct",
                    "partial_take_profit_pct",
                    "kelly_fraction",
                    "target_leverage",
                    "entry_stake_fraction",
                    "layer_stake_fraction",
                    "reduce_stake_fraction",
                    "rotation_score",
                    "rotation_weight",
                }
                else 1 if column == "max_entry_layers" else 0
            )
    output = output[SIGNAL_COLUMNS].sort_values(["canonical_pair", "date"]).reset_index(drop=True)
    output.to_parquet(path, index=False)
    return path


def to_external_signal_frame(feature_frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if feature_frame.empty:
        return _empty_signal_frame()

    frame = feature_frame.copy()
    frame["date"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = symbol
    frame["canonical_pair"] = canonical_pair(symbol)
    frame["freqtrade_pair"] = as_freqtrade_pair(symbol)
    defaults = {
        "entry_long_signal": 0,
        "exit_long_signal": 0,
        "add_position_signal": 0,
        "reduce_position_signal": 0,
        "signal_confidence": 0.0,
        "qlib_score": 0.0,
        "qlib_score_rank": 0.5,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "partial_take_profit_pct": 0.02,
        "kelly_fraction": 0.01,
        "target_leverage": 1.0,
        "entry_stake_fraction": 0.25,
        "layer_stake_fraction": 0.10,
        "reduce_stake_fraction": 0.33,
        "max_entry_layers": 1,
        "rotation_score": 0.0,
        "rotation_rank": 0,
        "rotation_weight": 0.0,
        "rotation_entry_long_signal": 0,
    }
    for key, value in defaults.items():
        if key not in frame.columns:
            frame[key] = value
    return frame[SIGNAL_COLUMNS].sort_values("date").reset_index(drop=True)
