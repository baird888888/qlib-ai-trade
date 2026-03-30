from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATAFRAME_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
SUPPORTED_FREQTRADE_FORMATS = ("feather", "parquet", "json", "jsongz")


def pair_to_filename(pair: str) -> str:
    normalized = str(pair)
    for character in ["/", " ", ".", "@", "$", "+", ":"]:
        normalized = normalized.replace(character, "_")
    return normalized


def timeframe_to_filename(timeframe: str) -> str:
    return str(timeframe).replace("M", "Mo")


def _candidate_roots(datadir: Path | str, exchange: str | None, candle_type: str) -> list[Path]:
    base_dir = Path(datadir)
    roots = [base_dir]
    if exchange and base_dir.name.lower() != exchange.lower():
        roots.insert(0, base_dir / exchange)

    suffix = [] if candle_type == "spot" else ["futures"]
    resolved: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        candidate = root.joinpath(*suffix)
        if candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)
    return resolved


def _normalize_freqtrade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=DEFAULT_DATAFRAME_COLUMNS)

    output = frame.copy()
    if set(DEFAULT_DATAFRAME_COLUMNS).issubset(output.columns):
        output = output[DEFAULT_DATAFRAME_COLUMNS]
    else:
        output.columns = DEFAULT_DATAFRAME_COLUMNS

    output = output.astype(
        {
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float",
        }
    )
    if pd.api.types.is_numeric_dtype(output["date"]):
        output["date"] = pd.to_datetime(output["date"], utc=True, unit="ms")
    else:
        output["date"] = pd.to_datetime(output["date"], utc=True)
    return output


def _read_freqtrade_frame(path: Path, data_format: str) -> pd.DataFrame:
    if data_format == "parquet":
        frame = pd.read_parquet(path)
    elif data_format == "feather":
        frame = pd.read_feather(path)
    elif data_format == "json":
        frame = pd.read_json(path, orient="values")
    elif data_format == "jsongz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            frame = pd.read_json(handle, orient="values")
    else:
        raise ValueError(f"Unsupported Freqtrade data format: {data_format}")
    return _normalize_freqtrade_frame(frame)


@dataclass(slots=True)
class FreqtradeHistoryLoader:
    datadir: Path | str
    exchange: str | None = None
    data_format: str = "auto"
    candle_type: str = "spot"

    def resolve_ohlcv_path(self, pair: str, timeframe: str) -> tuple[Path, str]:
        formats = SUPPORTED_FREQTRADE_FORMATS if self.data_format == "auto" else (self.data_format,)
        pair_slug = pair_to_filename(pair)
        timeframe_slug = timeframe_to_filename(timeframe)
        candle_suffix = "" if self.candle_type == "spot" else f"-{self.candle_type}"
        timeframe_candidates = [timeframe_slug]
        if timeframe_slug != timeframe:
            timeframe_candidates.append(str(timeframe))

        for root in _candidate_roots(self.datadir, self.exchange, self.candle_type):
            for timeframe_candidate in timeframe_candidates:
                for data_format in formats:
                    extension = "json.gz" if data_format == "jsongz" else data_format
                    path = root / f"{pair_slug}-{timeframe_candidate}{candle_suffix}.{extension}"
                    if path.exists():
                        return path, data_format

        searched_roots = ", ".join(str(path) for path in _candidate_roots(self.datadir, self.exchange, self.candle_type))
        raise FileNotFoundError(
            f"No Freqtrade OHLCV file found for {pair} {timeframe}. "
            f"Searched roots: {searched_roots}"
        )

    def load_history(self, pair: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        path, resolved_format = self.resolve_ohlcv_path(pair=pair, timeframe=timeframe)
        frame = _read_freqtrade_frame(path, resolved_format)
        if limit is not None:
            frame = frame.tail(limit).reset_index(drop=True)

        output = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(frame["date"], utc=True),
                "open": frame["open"],
                "high": frame["high"],
                "low": frame["low"],
                "close": frame["close"],
                "vol": frame["volume"],
                "confirm": 1,
                "inst_id": pair,
                "bar": timeframe,
            }
        )
        return output.reset_index(drop=True)
