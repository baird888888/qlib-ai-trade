from __future__ import annotations

from pathlib import Path

import fire
import pandas as pd


def normalize_ohlcv(frame: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "date": "date",
            "timestamp": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "vol": "volume",
        }
    ).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    renamed["date"] = pd.to_datetime(renamed["date"], utc=True)
    renamed["symbol"] = symbol or renamed.get("symbol", "UNKNOWN")
    output = renamed[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
    return output.sort_values("date").reset_index(drop=True)


def convert_file(input_path: str, output_dir: str, symbol: str | None = None) -> str:
    source = Path(input_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source)

    normalized = normalize_ohlcv(frame, symbol=symbol)
    output_symbol = symbol or normalized["symbol"].iloc[0]
    output_path = target_dir / f"{output_symbol.lower().replace('-', '_')}.csv"
    normalized.to_csv(output_path, index=False)
    return str(output_path)


if __name__ == "__main__":
    fire.Fire({"convert": convert_file})
