from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
import requests
from requests import RequestException

OKX_HISTORY_ENDPOINT = "https://www.okx.com/api/v5/market/history-candles"


@dataclass(slots=True)
class OKXDataFetcher:
    page_size: int = 300
    pause_seconds: float = 0.11
    timeout_seconds: float = 40.0
    max_retries: int = 4
    session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "qlib-ai-trade/1.0"})

    def fetch_history(self, inst_id: str, bar: str, limit: int | None) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        after: str | None = None
        fetch_all = limit is None or limit <= 0
        remaining = None if fetch_all else int(limit)
        previous_oldest_open_time_ms: int | None = None

        while fetch_all or (remaining is not None and remaining > 0):
            batch_limit = self.page_size if fetch_all else min(self.page_size, remaining)
            params = {"instId": inst_id, "bar": bar, "limit": str(batch_limit)}
            if after is not None:
                params["after"] = after

            payload = self._request_payload(params)
            if payload.get("code") != "0":
                raise RuntimeError(f"OKX returned error payload: {payload}")

            rows = payload.get("data", [])
            if not rows:
                break

            batch = self._parse_rows(rows=rows, inst_id=inst_id, bar=bar)
            frames.append(batch)
            if remaining is not None:
                remaining -= len(batch)

            oldest_open_time_ms = int(batch["open_time_ms"].min())
            if previous_oldest_open_time_ms is not None and oldest_open_time_ms >= previous_oldest_open_time_ms:
                break
            previous_oldest_open_time_ms = oldest_open_time_ms
            after = str(oldest_open_time_ms - 1)

            if len(rows) < batch_limit:
                break
            time.sleep(self.pause_seconds)

        if not frames:
            return pd.DataFrame(
                columns=["open_time_ms", "timestamp", "open", "high", "low", "close", "vol", "confirm", "inst_id", "bar"]
            )

        frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        if not fetch_all and limit is not None and limit > 0:
            frame = frame.tail(limit)
        frame = frame.reset_index(drop=True)
        return frame

    def _request_payload(self, params: dict[str, str]) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(OKX_HISTORY_ENDPOINT, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.json()
            except RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.pause_seconds * attempt * 3.0)
        raise RuntimeError(f"OKX request failed after {self.max_retries} attempts for params={params}") from last_error

    @staticmethod
    def _parse_rows(rows: list[list[str]], inst_id: str, bar: str) -> pd.DataFrame:
        parsed_rows = []
        for row in rows:
            open_time_ms = int(row[0])
            parsed_rows.append(
                {
                    "open_time_ms": open_time_ms,
                    "timestamp": pd.to_datetime(open_time_ms, unit="ms", utc=True),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "vol": float(row[5]),
                    "confirm": int(row[8]) if len(row) > 8 else 1,
                    "inst_id": inst_id,
                    "bar": bar,
                }
            )
        return pd.DataFrame(parsed_rows)
