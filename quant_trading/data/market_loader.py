from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bridge.signal_store import as_freqtrade_pair
from quant_trading.config import ResearchConfig
from quant_trading.data.freqtrade_loader import FreqtradeHistoryLoader
from quant_trading.data.okx_fetcher import OKXDataFetcher


@dataclass(slots=True)
class MarketDataSourceConfig:
    source: str = "auto"
    freqtrade_datadir: str = "freqtrade-develop/user_data/data"
    freqtrade_exchange: str = "okx"
    freqtrade_format: str = "auto"
    freqtrade_candle_type: str = "spot"


def symbol_to_freqtrade_pair(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    return as_freqtrade_pair(symbol)


def load_market_frames(
    symbol: str,
    config: ResearchConfig,
    source_config: MarketDataSourceConfig,
    okx_fetcher: OKXDataFetcher | None = None,
    freqtrade_loader: FreqtradeHistoryLoader | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    okx_fetcher = okx_fetcher or OKXDataFetcher(page_size=config.data.page_size)
    freqtrade_loader = freqtrade_loader or FreqtradeHistoryLoader(
        datadir=source_config.freqtrade_datadir,
        exchange=source_config.freqtrade_exchange,
        data_format=source_config.freqtrade_format,
        candle_type=source_config.freqtrade_candle_type,
    )

    if source_config.source in {"auto", "freqtrade"}:
        pair = symbol_to_freqtrade_pair(symbol)
        try:
            lower_frame = freqtrade_loader.load_history(pair, config.data.lower_bar, limit=config.data.lower_limit)
            higher_frame = freqtrade_loader.load_history(pair, config.data.higher_bar, limit=config.data.higher_limit)
            return lower_frame, higher_frame, "freqtrade"
        except FileNotFoundError:
            if source_config.source == "freqtrade":
                raise

    if source_config.source in {"auto", "okx"}:
        lower_frame = okx_fetcher.fetch_history(symbol, config.data.lower_bar, config.data.lower_limit)
        higher_frame = okx_fetcher.fetch_history(symbol, config.data.higher_bar, config.data.higher_limit)
        return lower_frame, higher_frame, "okx"

    raise ValueError(f"Unsupported data source: {source_config.source}")
