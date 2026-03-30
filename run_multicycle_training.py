from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bridge.signal_store import to_external_signal_frame, write_signal_frame
from quant_trading.backtest.engine import run_backtest
from quant_trading.config import DEFAULT_PARAMETER_NOTES, ResearchConfig
from quant_trading.data import FreqtradeHistoryLoader, MarketDataSourceConfig, OKXDataFetcher, load_market_frames
from quant_trading.signals.multi_timeframe import build_multitimeframe_signal_frame


def _slug(symbol: str) -> str:
    return symbol.lower().replace("-", "_").replace("/", "_")


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-cycle crypto trend-following research pipeline.")
    parser.add_argument("--symbols", nargs="*", default=["BTC-USDT", "ETH-USDT", "DOGE-USDT", "PEPE-USDT"])
    parser.add_argument("--lower-limit", type=int, default=2400)
    parser.add_argument("--higher-limit", type=int, default=1200)
    parser.add_argument("--source", choices=["auto", "freqtrade", "okx"], default="auto")
    parser.add_argument("--freqtrade-datadir", default="freqtrade-develop/user_data/data")
    parser.add_argument("--freqtrade-exchange", default="okx")
    parser.add_argument(
        "--freqtrade-format",
        choices=["auto", "feather", "parquet", "json", "jsongz"],
        default="auto",
    )
    parser.add_argument("--freqtrade-candle-type", default="spot")
    args = parser.parse_args()

    config = ResearchConfig()
    config.data.symbols = tuple(args.symbols)
    config.data.lower_limit = args.lower_limit
    config.data.higher_limit = args.higher_limit

    runtime_root = Path("bridge/runtime")
    raw_dir = runtime_root / "raw"
    feature_dir = runtime_root / "features"
    report_dir = runtime_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    okx_fetcher = OKXDataFetcher(page_size=config.data.page_size)
    freqtrade_loader = FreqtradeHistoryLoader(
        datadir=args.freqtrade_datadir,
        exchange=args.freqtrade_exchange,
        data_format=args.freqtrade_format,
        candle_type=args.freqtrade_candle_type,
    )
    source_config = MarketDataSourceConfig(
        source=args.source,
        freqtrade_datadir=args.freqtrade_datadir,
        freqtrade_exchange=args.freqtrade_exchange,
        freqtrade_format=args.freqtrade_format,
        freqtrade_candle_type=args.freqtrade_candle_type,
    )
    external_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for symbol in config.data.symbols:
        lower_frame, higher_frame, data_source = load_market_frames(
            symbol=symbol,
            config=config,
            source_config=source_config,
            okx_fetcher=okx_fetcher,
            freqtrade_loader=freqtrade_loader,
        )
        if lower_frame.empty or higher_frame.empty:
            raise RuntimeError(f"The selected market data source returned empty data for {symbol}")

        _save_frame(lower_frame, raw_dir / f"{_slug(symbol)}_{config.data.lower_bar}.parquet")
        _save_frame(higher_frame, raw_dir / f"{_slug(symbol)}_{config.data.higher_bar}.parquet")

        signal_frame = build_multitimeframe_signal_frame(
            lower_frame,
            higher_frame,
            feature_config=config.feature,
            signal_config=config.signal,
            risk_config=config.risk,
        )
        signal_frame["symbol"] = symbol
        _save_frame(signal_frame, feature_dir / f"{_slug(symbol)}_signals.parquet")

        external_frame = to_external_signal_frame(signal_frame, symbol=symbol)
        external_frames.append(external_frame)

        backtest_result = run_backtest(
            signal_frame,
            backtest_config=config.backtest,
            risk_config=config.risk,
        )
        trades = backtest_result["trades"]
        trades.to_csv(report_dir / f"{_slug(symbol)}_trades.csv", index=False)

        summary_row = {
            "symbol": symbol,
            "data_source": data_source,
            "entry_signals": int(external_frame["entry_long_signal"].sum()),
            **backtest_result["summary"],
        }
        summaries.append(summary_row)

    combined_signals = pd.concat(external_frames, ignore_index=True) if external_frames else pd.DataFrame()
    write_signal_frame(combined_signals)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(report_dir / "backtest_summary.csv", index=False)

    json_payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "requested_source": args.source,
        "config": config.to_dict(),
        "parameter_notes": DEFAULT_PARAMETER_NOTES,
        "results": summaries,
    }
    (report_dir / "backtest_summary.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    print(json.dumps(json_payload, indent=2))


if __name__ == "__main__":
    main()
