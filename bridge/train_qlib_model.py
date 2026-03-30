from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from bridge.signal_store import to_external_signal_frame, write_signal_frame
from quant_trading.backtest.engine import run_backtest
from quant_trading.config import DEFAULT_PARAMETER_NOTES, ResearchConfig
from quant_trading.data import FreqtradeHistoryLoader, MarketDataSourceConfig, OKXDataFetcher, load_market_frames
from quant_trading.signals.multi_timeframe import apply_qlib_score_filter, build_multitimeframe_signal_frame

PHASE_CODE_MAP = {
    "markdown": -2,
    "distribution": -1,
    "neutral": 0,
    "accumulation": 1,
    "markup": 2,
}

EXCLUDED_MODEL_COLUMNS = {
    "timestamp",
    "symbol",
    "inst_id_x",
    "bar_x",
    "inst_id_y",
    "bar_y",
    "wyckoff_phase",
    "h1_wyckoff_phase",
    "confirm",
    "h1_confirm",
    "entry_long_signal",
    "exit_long_signal",
    "stop_loss_pct",
    "take_profit_pct",
    "kelly_fraction",
    "heuristic_entry_long_signal",
    "heuristic_exit_long_signal",
    "qlib_score",
    "qlib_score_rank",
    "qlib_entry_threshold",
    "qlib_exit_threshold",
    "future_return",
    "forward_return",
    "open_time_ms",
    "h1_open_time_ms",
}

QLIB_PARAMETER_NOTES = {
    "prediction_horizon": "Future 5-minute bars used to define LABEL0 as a forward return target.",
    "test_days": "Minimum out-of-sample test duration measured in calendar days.",
    "max_test_window": "When enabled, the script expands the test segment to the largest window that still leaves the requested validation and minimum training spans.",
    "valid_days": "Validation window size immediately before the test window.",
    "min_train_days": "Minimum in-sample training duration required before validation begins.",
    "entry_quantile": "Validation-score quantile that a heuristic long signal must exceed to open a trade.",
    "exit_quantile": "Validation-score quantile below which the model forces an exit or refuses to hold.",
    "num_boost_round": "Maximum LightGBM trees used by Qlib's LGBModel.",
    "early_stopping_rounds": "Validation patience before Qlib stops adding new trees.",
}


def _slug(symbol: str) -> str:
    return symbol.lower().replace("-", "_").replace("/", "_")


def _save_frame(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=index)


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _timestamp_to_utc(value: pd.Series) -> pd.Series:
    timestamp = pd.to_datetime(value, utc=True)
    return timestamp


def _timestamp_to_qlib(value: pd.Series) -> pd.Series:
    timestamp = _timestamp_to_utc(value)
    return timestamp.dt.tz_convert("UTC").dt.tz_localize(None)


def _inject_local_qlib_source() -> Path | None:
    local_qlib_root = PROJECT_ROOT / "qlib-main"
    if local_qlib_root.exists() and str(local_qlib_root) not in sys.path:
        sys.path.insert(0, str(local_qlib_root))
        return local_qlib_root
    return local_qlib_root if local_qlib_root.exists() else None


def _init_qlib(runtime_root: Path, experiment_name: str):
    os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")
    local_qlib_root = _inject_local_qlib_source()
    try:
        import qlib
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.workflow import R
    except Exception as exc:  # pragma: no cover - environment-specific import path
        message = str(exc)
        if "No module named 'lightgbm'" in message:
            raise RuntimeError(
                "qlib was found, but LightGBM is missing from the active environment. "
                "Install `requirements-quant.txt` into `.venv` before retrying."
            ) from exc
        raise RuntimeError(
            "qlib import failed before training could start. Make sure the active environment can "
            "see qlib either from the local source tree"
            f"{f' at {local_qlib_root}' if local_qlib_root else ''} or from pip."
        ) from exc

    provider_dir = runtime_root / "qlib_provider_stub"
    mlruns_dir = runtime_root / "qlib_mlruns"
    provider_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    qlib.init(
        provider_uri=str(provider_dir.resolve()),
        region="cn",
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": mlruns_dir.resolve().as_uri(),
                "default_exp_name": experiment_name,
            },
        },
    )
    return DatasetH, DataHandlerLP, LGBModel, R


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        casted = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(casted) or math.isinf(casted):
        return 0.0
    return casted


def _optimize_numeric_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    optimized = frame.copy()
    float_columns = optimized.select_dtypes(include=["float64"]).columns
    int_columns = optimized.select_dtypes(include=["int64"]).columns
    for column in float_columns:
        optimized[column] = pd.to_numeric(optimized[column], downcast="float")
    for column in int_columns:
        optimized[column] = pd.to_numeric(optimized[column], downcast="integer")
    return optimized


def _prepare_symbol_training_frame(signal_frame: pd.DataFrame, symbol: str, prediction_horizon: int) -> pd.DataFrame:
    prepared = signal_frame.copy()
    prepared["symbol"] = symbol
    prepared["timestamp"] = _timestamp_to_utc(prepared["timestamp"])
    prepared["wyckoff_phase_code"] = prepared["wyckoff_phase"].map(PHASE_CODE_MAP).fillna(0).astype(int)
    prepared["h1_wyckoff_phase_code"] = prepared["h1_wyckoff_phase"].map(PHASE_CODE_MAP).fillna(0).astype(int)
    prepared["forward_return"] = prepared["close"].shift(-prediction_horizon) / prepared["close"] - 1.0
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    prepared = prepared.dropna(
        subset=[
            "forward_return",
            "ema_slow",
            "rolling_high",
            "h1_close",
            "h1_ema_slow",
        ]
    ).reset_index(drop=True)
    return prepared


def _prepare_symbol_inference_frame(signal_frame: pd.DataFrame, symbol: str, prediction_horizon: int) -> pd.DataFrame:
    prepared = signal_frame.copy()
    prepared["symbol"] = symbol
    prepared["timestamp"] = _timestamp_to_utc(prepared["timestamp"])
    prepared["wyckoff_phase_code"] = prepared["wyckoff_phase"].map(PHASE_CODE_MAP).fillna(0).astype(int)
    prepared["h1_wyckoff_phase_code"] = prepared["h1_wyckoff_phase"].map(PHASE_CODE_MAP).fillna(0).astype(int)
    prepared["forward_return"] = prepared["close"].shift(-prediction_horizon) / prepared["close"] - 1.0
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    prepared = prepared.dropna(
        subset=[
            "ema_slow",
            "rolling_high",
            "h1_close",
            "h1_ema_slow",
        ]
    ).reset_index(drop=True)
    return prepared


def _select_model_features(frame: pd.DataFrame) -> list[str]:
    feature_names: list[str] = []
    for column in frame.columns:
        if column in EXCLUDED_MODEL_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        feature_names.append(column)
    return sorted(feature_names)


def _build_qlib_frame(feature_frame: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    qlib_ready = feature_frame.copy()
    qlib_ready["datetime"] = _timestamp_to_qlib(qlib_ready["timestamp"])
    qlib_ready["instrument"] = qlib_ready["symbol"].astype(str)

    index = pd.MultiIndex.from_arrays(
        [qlib_ready["datetime"], qlib_ready["instrument"]],
        names=("datetime", "instrument"),
    )

    feature_block = qlib_ready[feature_names].astype(np.float32)
    label_block = qlib_ready[["forward_return"]].rename(columns={"forward_return": "LABEL0"}).astype(np.float32)
    labels = label_block["LABEL0"].copy()
    labels.index = index

    feature_block.columns = pd.MultiIndex.from_product([["feature"], feature_block.columns])
    label_block.columns = pd.MultiIndex.from_product([["label"], label_block.columns])

    qlib_frame = pd.concat([feature_block, label_block], axis=1)
    qlib_frame.index = index
    qlib_frame = qlib_frame.sort_index()
    labels = labels.sort_index()
    return qlib_frame, labels


def _build_duration_segments(
    datetimes: pd.Series,
    test_days: int,
    valid_days: int,
    min_train_days: int,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    unique_times = pd.Index(sorted(_timestamp_to_qlib(pd.Series(datetimes)).dropna().unique()))
    if len(unique_times) < 30:
        raise ValueError("Not enough timestamps to create train/valid/test splits. Increase candle limits.")

    start_time = pd.Timestamp(unique_times[0])
    end_time = pd.Timestamp(unique_times[-1])
    test_target = end_time - pd.Timedelta(days=test_days)
    test_start_pos = int(max(unique_times.searchsorted(test_target, side="right") - 1, 0))
    test_start = pd.Timestamp(unique_times[test_start_pos])

    if (end_time - test_start) < pd.Timedelta(days=test_days):
        raise ValueError(
            f"Fetched history only covers {(end_time - start_time).days} days, which is insufficient for a {test_days}-day test window."
        )

    valid_target = test_start - pd.Timedelta(days=valid_days)
    valid_start_pos = int(max(unique_times.searchsorted(valid_target, side="right") - 1, 0))
    valid_start = pd.Timestamp(unique_times[valid_start_pos])
    valid_end_pos = int(max(test_start_pos - 1, valid_start_pos))
    valid_end = pd.Timestamp(unique_times[valid_end_pos])

    train_end_pos = int(valid_start_pos - 1)
    if train_end_pos <= 0:
        raise ValueError("No room left for training data before the validation window. Fetch more history from OKX.")
    train_end = pd.Timestamp(unique_times[train_end_pos])

    if (train_end - start_time) < pd.Timedelta(days=min_train_days):
        raise ValueError(
            f"Training span would be only {(train_end - start_time).days} days, below the required minimum of {min_train_days} days."
        )

    if valid_end < valid_start:
        raise ValueError("Validation window is empty. Fetch more history or reduce --valid-days.")

    return {
        "train": (start_time, train_end),
        "valid": (valid_start, valid_end),
        "test": (test_start, end_time),
    }


def _maximum_feasible_test_days(
    datetimes: pd.Series,
    valid_days: int,
    min_train_days: int,
    minimum_test_days: int,
) -> int:
    unique_times = pd.Index(sorted(_timestamp_to_qlib(pd.Series(datetimes)).dropna().unique()))
    if len(unique_times) < 30:
        raise ValueError("Not enough timestamps to create train/valid/test splits. Increase candle limits.")

    total_span_days = int((pd.Timestamp(unique_times[-1]) - pd.Timestamp(unique_times[0])) / pd.Timedelta(days=1))
    low = minimum_test_days
    high = total_span_days
    best = None

    while low <= high:
        midpoint = (low + high) // 2
        try:
            _build_duration_segments(
                datetimes=datetimes,
                test_days=midpoint,
                valid_days=valid_days,
                min_train_days=min_train_days,
            )
        except ValueError:
            high = midpoint - 1
            continue
        best = midpoint
        low = midpoint + 1

    if best is None:
        raise ValueError(
            "Unable to allocate the requested minimum two-year test span while preserving the validation and training windows."
        )
    return int(best)


def _describe_symbol_history(frame: pd.DataFrame) -> dict[str, object]:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    if timestamp.empty:
        return {"start": None, "end": None, "bars": 0, "days": 0.0}
    span = timestamp.max() - timestamp.min()
    return {
        "start": timestamp.min(),
        "end": timestamp.max(),
        "bars": int(len(frame)),
        "days": round(span / timedelta(days=1), 2),
    }


def _prediction_series_to_frame(predictions: pd.Series, segment: str) -> pd.DataFrame:
    frame = predictions.rename("qlib_score").reset_index()
    if "datetime" not in frame.columns or "instrument" not in frame.columns:
        raise RuntimeError("Unexpected Qlib prediction index layout.")
    frame["timestamp"] = pd.to_datetime(frame["datetime"], utc=True)
    frame["symbol"] = frame["instrument"].astype(str)
    frame["segment"] = segment
    return frame[["timestamp", "symbol", "segment", "qlib_score"]].sort_values(["symbol", "timestamp"])


def _annotate_prediction_segments(
    prediction_frame: pd.DataFrame,
    segments: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    annotated = prediction_frame.copy()
    annotated["segment"] = "live"
    for segment_name, (start, end) in segments.items():
        start_utc = pd.to_datetime(start, utc=True)
        end_utc = pd.to_datetime(end, utc=True)
        mask = (annotated["timestamp"] >= start_utc) & (annotated["timestamp"] <= end_utc)
        annotated.loc[mask, "segment"] = segment_name
    return annotated


def _evaluate_predictions(label_series: pd.Series, predictions: pd.Series) -> dict[str, float]:
    evaluation = pd.concat(
        [predictions.rename("prediction"), label_series.rename("label")],
        axis=1,
        join="inner",
    ).dropna()
    if evaluation.empty:
        return {
            "samples": 0,
            "rmse": 0.0,
            "mae": 0.0,
            "pearson_ic": 0.0,
            "spearman_ic": 0.0,
            "directional_accuracy": 0.0,
            "positive_prediction_hit_rate": 0.0,
        }

    positive_mask = evaluation["prediction"] > 0
    positive_hit_rate = (
        float((evaluation.loc[positive_mask, "label"] > 0).mean())
        if positive_mask.any()
        else 0.0
    )
    return {
        "samples": int(len(evaluation)),
        "rmse": _safe_float(np.sqrt(np.mean((evaluation["prediction"] - evaluation["label"]) ** 2))),
        "mae": _safe_float(np.mean(np.abs(evaluation["prediction"] - evaluation["label"]))),
        "pearson_ic": _safe_float(evaluation["prediction"].corr(evaluation["label"], method="pearson")),
        "spearman_ic": _safe_float(evaluation["prediction"].corr(evaluation["label"], method="spearman")),
        "directional_accuracy": _safe_float(
            (np.sign(evaluation["prediction"]) == np.sign(evaluation["label"])).mean()
        ),
        "positive_prediction_hit_rate": positive_hit_rate,
    }


def _quantile_or_default(series: pd.Series, quantile: float, default: float = 0.0) -> float:
    clean = pd.Series(series).dropna()
    if clean.empty:
        return default
    return _safe_float(clean.quantile(quantile))


def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Qlib LightGBM model on the multi-timeframe crypto feature stack and backtest the filtered signals."
    )
    parser.add_argument("--symbols", nargs="*", default=["BTC-USDT", "ETH-USDT", "DOGE-USDT", "PEPE-USDT"])
    parser.add_argument("--lower-limit", type=int, default=0)
    parser.add_argument("--higher-limit", type=int, default=0)
    parser.add_argument("--source", choices=["auto", "freqtrade", "okx"], default="auto")
    parser.add_argument("--reuse-raw-cache", action="store_true")
    parser.add_argument("--freqtrade-datadir", default="freqtrade-develop/user_data/data")
    parser.add_argument("--freqtrade-exchange", default="okx")
    parser.add_argument(
        "--freqtrade-format",
        choices=["auto", "feather", "parquet", "json", "jsongz"],
        default="auto",
    )
    parser.add_argument("--freqtrade-candle-type", default="spot")
    parser.add_argument("--prediction-horizon", type=int, default=12)
    parser.add_argument("--test-days", type=int, default=730)
    parser.add_argument("--max-test-window", action="store_true")
    parser.add_argument("--valid-days", type=int, default=180)
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--entry-quantile", type=float, default=0.70)
    parser.add_argument("--exit-quantile", type=float, default=0.45)
    parser.add_argument("--entry-threshold", type=float, default=None)
    parser.add_argument("--exit-threshold", type=float, default=None)
    parser.add_argument("--confidence-weight", type=float, default=0.35)
    parser.add_argument("--experiment-name", default="multicycle_qlib")
    parser.add_argument("--num-boost-round", type=int, default=400)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--min-data-in-leaf", type=int, default=50)
    parser.add_argument("--verbose-eval", type=int, default=50)
    args = parser.parse_args()

    if args.prediction_horizon <= 0:
        raise ValueError("--prediction-horizon must be positive.")
    if not args.max_test_window and args.test_days < 730:
        raise ValueError("--test-days must be at least 730 to satisfy the requested minimum two-year test window.")
    if args.valid_days <= 0 or args.min_train_days <= 0:
        raise ValueError("--valid-days and --min-train-days must be positive.")
    if not (0.0 <= args.entry_quantile <= 1.0 and 0.0 <= args.exit_quantile <= 1.0):
        raise ValueError("Quantiles must be between 0 and 1.")

    config = ResearchConfig()
    config.data.symbols = tuple(args.symbols)
    config.data.lower_limit = args.lower_limit
    config.data.higher_limit = args.higher_limit

    runtime_root = PROJECT_ROOT / "bridge" / "runtime"
    raw_dir = runtime_root / "raw"
    feature_dir = runtime_root / "features"
    model_dir = runtime_root / "models"
    qlib_dir = runtime_root / "qlib"
    report_dir = runtime_root / "reports"
    for directory in [raw_dir, feature_dir, model_dir, qlib_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

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

    symbol_frames: dict[str, pd.DataFrame] = {}
    source_map: dict[str, str] = {}
    history_ranges: dict[str, dict[str, object]] = {}
    combined_training_parts: list[pd.DataFrame] = []
    combined_inference_parts: list[pd.DataFrame] = []

    for symbol in config.data.symbols:
        cached_lower_path = raw_dir / f"{_slug(symbol)}_{config.data.lower_bar}.parquet"
        cached_higher_path = raw_dir / f"{_slug(symbol)}_{config.data.higher_bar}.parquet"
        use_cached_okx = (
            args.reuse_raw_cache
            and args.source == "okx"
            and cached_lower_path.exists()
            and cached_higher_path.exists()
        )

        if use_cached_okx:
            lower_frame = _load_frame(cached_lower_path)
            higher_frame = _load_frame(cached_higher_path)
            data_source = "okx_cache"
        else:
            lower_frame, higher_frame, data_source = load_market_frames(
                symbol=symbol,
                config=config,
                source_config=source_config,
                okx_fetcher=okx_fetcher,
                freqtrade_loader=freqtrade_loader,
            )
        if lower_frame.empty or higher_frame.empty:
            raise RuntimeError(f"The selected market data source returned empty data for {symbol}")

        _save_frame(lower_frame, cached_lower_path)
        _save_frame(higher_frame, cached_higher_path)

        signal_frame = build_multitimeframe_signal_frame(
            lower_frame,
            higher_frame,
            feature_config=config.feature,
            signal_config=config.signal,
            risk_config=config.risk,
        )
        signal_frame = _optimize_numeric_dtypes(signal_frame)
        signal_frame["symbol"] = symbol
        symbol_frames[symbol] = signal_frame.copy()
        source_map[symbol] = data_source
        history_ranges[symbol] = {
            "lower_bar": _describe_symbol_history(lower_frame),
            "higher_bar": _describe_symbol_history(higher_frame),
        }
        combined_training_parts.append(
            _prepare_symbol_training_frame(
                signal_frame=signal_frame,
                symbol=symbol,
                prediction_horizon=args.prediction_horizon,
            )
        )
        combined_inference_parts.append(
            _prepare_symbol_inference_frame(
                signal_frame=signal_frame,
                symbol=symbol,
                prediction_horizon=args.prediction_horizon,
            )
        )

    combined_training_frame = (
        pd.concat(combined_training_parts, ignore_index=True)
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )
    combined_inference_frame = (
        pd.concat(combined_inference_parts, ignore_index=True)
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )
    feature_names = _select_model_features(combined_training_frame)
    if not feature_names:
        raise RuntimeError("No numeric feature columns were available for Qlib training.")

    qlib_frame, label_series = _build_qlib_frame(combined_training_frame, feature_names)
    inference_qlib_frame, _ = _build_qlib_frame(combined_inference_frame, feature_names)
    _save_frame(qlib_frame, qlib_dir / "multicycle_dataset.parquet", index=True)
    _save_frame(inference_qlib_frame, qlib_dir / "multicycle_inference_dataset.parquet", index=True)
    effective_test_days = (
        _maximum_feasible_test_days(
            datetimes=combined_training_frame["timestamp"],
            valid_days=args.valid_days,
            min_train_days=args.min_train_days,
            minimum_test_days=730,
        )
        if args.max_test_window
        else args.test_days
    )
    segments = _build_duration_segments(
        datetimes=combined_training_frame["timestamp"],
        test_days=effective_test_days,
        valid_days=args.valid_days,
        min_train_days=args.min_train_days,
    )

    DatasetH, DataHandlerLP, LGBModel, R = _init_qlib(runtime_root=runtime_root, experiment_name=args.experiment_name)
    handler = DataHandlerLP.from_df(qlib_frame)
    dataset = DatasetH(handler=handler, segments=segments)
    inference_handler = DataHandlerLP.from_df(inference_qlib_frame)
    inference_segments = {
        "all": (
            _timestamp_to_qlib(combined_inference_frame["timestamp"]).min(),
            _timestamp_to_qlib(combined_inference_frame["timestamp"]).max(),
        )
    }
    inference_dataset = DatasetH(handler=inference_handler, segments=inference_segments)
    model = LGBModel(
        loss="mse",
        early_stopping_rounds=args.early_stopping_rounds,
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        min_data_in_leaf=args.min_data_in_leaf,
    )

    recorder_id = ""
    with R.start(experiment_name=args.experiment_name, recorder_name="multicycle_lgb_run"):
        R.log_params(
            symbols=",".join(config.data.symbols),
            lower_limit=args.lower_limit,
            higher_limit=args.higher_limit,
            prediction_horizon=args.prediction_horizon,
            test_days=effective_test_days,
            requested_test_days=args.test_days,
            max_test_window=args.max_test_window,
            valid_days=args.valid_days,
            min_train_days=args.min_train_days,
            entry_quantile=args.entry_quantile,
            exit_quantile=args.exit_quantile,
            num_features=len(feature_names),
            data_source=args.source,
        )
        model.fit(dataset, verbose_eval=args.verbose_eval)
        valid_predictions = model.predict(dataset, segment="valid")
        test_predictions = model.predict(dataset, segment="test")
        recorder_id = R.get_recorder().id

    model_path = model_dir / "multicycle_lgb_model.txt"
    model.model.save_model(str(model_path))

    prediction_frame = _prediction_series_to_frame(model.predict(inference_dataset, segment="all"), segment="all")
    prediction_frame = _annotate_prediction_segments(prediction_frame, segments)
    prediction_frame["qlib_score_rank"] = (
        prediction_frame.groupby("symbol")["qlib_score"].rank(method="average", pct=True).fillna(0.5)
    )
    _save_frame(prediction_frame, qlib_dir / "multicycle_predictions.parquet")

    entry_threshold = (
        _safe_float(args.entry_threshold)
        if args.entry_threshold is not None
        else _quantile_or_default(valid_predictions, args.entry_quantile, default=0.0)
    )
    exit_threshold = (
        _safe_float(args.exit_threshold)
        if args.exit_threshold is not None
        else _quantile_or_default(valid_predictions, args.exit_quantile, default=0.0)
    )
    if exit_threshold > entry_threshold:
        raise ValueError("The exit threshold cannot be greater than the entry threshold.")

    model_metrics = {
        "valid": _evaluate_predictions(label_series=label_series, predictions=valid_predictions),
        "test": _evaluate_predictions(label_series=label_series, predictions=test_predictions),
    }

    backtest_rows: list[dict] = []
    external_frames: list[pd.DataFrame] = []
    enhanced_feature_frames: list[pd.DataFrame] = []
    test_start = pd.to_datetime(segments["test"][0], utc=True)
    test_end = pd.to_datetime(segments["test"][1], utc=True)

    for symbol, signal_frame in symbol_frames.items():
        symbol_prediction_frame = prediction_frame.loc[prediction_frame["symbol"] == symbol].copy()
        merged = signal_frame.merge(
            symbol_prediction_frame[["timestamp", "qlib_score", "qlib_score_rank"]],
            on="timestamp",
            how="left",
        )
        filtered = apply_qlib_score_filter(
            merged,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            confidence_weight=args.confidence_weight,
            risk_config=config.risk,
        )
        filtered = _optimize_numeric_dtypes(filtered)
        filtered["symbol"] = symbol
        filtered["data_source"] = source_map[symbol]
        enhanced_feature_frames.append(filtered)
        _save_frame(filtered, feature_dir / f"{_slug(symbol)}_qlib_signals.parquet")

        external_frames.append(to_external_signal_frame(filtered, symbol=symbol))

        test_slice = filtered.loc[
            (pd.to_datetime(filtered["timestamp"], utc=True) >= test_start)
            & (pd.to_datetime(filtered["timestamp"], utc=True) <= test_end)
        ].reset_index(drop=True)
        heuristic_slice = signal_frame.loc[
            (pd.to_datetime(signal_frame["timestamp"], utc=True) >= test_start)
            & (pd.to_datetime(signal_frame["timestamp"], utc=True) <= test_end)
        ].reset_index(drop=True)

        baseline_backtest = run_backtest(
            heuristic_slice,
            backtest_config=config.backtest,
            risk_config=config.risk,
        )
        qlib_backtest = run_backtest(
            test_slice,
            backtest_config=config.backtest,
            risk_config=config.risk,
        )

        baseline_trades = baseline_backtest["trades"]
        qlib_trades = qlib_backtest["trades"]
        baseline_trades.to_csv(report_dir / f"{_slug(symbol)}_baseline_test_trades.csv", index=False)
        qlib_trades.to_csv(report_dir / f"{_slug(symbol)}_qlib_test_trades.csv", index=False)

        backtest_rows.append(
            {
                "symbol": symbol,
                "data_source": source_map[symbol],
                "variant": "baseline_test",
                "entry_signals": int(heuristic_slice["entry_long_signal"].sum()),
                **baseline_backtest["summary"],
            }
        )
        backtest_rows.append(
            {
                "symbol": symbol,
                "data_source": source_map[symbol],
                "variant": "qlib_filtered_test",
                "entry_signals": int(test_slice["entry_long_signal"].sum()),
                **qlib_backtest["summary"],
            }
        )

    write_signal_frame(pd.concat(external_frames, ignore_index=True))

    backtest_summary = pd.DataFrame(backtest_rows)
    backtest_summary.to_csv(report_dir / "qlib_backtest_summary.csv", index=False)
    _save_frame(pd.concat(enhanced_feature_frames, ignore_index=True), qlib_dir / "enhanced_signal_frames.parquet")

    payload = {
        "generated_at": pd.Timestamp.utcnow(),
        "requested_source": args.source,
        "symbols": list(config.data.symbols),
        "data_sources": source_map,
        "history_ranges": history_ranges,
        "segments": {key: [pd.Timestamp(start), pd.Timestamp(end)] for key, (start, end) in segments.items()},
        "requested_test_days": args.test_days,
        "effective_test_days": effective_test_days,
        "max_test_window": args.max_test_window,
        "prediction_horizon": args.prediction_horizon,
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "qlib_parameter_notes": QLIB_PARAMETER_NOTES,
        "parameter_notes": DEFAULT_PARAMETER_NOTES,
        "config": config.to_dict(),
        "model_metrics": model_metrics,
        "backtest_results": backtest_rows,
        "artifacts": {
            "qlib_dataset": str(qlib_dir / "multicycle_dataset.parquet"),
            "predictions": str(qlib_dir / "multicycle_predictions.parquet"),
            "enhanced_signals": str(qlib_dir / "enhanced_signal_frames.parquet"),
            "model": str(model_path),
            "freqtrade_signals": str(PROJECT_ROOT / "bridge" / "runtime" / "freqtrade_signals.parquet"),
            "recorder_id": recorder_id,
        },
    }
    _save_json(payload, report_dir / "qlib_training_summary.json")
    print(json.dumps(payload, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
