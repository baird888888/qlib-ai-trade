from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path

import pandas as pd
from pandas import DataFrame

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from bridge.signal_store import DEFAULT_SIGNAL_PATH, canonical_pair, read_signal_frame
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy


class QlibExternalSignalStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "5m"
    startup_candle_count = 240
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 3

    minimal_roi = {"0": 10.0}
    stoploss = -0.20

    _cached_mtime: float | None = None
    _cached_signals: DataFrame = pd.DataFrame()

    @staticmethod
    def _series_with_default(frame: DataFrame, column: str, default: float | int) -> pd.Series:
        if column in frame.columns:
            return frame[column]
        return pd.Series(default, index=frame.index)

    @classmethod
    def _apply_default_signal_columns(cls, frame: DataFrame) -> DataFrame:
        output = frame.copy()
        column_defaults = {
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
        integer_columns = {
            "entry_long_signal",
            "exit_long_signal",
            "add_position_signal",
            "reduce_position_signal",
            "max_entry_layers",
            "rotation_rank",
            "rotation_entry_long_signal",
        }
        for column, default in column_defaults.items():
            series = cls._series_with_default(output, column, default)
            if column in integer_columns:
                output[column] = pd.to_numeric(series, errors="coerce").fillna(default).astype(int)
            else:
                output[column] = pd.to_numeric(series, errors="coerce").fillna(default)
        return output

    @classmethod
    def _load_signals(cls) -> DataFrame:
        if not DEFAULT_SIGNAL_PATH.exists():
            cls._cached_signals = pd.DataFrame()
            cls._cached_mtime = None
            return cls._cached_signals

        mtime = DEFAULT_SIGNAL_PATH.stat().st_mtime
        if cls._cached_mtime != mtime:
            cls._cached_signals = read_signal_frame(DEFAULT_SIGNAL_PATH)
            cls._cached_mtime = mtime
        return cls._cached_signals

    @staticmethod
    def _stale_threshold() -> pd.Timedelta:
        return pd.Timedelta("5m")

    def _pair_signals(self, pair: str) -> DataFrame:
        pair_key = canonical_pair(pair)
        signal_frame = self._load_signals()
        if signal_frame.empty or not pair_key:
            return pd.DataFrame()
        return signal_frame.loc[signal_frame["canonical_pair"] == pair_key].sort_values("date").reset_index(drop=True)

    def _signal_snapshot(self, pair: str, current_time: datetime) -> pd.Series | None:
        pair_signals = self._pair_signals(pair)
        if pair_signals.empty:
            return None

        snapshot_time = pd.Timestamp(current_time, tz="UTC") if current_time.tzinfo else pd.Timestamp(current_time, tz="UTC")
        dates = pair_signals["date"]
        position = dates.searchsorted(snapshot_time, side="right") - 1
        if position < 0:
            return None
        row = pair_signals.iloc[int(position)]
        if snapshot_time - row["date"] > self._stale_threshold():
            return None
        return row

    @staticmethod
    def _clamp_stake(value: float, min_stake: float | None, max_stake: float) -> float | None:
        stake = min(float(value), float(max_stake))
        if stake <= 0:
            return None
        if min_stake is not None and stake < float(min_stake):
            return None
        return stake

    @staticmethod
    def _mark_adjustment(trade: Trade, candle_time: pd.Timestamp) -> None:
        trade.set_custom_data("last_adjustment_candle", candle_time.isoformat())

    @staticmethod
    def _last_adjustment_matches(trade: Trade, candle_time: pd.Timestamp) -> bool:
        return trade.get_custom_data("last_adjustment_candle") == candle_time.isoformat()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        output = dataframe.copy()
        output["date"] = pd.to_datetime(output["date"], utc=True)

        pair_key = canonical_pair(metadata.get("pair", ""))
        signal_frame = self._load_signals()
        if signal_frame.empty or not pair_key:
            return self._apply_default_signal_columns(output)

        pair_signals = signal_frame.loc[signal_frame["canonical_pair"] == pair_key].copy()
        if pair_signals.empty:
            return self._apply_default_signal_columns(output)

        pair_signals = pair_signals.sort_values("date")
        merge_columns = [
            "date",
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
        merged = pd.merge_asof(
            output.sort_values("date"),
            pair_signals[merge_columns],
            on="date",
            direction="backward",
            tolerance=self._stale_threshold(),
        )
        return self._apply_default_signal_columns(merged)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["entry_long_signal"] > 0, "enter_long"] = 1
        dataframe.loc[dataframe["entry_long_signal"] > 0, "enter_tag"] = "qlib_rotation_long"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["exit_long_signal"] > 0, "exit_long"] = 1
        dataframe.loc[dataframe["exit_long_signal"] > 0, "exit_tag"] = "qlib_rotation_exit"
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        row = self._signal_snapshot(pair, current_time)
        if row is None:
            return 1.0
        target = float(row.get("target_leverage", 1.0) or 1.0)
        return min(max(target, 1.0), max_leverage)

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        row = self._signal_snapshot(pair, current_time)
        if row is None:
            return proposed_stake
        stake_fraction = float(row.get("entry_stake_fraction", 0.25) or 0.25)
        desired_stake = proposed_stake * max(stake_fraction, 0.0)
        return self._clamp_stake(desired_stake, min_stake, max_stake) or proposed_stake

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        if trade.has_open_orders:
            return None

        row = self._signal_snapshot(trade.pair, current_time)
        if row is None:
            return None

        candle_time = pd.Timestamp(row["date"])
        if self._last_adjustment_matches(trade, candle_time):
            return None

        reduce_signal = int(row.get("reduce_position_signal", 0) or 0)
        add_signal = int(row.get("add_position_signal", 0) or 0)
        max_layers = max(int(row.get("max_entry_layers", 1) or 1), 1)

        if reduce_signal and current_profit >= max(float(row.get("partial_take_profit_pct", 0.02) or 0.02) * 0.5, 0.01):
            reduce_fraction = min(max(float(row.get("reduce_stake_fraction", 0.33) or 0.33), 0.10), 0.95)
            if trade.nr_of_successful_exits < trade.nr_of_successful_entries:
                reduce_stake = trade.stake_amount * reduce_fraction
                self._mark_adjustment(trade, candle_time)
                return -reduce_stake, "signal_scale_out"

        if not add_signal:
            return None
        if trade.nr_of_successful_entries >= max_layers:
            return None
        if current_profit <= -(float(row.get("stop_loss_pct", 0.02) or 0.02) * 0.75):
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        if not filled_entries:
            return None
        base_stake = float(
            getattr(filled_entries[0], "stake_amount_filled", None)
            or getattr(filled_entries[0], "stake_amount", None)
            or trade.stake_amount
        )
        layer_fraction = float(row.get("layer_stake_fraction", 0.15) or 0.15)
        layer_decay = max(0.55, 1.0 - 0.15 * max(trade.nr_of_successful_entries - 1, 0))
        desired_stake = base_stake * layer_fraction * layer_decay
        bounded_stake = self._clamp_stake(desired_stake, min_stake, max_stake)
        if bounded_stake is None:
            return None

        self._mark_adjustment(trade, candle_time)
        return bounded_stake, f"signal_layer_{trade.nr_of_successful_entries + 1}"
