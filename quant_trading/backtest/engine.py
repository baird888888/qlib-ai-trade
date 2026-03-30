from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from quant_trading.config import BacktestConfig, RiskConfig
from quant_trading.risk.kelly import estimate_trade_statistics, position_fraction_from_stop


def _closed_trades_to_frame(closed_trades: tuple) -> pd.DataFrame:
    return pd.DataFrame({"PnL": [trade.pl for trade in closed_trades]})


def _empty_backtest_summary() -> dict[str, float | int]:
    return {
        "return_pct": 0.0,
        "return_ann_pct": 0.0,
        "cagr_pct": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "payoff_ratio": 0.0,
        "trades": 0,
    }


@dataclass(slots=True)
class StrategyRuntimeConfig:
    risk: RiskConfig
    max_backtest_leverage: float


@dataclass(slots=True)
class PortfolioPosition:
    symbol: str
    quantity: float
    margin: float
    avg_entry_price: float
    leverage: float
    successful_entries: int
    successful_exits: int


class MultiCycleTrendStrategy(Strategy):
    runtime_config = StrategyRuntimeConfig(risk=RiskConfig(), max_backtest_leverage=1.0)

    def init(self) -> None:
        pass

    def _value(self, column: str, default: float) -> float:
        series = getattr(self.data, column, None)
        if series is None:
            return float(default)
        try:
            value = series[-1]
        except Exception:
            return float(default)
        if pd.isna(value):
            return float(default)
        return float(value)

    def _int_value(self, column: str, default: int) -> int:
        return int(round(self._value(column, float(default))))

    def _observed_kelly(self, seeded_kelly: float) -> float:
        observed_profile = estimate_trade_statistics(
            _closed_trades_to_frame(self.closed_trades),
            default_win_rate=self.runtime_config.risk.default_win_rate,
            default_payoff_ratio=self.runtime_config.risk.default_payoff_ratio,
            floor=self.runtime_config.risk.kelly_floor,
            cap=self.runtime_config.risk.kelly_cap,
        )
        return max(seeded_kelly * 0.5, observed_profile["kelly_fraction"])

    def _order_fraction(
        self,
        stop_loss_pct: float,
        seeded_kelly: float,
        stake_fraction: float,
        target_leverage: float,
    ) -> float:
        effective_kelly = self._observed_kelly(seeded_kelly)
        position_fraction = position_fraction_from_stop(
            stop_loss_pct=stop_loss_pct,
            kelly_value=effective_kelly,
            min_risk_fraction=self.runtime_config.risk.min_risk_fraction,
            max_risk_fraction=self.runtime_config.risk.max_risk_fraction,
            kelly_cap=self.runtime_config.risk.kelly_cap,
        )
        if position_fraction <= 0:
            return 0.0

        leverage_scale = max(target_leverage, 1.0) / max(self.runtime_config.max_backtest_leverage, 1.0)
        order_fraction = position_fraction * max(stake_fraction, 0.0) * leverage_scale
        return float(min(max(order_fraction, 0.0), 0.999))

    def _open_long(self, stake_fraction: float) -> None:
        stop_loss_pct = self._value("stop_loss_pct", 0.02)
        take_profit_pct = self._value("take_profit_pct", 0.04)
        seeded_kelly = self._value("kelly_fraction", 0.01)
        target_leverage = self._value("target_leverage", 1.0)
        order_fraction = self._order_fraction(
            stop_loss_pct=stop_loss_pct,
            seeded_kelly=seeded_kelly,
            stake_fraction=stake_fraction,
            target_leverage=target_leverage,
        )
        if order_fraction <= 0:
            return

        price = float(self.data.Close[-1])
        self.buy(
            size=order_fraction,
            sl=price * (1.0 - stop_loss_pct),
            tp=price * (1.0 + take_profit_pct),
        )

    def next(self) -> None:
        exit_signal = self._int_value("exit_long_signal", 0)
        add_signal = self._int_value("add_position_signal", 0)
        reduce_signal = self._int_value("reduce_position_signal", 0)

        if self.position:
            current_profit = self.position.pl_pct / 100.0
            if exit_signal:
                self.position.close()
                return

            partial_take_profit = self._value("partial_take_profit_pct", self._value("take_profit_pct", 0.04) * 0.5)
            reduce_fraction = self._value("reduce_stake_fraction", 0.33)
            if reduce_signal and current_profit >= max(partial_take_profit * 0.5, 0.01):
                self.position.close(portion=min(max(reduce_fraction, 0.10), 0.95))
                return

            active_layers = max(len(self.trades), 1)
            max_layers = max(self._int_value("max_entry_layers", 1), 1)
            stop_loss_pct = self._value("stop_loss_pct", 0.02)
            if add_signal and active_layers < max_layers and current_profit > -(stop_loss_pct * 0.75):
                self._open_long(stake_fraction=self._value("layer_stake_fraction", 0.15))
                return

            return

        if self._int_value("entry_long_signal", 0) != 1:
            return

        self._open_long(stake_fraction=self._value("entry_stake_fraction", 0.25))


def prepare_backtest_frame(signal_frame: pd.DataFrame) -> pd.DataFrame:
    frame = signal_frame.copy()
    frame["Date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None)
    frame = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "vol": "Volume",
        }
    )
    frame = frame.set_index("Date")
    return frame


def _payoff_ratio_from_trades(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    wins = trades.loc[trades["PnL"] > 0, "PnL"]
    losses = trades.loc[trades["PnL"] < 0, "PnL"].abs()
    if wins.empty or losses.empty:
        return 0.0
    return float(wins.mean() / losses.mean())


def summarize_backtest_stats(stats: pd.Series) -> tuple[dict, pd.DataFrame]:
    trades = stats["_trades"].copy() if "_trades" in stats else pd.DataFrame()
    summary = {
        "return_pct": float(stats.get("Return [%]", 0.0)),
        "return_ann_pct": float(stats.get("Return (Ann.) [%]", 0.0) or 0.0),
        "cagr_pct": float(stats.get("CAGR [%]", 0.0) or 0.0),
        "sharpe": float(stats.get("Sharpe Ratio", 0.0) or 0.0),
        "sortino": float(stats.get("Sortino Ratio", 0.0) or 0.0),
        "calmar": float(stats.get("Calmar Ratio", 0.0) or 0.0),
        "max_drawdown_pct": abs(float(stats.get("Max. Drawdown [%]", 0.0))),
        "win_rate_pct": float(stats.get("Win Rate [%]", 0.0) or 0.0),
        "payoff_ratio": _payoff_ratio_from_trades(trades),
        "trades": int(stats.get("# Trades", 0) or 0),
    }
    return summary, trades


def run_backtest(
    signal_frame: pd.DataFrame,
    backtest_config: BacktestConfig | None = None,
    risk_config: RiskConfig | None = None,
) -> dict:
    backtest_config = backtest_config or BacktestConfig()
    risk_config = risk_config or RiskConfig()
    if signal_frame.empty or len(signal_frame) < 2:
        empty_trades = pd.DataFrame()
        empty_summary = _empty_backtest_summary()
        return {"summary": empty_summary, "trades": empty_trades, "stats": pd.Series(dtype="float64")}

    leverage_series = (
        pd.to_numeric(signal_frame["target_leverage"], errors="coerce").fillna(1.0)
        if "target_leverage" in signal_frame.columns
        else pd.Series(1.0, index=signal_frame.index, dtype="float64")
    )
    max_signal_leverage = float(leverage_series.max() or 1.0)
    configured_leverage = 1.0 / max(float(backtest_config.margin), 1e-6)
    max_backtest_leverage = max(1.0, float(max(max_signal_leverage, configured_leverage)))
    margin = 1.0 / max_backtest_leverage

    MultiCycleTrendStrategy.runtime_config = StrategyRuntimeConfig(
        risk=risk_config,
        max_backtest_leverage=max_backtest_leverage,
    )
    prepared = prepare_backtest_frame(signal_frame)
    backtest = Backtest(
        prepared,
        MultiCycleTrendStrategy,
        cash=backtest_config.initial_cash,
        spread=backtest_config.spread,
        commission=backtest_config.commission,
        margin=margin,
        exclusive_orders=backtest_config.exclusive_orders,
        finalize_trades=True,
    )
    stats = backtest.run()
    summary, trades = summarize_backtest_stats(stats)
    return {"summary": summary, "trades": trades, "stats": stats}


def _prepare_portfolio_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    local = frame.copy()
    local["symbol"] = symbol
    local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
    local["open"] = pd.to_numeric(local.get("open", local.get("close")), errors="coerce")
    local["close"] = pd.to_numeric(local["close"], errors="coerce")
    local = local.dropna(subset=["timestamp", "open", "close"]).sort_values("timestamp").reset_index(drop=True)

    default_columns = {
        "entry_long_signal": 0,
        "exit_long_signal": 0,
        "add_position_signal": 0,
        "reduce_position_signal": 0,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "partial_take_profit_pct": 0.02,
        "kelly_fraction": 0.01,
        "target_leverage": 1.0,
        "entry_stake_fraction": 0.25,
        "layer_stake_fraction": 0.10,
        "reduce_stake_fraction": 0.33,
        "max_entry_layers": 1,
    }
    integer_columns = {
        "entry_long_signal",
        "exit_long_signal",
        "add_position_signal",
        "reduce_position_signal",
        "max_entry_layers",
    }
    for column, default in default_columns.items():
        series = pd.to_numeric(local[column], errors="coerce") if column in local.columns else pd.Series(default, index=local.index)
        if column in integer_columns:
            local[column] = series.fillna(default).astype(int)
        else:
            local[column] = series.fillna(default).astype(float)

    execution_columns = {
        "entry_long_signal": 0,
        "exit_long_signal": 0,
        "add_position_signal": 0,
        "reduce_position_signal": 0,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "partial_take_profit_pct": 0.02,
        "kelly_fraction": 0.01,
        "target_leverage": 1.0,
        "entry_stake_fraction": 0.25,
        "layer_stake_fraction": 0.10,
        "reduce_stake_fraction": 0.33,
        "max_entry_layers": 1,
    }
    for column, default in execution_columns.items():
        shifted = local[column].shift(1)
        exec_column = f"exec_{column}"
        if column in integer_columns:
            local[exec_column] = shifted.fillna(default).astype(int)
        else:
            local[exec_column] = shifted.fillna(default).astype(float)

    local["timestamp_ns"] = local["timestamp"].astype("int64")
    return local[
        [
            "timestamp",
            "timestamp_ns",
            "symbol",
            "open",
            "close",
            "entry_long_signal",
            "exit_long_signal",
            "add_position_signal",
            "reduce_position_signal",
            "stop_loss_pct",
            "take_profit_pct",
            "partial_take_profit_pct",
            "kelly_fraction",
            "target_leverage",
            "entry_stake_fraction",
            "layer_stake_fraction",
            "reduce_stake_fraction",
            "max_entry_layers",
            "exec_entry_long_signal",
            "exec_exit_long_signal",
            "exec_add_position_signal",
            "exec_reduce_position_signal",
            "exec_stop_loss_pct",
            "exec_take_profit_pct",
            "exec_partial_take_profit_pct",
            "exec_kelly_fraction",
            "exec_target_leverage",
            "exec_entry_stake_fraction",
            "exec_layer_stake_fraction",
            "exec_reduce_stake_fraction",
            "exec_max_entry_layers",
        ]
    ]


def _position_pnl(position: PortfolioPosition, price: float) -> float:
    raw_pnl = position.quantity * (float(price) - position.avg_entry_price)
    return float(max(raw_pnl, -position.margin))


def _position_equity(position: PortfolioPosition, price: float) -> float:
    return float(max(position.margin + _position_pnl(position, price), 0.0))


def _apply_execution_slippage(price: float, slippage: float, *, is_entry: bool) -> float:
    clean_price = max(float(price), 1e-9)
    clean_slippage = max(float(slippage), 0.0)
    multiplier = 1.0 + clean_slippage if is_entry else max(1.0 - clean_slippage, 1e-9)
    return float(clean_price * multiplier)


def _portfolio_equity(
    free_cash: float,
    positions: dict[str, PortfolioPosition],
    latest_prices: dict[str, float],
) -> tuple[float, float, float]:
    used_margin = 0.0
    gross_notional = 0.0
    equity = float(free_cash)
    for symbol, position in positions.items():
        price = latest_prices.get(symbol, position.avg_entry_price)
        equity += _position_equity(position, price)
        used_margin += position.margin
        gross_notional += position.quantity * price
    return float(equity), float(used_margin), float(gross_notional)


def _portfolio_order_margin(
    equity: float,
    stop_loss_pct: float,
    seeded_kelly: float,
    stake_fraction: float,
    risk_config: RiskConfig,
) -> float:
    position_fraction = position_fraction_from_stop(
        stop_loss_pct=max(float(stop_loss_pct), 1e-6),
        kelly_value=float(seeded_kelly),
        min_risk_fraction=risk_config.min_risk_fraction,
        max_risk_fraction=risk_config.max_risk_fraction,
        kelly_cap=risk_config.kelly_cap,
    )
    desired_margin = float(equity) * position_fraction * max(float(stake_fraction), 0.0)
    return float(max(desired_margin, 0.0))


def _open_or_add_portfolio_position(
    positions: dict[str, PortfolioPosition],
    symbol: str,
    price: float,
    margin: float,
    leverage: float,
    commission_rate: float,
) -> tuple[float, float]:
    leverage = max(float(leverage), 1.0)
    price = max(float(price), 1e-9)
    margin = max(float(margin), 0.0)
    notional = margin * leverage
    if notional <= 0:
        return 0.0, 0.0

    commission = notional * float(commission_rate)
    quantity = notional / price
    existing = positions.get(symbol)
    if existing is None:
        positions[symbol] = PortfolioPosition(
            symbol=symbol,
            quantity=quantity,
            margin=margin,
            avg_entry_price=price,
            leverage=leverage,
            successful_entries=1,
            successful_exits=0,
        )
    else:
        total_quantity = existing.quantity + quantity
        if total_quantity <= 0:
            return 0.0, 0.0
        existing.avg_entry_price = (
            existing.avg_entry_price * existing.quantity + price * quantity
        ) / total_quantity
        existing.quantity = total_quantity
        existing.margin += margin
        existing.leverage = (existing.quantity * price) / max(existing.margin, 1e-9)
        existing.successful_entries += 1
    return float(margin), float(commission)


def _close_portfolio_position_portion(
    position: PortfolioPosition,
    price: float,
    portion: float,
    commission_rate: float,
) -> tuple[float, dict]:
    portion = min(max(float(portion), 0.0), 1.0)
    if portion <= 0:
        return 0.0, {}

    price = max(float(price), 1e-9)
    quantity = position.quantity * portion
    margin_release = position.margin * portion
    raw_pnl = quantity * (price - position.avg_entry_price)
    realized_pnl = float(max(raw_pnl, -margin_release))
    commission = float(quantity * price * commission_rate)

    position.quantity -= quantity
    position.margin -= margin_release
    position.successful_exits += 1

    cash_delta = margin_release + realized_pnl - commission
    trade_row = {
        "symbol": position.symbol,
        "PnL": realized_pnl - commission,
        "gross_pnl": realized_pnl,
        "commission": commission,
        "entry_price": position.avg_entry_price,
        "exit_price": price,
        "portion": portion,
    }
    return float(cash_delta), trade_row


def _build_equity_summary(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
) -> dict[str, float | int]:
    if equity_curve.empty:
        return _empty_backtest_summary()

    equity = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna()
    if equity.empty:
        return _empty_backtest_summary()

    timestamps = pd.to_datetime(equity_curve["timestamp"], utc=True)
    total_return = float(equity.iloc[-1] / initial_cash - 1.0)
    elapsed_years = max((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / (365.25 * 24 * 3600), 0.0)
    if elapsed_years > 0 and equity.iloc[-1] > 0:
        annualized_log_return = np.log(max(equity.iloc[-1] / initial_cash, 1e-12)) / elapsed_years
        annualized_log_return = float(np.clip(annualized_log_return, -20.0, 20.0))
        cagr_pct = float(np.expm1(annualized_log_return) * 100.0)
    else:
        cagr_pct = 0.0

    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(timestamps) > 1:
        deltas = pd.Series(timestamps).diff().dropna().dt.total_seconds()
        median_seconds = float(deltas.median()) if not deltas.empty else 0.0
        periods_per_year = (365.25 * 24 * 3600) / median_seconds if median_seconds > 0 else 0.0
    else:
        periods_per_year = 0.0

    if not returns.empty and returns.std(ddof=0) > 0 and periods_per_year > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    downside = returns[returns < 0]
    if not downside.empty and downside.std(ddof=0) > 0 and periods_per_year > 0:
        sortino = float(returns.mean() / downside.std(ddof=0) * np.sqrt(periods_per_year))
    else:
        sortino = 0.0

    drawdown = 1.0 - equity / equity.cummax()
    max_drawdown_pct = float(drawdown.max() * 100.0) if not drawdown.empty else 0.0
    calmar = float(cagr_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0

    pnl = pd.to_numeric(trades.get("PnL"), errors="coerce").dropna() if not trades.empty else pd.Series(dtype="float64")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0].abs()
    win_rate_pct = float((pnl > 0).mean() * 100.0) if not pnl.empty else 0.0
    payoff_ratio = float(wins.mean() / losses.mean()) if not wins.empty and not losses.empty else 0.0

    return {
        "return_pct": total_return * 100.0,
        "return_ann_pct": cagr_pct,
        "cagr_pct": cagr_pct,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate_pct": win_rate_pct,
        "payoff_ratio": payoff_ratio,
        "trades": int(len(trades)),
    }


def run_portfolio_backtest(
    symbol_frames: dict[str, pd.DataFrame],
    backtest_config: BacktestConfig | None = None,
    risk_config: RiskConfig | None = None,
) -> dict:
    backtest_config = backtest_config or BacktestConfig()
    risk_config = risk_config or RiskConfig()

    prepared_frames = {
        symbol: _prepare_portfolio_frame(frame, symbol)
        for symbol, frame in symbol_frames.items()
        if frame is not None and not frame.empty
    }
    prepared_frames = {symbol: frame for symbol, frame in prepared_frames.items() if not frame.empty}
    if not prepared_frames:
        empty_frame = pd.DataFrame()
        return {"summary": _empty_backtest_summary(), "trades": empty_frame, "equity_curve": empty_frame}

    all_timestamp_arrays = [frame["timestamp_ns"].to_numpy(dtype="int64") for frame in prepared_frames.values()]
    if not all_timestamp_arrays:
        empty_frame = pd.DataFrame()
        return {"summary": _empty_backtest_summary(), "trades": empty_frame, "equity_curve": empty_frame}
    all_timestamps = np.unique(np.concatenate(all_timestamp_arrays))

    symbol_order = sorted(prepared_frames)
    iterators: dict[str, object] = {}
    next_rows: dict[str, object | None] = {}
    for symbol in symbol_order:
        iterator = prepared_frames[symbol].itertuples(index=False)
        iterators[symbol] = iterator
        next_rows[symbol] = next(iterator, None)

    free_cash = float(backtest_config.initial_cash)
    positions: dict[str, PortfolioPosition] = {}
    latest_prices: dict[str, float] = {}
    trade_rows: list[dict] = []
    equity_rows: list[dict] = []

    for timestamp_ns in all_timestamps:
        current_rows: dict[str, object] = {}
        for symbol in symbol_order:
            row = next_rows[symbol]
            if row is not None and row.timestamp_ns == timestamp_ns:
                current_rows[symbol] = row
                latest_prices[symbol] = float(row.close)
                next_rows[symbol] = next(iterators[symbol], None)

        for symbol in symbol_order:
            row = current_rows.get(symbol)
            position = positions.get(symbol)
            if row is None or position is None:
                continue

            current_profit = float(row.close / max(position.avg_entry_price, 1e-9) - 1.0)
            stop_loss_pct = max(float(row.exec_stop_loss_pct), 1e-6)
            take_profit_pct = max(float(row.exec_take_profit_pct), stop_loss_pct * 1.5)

            exit_reason: str | None = None
            if current_profit <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif current_profit >= take_profit_pct:
                exit_reason = "take_profit"
            elif int(row.exec_exit_long_signal) == 1:
                exit_reason = "signal_exit"

            if exit_reason is not None:
                execution_price = _apply_execution_slippage(
                    row.open,
                    backtest_config.spread,
                    is_entry=False,
                )
                cash_delta, trade_row = _close_portfolio_position_portion(
                    position,
                    price=execution_price,
                    portion=1.0,
                    commission_rate=backtest_config.commission,
                )
                free_cash += cash_delta
                trade_row["timestamp"] = pd.to_datetime(timestamp_ns, utc=True)
                trade_row["reason"] = exit_reason
                trade_rows.append(trade_row)
                positions.pop(symbol, None)
                continue

            if int(row.exec_reduce_position_signal) == 1 and current_profit >= max(float(row.exec_partial_take_profit_pct) * 0.5, 0.01):
                if position.successful_exits < position.successful_entries:
                    reduce_fraction = min(max(float(row.exec_reduce_stake_fraction), 0.10), 0.95)
                    execution_price = _apply_execution_slippage(
                        row.open,
                        backtest_config.spread,
                        is_entry=False,
                    )
                    cash_delta, trade_row = _close_portfolio_position_portion(
                        position,
                        price=execution_price,
                        portion=reduce_fraction,
                        commission_rate=backtest_config.commission,
                    )
                    free_cash += cash_delta
                    trade_row["timestamp"] = pd.to_datetime(timestamp_ns, utc=True)
                    trade_row["reason"] = "signal_scale_out"
                    trade_rows.append(trade_row)
                    if position.quantity <= 1e-12 or position.margin <= 1e-12:
                        positions.pop(symbol, None)

        for symbol in symbol_order:
            row = current_rows.get(symbol)
            if row is None:
                continue

            equity, _, _ = _portfolio_equity(free_cash, positions, latest_prices)
            target_leverage = max(float(row.exec_target_leverage), 1.0)
            commission_multiplier = 1.0 + target_leverage * float(backtest_config.commission)
            affordable_margin = free_cash / commission_multiplier if commission_multiplier > 0 else 0.0
            if affordable_margin <= 0:
                continue

            position = positions.get(symbol)
            if position is None:
                if int(row.exec_entry_long_signal) != 1:
                    continue
                desired_margin = _portfolio_order_margin(
                    equity=equity,
                    stop_loss_pct=float(row.exec_stop_loss_pct),
                    seeded_kelly=float(row.exec_kelly_fraction),
                    stake_fraction=float(row.exec_entry_stake_fraction),
                    risk_config=risk_config,
                )
                margin_to_use = min(desired_margin, affordable_margin)
                if margin_to_use <= 0:
                    continue
                execution_price = _apply_execution_slippage(
                    row.open,
                    backtest_config.spread,
                    is_entry=True,
                )
                reserved_margin, commission = _open_or_add_portfolio_position(
                    positions,
                    symbol=symbol,
                    price=execution_price,
                    margin=margin_to_use,
                    leverage=target_leverage,
                    commission_rate=backtest_config.commission,
                )
                free_cash -= reserved_margin + commission
                continue

            if int(row.exec_add_position_signal) != 1:
                continue
            if position.successful_entries >= max(int(row.exec_max_entry_layers), 1):
                continue

            current_profit = float(row.close / max(position.avg_entry_price, 1e-9) - 1.0)
            if current_profit <= -(float(row.exec_stop_loss_pct) * 0.75):
                continue

            desired_margin = _portfolio_order_margin(
                equity=equity,
                stop_loss_pct=float(row.exec_stop_loss_pct),
                seeded_kelly=float(row.exec_kelly_fraction),
                stake_fraction=float(row.exec_layer_stake_fraction),
                risk_config=risk_config,
            )
            margin_to_use = min(desired_margin, affordable_margin)
            if margin_to_use <= 0:
                continue
            execution_price = _apply_execution_slippage(
                row.open,
                backtest_config.spread,
                is_entry=True,
            )
            reserved_margin, commission = _open_or_add_portfolio_position(
                positions,
                symbol=symbol,
                price=execution_price,
                margin=margin_to_use,
                leverage=target_leverage,
                commission_rate=backtest_config.commission,
            )
            free_cash -= reserved_margin + commission

        equity, used_margin, gross_notional = _portfolio_equity(free_cash, positions, latest_prices)
        equity_rows.append(
            {
                "timestamp": pd.to_datetime(timestamp_ns, utc=True),
                "equity": equity,
                "free_cash": free_cash,
                "used_margin": used_margin,
                "gross_notional": gross_notional,
                "open_positions": len(positions),
            }
        )

    if positions and latest_prices:
        final_timestamp = pd.to_datetime(all_timestamps[-1], utc=True)
        for symbol in list(positions):
            position = positions[symbol]
            price = latest_prices.get(symbol, position.avg_entry_price)
            execution_price = _apply_execution_slippage(
                price,
                backtest_config.spread,
                is_entry=False,
            )
            cash_delta, trade_row = _close_portfolio_position_portion(
                position,
                price=execution_price,
                portion=1.0,
                commission_rate=backtest_config.commission,
            )
            free_cash += cash_delta
            trade_row["timestamp"] = final_timestamp
            trade_row["reason"] = "finalize"
            trade_rows.append(trade_row)
            positions.pop(symbol, None)
        equity_rows.append(
            {
                "timestamp": final_timestamp,
                "equity": free_cash,
                "free_cash": free_cash,
                "used_margin": 0.0,
                "gross_notional": 0.0,
                "open_positions": 0,
            }
        )

    trades = pd.DataFrame(trade_rows)
    equity_curve = pd.DataFrame(equity_rows)
    summary = _build_equity_summary(
        equity_curve=equity_curve,
        trades=trades,
        initial_cash=float(backtest_config.initial_cash),
    )
    return {"summary": summary, "trades": trades, "equity_curve": equity_curve}
