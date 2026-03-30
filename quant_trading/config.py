from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class DataConfig:
    """Parameters controlling OKX historical data acquisition."""

    symbols: tuple[str, ...] = ("BTC-USDT", "ETH-USDT", "DOGE-USDT", "PEPE-USDT", "SUI-USDT", "SOL-USDT")
    lower_bar: str = "5m"
    higher_bar: str = "1H"
    lower_limit: int | None = 2400
    higher_limit: int | None = 1200
    page_size: int = 300


@dataclass(slots=True)
class FeatureConfig:
    """Windows used by the technical, Chan, Wyckoff, and Brooks feature blocks."""

    ema_fast: int = 20
    ema_slow: int = 60
    atr_period: int = 14
    volume_window: int = 20
    breakout_window: int = 20
    chan_min_separation: int = 3
    wyckoff_range_window: int = 36
    wyckoff_recent_window: int = 12
    brooks_body_window: int = 20


@dataclass(slots=True)
class SignalConfig:
    """Thresholds that combine the theory-specific features into trade signals."""

    simons_breakout_threshold: float = 0.35
    recent_spring_window: int = 12
    confidence_threshold: float = 0.72
    reward_multiple: float = 2.2
    atr_stop_multiple: float = 1.6
    min_stop_loss_pct: float = 0.008
    max_stop_loss_pct: float = 0.05


@dataclass(slots=True)
class RiskConfig:
    """Kelly and account-level risk constraints."""

    kelly_floor: float = 0.0
    kelly_cap: float = 0.25
    min_risk_fraction: float = 0.01
    max_risk_fraction: float = 0.02
    default_win_rate: float = 0.48
    default_payoff_ratio: float = 1.8


@dataclass(slots=True)
class BacktestConfig:
    """Execution assumptions for the backtesting.py engine."""

    initial_cash: float = 100_000.0
    spread: float = 0.0002
    commission: float = 0.0005
    margin: float = 1.0
    exclusive_orders: bool = False


@dataclass(slots=True)
class ResearchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMETER_NOTES = {
    "data.lower_limit": "Default number of 5-minute candles loaded from the selected market data source. Use 0 to request all available OKX history.",
    "data.higher_limit": "Default number of 1-hour candles loaded from the selected market data source. Use 0 to request all available OKX history.",
    "feature.ema_fast": "Fast EMA used to describe short-term directional momentum.",
    "feature.ema_slow": "Slow EMA used to define the dominant trend filter.",
    "feature.atr_period": "ATR lookback used for volatility-aware stop placement.",
    "feature.chan_min_separation": "Minimum candle gap between Chan fractal turning points.",
    "feature.wyckoff_range_window": "Lookback window for support, resistance, and accumulation range detection.",
    "signal.simons_breakout_threshold": "Minimum breakout strength score before the Simons proxy turns bullish.",
    "signal.reward_multiple": "Take-profit multiple applied to the stop-loss distance.",
    "risk.max_risk_fraction": "Maximum account equity risked on a single trade after Kelly scaling.",
    "backtest.margin": "Required margin ratio passed to backtesting.py. A value of 0.33 approximates 3x account leverage.",
    "backtest.spread": "Execution slippage / spread assumption. The portfolio backtest applies it as an adverse price adjustment on every fill.",
    "backtest.commission": "Trading fee assumption charged on fill notional. Default is 0.05% per side.",
}
