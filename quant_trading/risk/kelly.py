from __future__ import annotations

import math

import numpy as np
import pandas as pd


def kelly_fraction(win_rate: float, payoff_ratio: float, floor: float = 0.0, cap: float = 0.25) -> float:
    if payoff_ratio <= 0:
        return floor
    q = 1.0 - win_rate
    raw_fraction = ((payoff_ratio * win_rate) - q) / payoff_ratio
    return float(np.clip(raw_fraction, floor, cap))


def estimate_trade_statistics(
    trades: pd.DataFrame,
    default_win_rate: float = 0.48,
    default_payoff_ratio: float = 1.8,
    floor: float = 0.0,
    cap: float = 0.25,
) -> dict:
    if trades.empty or "PnL" not in trades.columns:
        estimated_kelly = kelly_fraction(default_win_rate, default_payoff_ratio, floor=floor, cap=cap)
        return {
            "win_rate": default_win_rate,
            "payoff_ratio": default_payoff_ratio,
            "kelly_fraction": estimated_kelly,
        }

    pnl = trades["PnL"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0].abs()
    win_rate = float((pnl > 0).mean()) if len(pnl) else default_win_rate
    payoff_ratio = float(wins.mean() / losses.mean()) if not losses.empty and not wins.empty else default_payoff_ratio
    estimated_kelly = kelly_fraction(win_rate, payoff_ratio, floor=floor, cap=cap)
    return {"win_rate": win_rate, "payoff_ratio": payoff_ratio, "kelly_fraction": estimated_kelly}


def risk_budget_from_kelly(
    kelly_value: float,
    min_risk_fraction: float = 0.01,
    max_risk_fraction: float = 0.02,
    kelly_cap: float = 0.25,
) -> float:
    clipped = float(np.clip(kelly_value, 0.0, kelly_cap))
    if math.isclose(kelly_cap, 0.0):
        return min_risk_fraction
    scaled = clipped / kelly_cap
    return float(min_risk_fraction + scaled * (max_risk_fraction - min_risk_fraction))


def position_fraction_from_stop(
    stop_loss_pct: float,
    kelly_value: float,
    min_risk_fraction: float = 0.01,
    max_risk_fraction: float = 0.02,
    kelly_cap: float = 0.25,
    max_position_fraction: float = 1.0,
) -> float:
    stop_loss_pct = max(float(stop_loss_pct), 1e-6)
    risk_budget = risk_budget_from_kelly(
        kelly_value=kelly_value,
        min_risk_fraction=min_risk_fraction,
        max_risk_fraction=max_risk_fraction,
        kelly_cap=kelly_cap,
    )
    fraction = risk_budget / stop_loss_pct
    return float(np.clip(fraction, 0.0, max_position_fraction))
