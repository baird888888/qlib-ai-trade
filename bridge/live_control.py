from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests

from bridge.optimize_live_strategy import (
    _build_variant_frames,
    _global_validation_scores,
    _load_feature_frames,
)
from bridge.signal_store import as_freqtrade_pair, to_external_signal_frame, write_signal_frame
from quant_trading.backtest.engine import run_backtest, run_portfolio_backtest
from quant_trading.config import BacktestConfig, RiskConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "bridge" / "runtime"
REPORT_DIR = RUNTIME_ROOT / "reports"
FEATURE_DIR = RUNTIME_ROOT / "features"
LIVE_ROOT = RUNTIME_ROOT / "live_console"
SETTINGS_PATH = LIVE_ROOT / "settings.json"
STATE_PATH = LIVE_ROOT / "state.json"
GENERATED_CONFIG_PATH = LIVE_ROOT / "generated_freqtrade_config.json"
GENERATED_PROFILE_PATH = LIVE_ROOT / "prepared_strategy_profile.json"
GENERATED_REPORT_PATH = LIVE_ROOT / "prepared_strategy_report.json"
GENERATED_EQUITY_CSV_PATH = LIVE_ROOT / "prepared_strategy_equity.csv"
GENERATED_EQUITY_PNG_PATH = LIVE_ROOT / "prepared_strategy_equity_curve.png"
GENERATED_LOG_PATH = LIVE_ROOT / "freqtrade_live.log"
GENERATED_DB_PATH = LIVE_ROOT / "tradesv3.live.sqlite"


def _ensure_live_root() -> None:
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_settings() -> dict[str, Any]:
    return {
        "selection": {
            "strategy_summary": str((REPORT_DIR / "live_strategy_search_summary.json").resolve()),
        },
        "exchange": {
            "name": "okx",
            "key": "",
            "secret": "",
            "password": "",
            "sandbox": False,
        },
        "telegram": {
            "enabled": False,
            "token": "",
            "chat_id": "",
        },
        "freqtrade": {
            "dry_run": True,
            "stake_amount": "unlimited",
            "max_open_trades": 3,
            "api_port": 8080,
            "api_username": "qlib",
            "api_password": "",
            "bot_name": "qlib-ai-trade-live",
        },
    }


def load_settings() -> dict[str, Any]:
    _ensure_live_root()
    if not SETTINGS_PATH.exists():
        settings = _default_settings()
        save_settings(settings)
        return settings
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    merged = _default_settings()
    for key, value in settings.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    if not merged["freqtrade"]["api_password"]:
        merged["freqtrade"]["api_password"] = secrets.token_urlsafe(18)
    return merged


def save_settings(settings: dict[str, Any]) -> Path:
    _ensure_live_root()
    merged = _default_settings()
    for key, value in settings.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    if not merged["freqtrade"]["api_password"]:
        merged["freqtrade"]["api_password"] = secrets.token_urlsafe(18)
    SETTINGS_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return SETTINGS_PATH


def _default_state() -> dict[str, Any]:
    return {
        "running": False,
        "pid": None,
        "started_at": None,
        "strategy_summary": None,
        "generated_config_path": None,
        "generated_profile_path": None,
        "log_path": str(GENERATED_LOG_PATH.resolve()),
        "last_prepare": None,
        "last_prepare_error": None,
        "last_start_error": None,
    }


def load_state() -> dict[str, Any]:
    _ensure_live_root()
    if not STATE_PATH.exists():
        state = _default_state()
        save_state(state)
        return state
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    merged = _default_state()
    merged.update(state)
    return merged


def save_state(state: dict[str, Any]) -> Path:
    _ensure_live_root()
    merged = _default_state()
    merged.update(state)
    STATE_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return STATE_PATH


def list_strategy_profiles() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for path in sorted(REPORT_DIR.glob("*search_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        best_params = payload.get("best_params")
        best_holdout = payload.get("best_holdout", {})
        holdout_portfolio = best_holdout.get("portfolio", {})
        holdout_stability = best_holdout.get("stability", {})
        profiles.append(
            {
                "id": path.name,
                "name": path.stem,
                "path": str(path.resolve()),
                "holdout_return_pct": float(holdout_portfolio.get("return_pct", 0.0) or 0.0),
                "holdout_drawdown_pct": float(holdout_portfolio.get("max_drawdown_pct", 0.0) or 0.0),
                "holdout_weekly_positive_ratio": float(holdout_stability.get("weekly_positive_ratio", 0.0) or 0.0),
                "holdout_monthly_positive_ratio": float(holdout_stability.get("monthly_positive_ratio", 0.0) or 0.0),
                "enable_range_reversion": bool(best_params.get("enable_range_reversion", False)) if isinstance(best_params, dict) else False,
                "symbols": list(payload.get("symbols", [])),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    profiles.sort(key=lambda item: item["holdout_return_pct"], reverse=True)
    return profiles


def _load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _frame_bounds(frames: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts = []
    ends = []
    for frame in frames.values():
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        starts.append(timestamps.min())
        ends.append(timestamps.max())
    return min(starts), max(ends)


def _to_sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def _render_equity_png(equity_curve: pd.DataFrame, output_path: Path) -> None:
    if equity_curve.empty:
        return
    plt.figure(figsize=(14, 7))
    plt.plot(pd.to_datetime(equity_curve["timestamp"], utc=True), equity_curve["equity"], color="#0f766e", linewidth=1.5)
    plt.title("Prepared Live Strategy Equity")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def prepare_strategy_signals(summary_path: str | Path) -> dict[str, Any]:
    _ensure_live_root()
    summary = _load_summary(summary_path)
    training_summary = _load_summary(REPORT_DIR / "qlib_training_summary.json")
    symbols = list(training_summary["symbols"])
    frames = _load_feature_frames(symbols)
    start, end = _frame_bounds(frames)
    calibration_days = int(summary.get("search_config", {}).get("calibration_window_days", 365) or 365)
    calibration_start = end - pd.Timedelta(days=calibration_days)
    risk_config = RiskConfig()
    backtest_config = BacktestConfig()
    fallback_scores = _global_validation_scores(frames, calibration_start, end)
    variant_frames = _build_variant_frames(
        frames=frames,
        symbols=symbols,
        start=start,
        end=end,
        validation_start=calibration_start,
        validation_end=end,
        params=summary["best_params"],
        risk_config=risk_config,
        fallback_scores=fallback_scores,
    )

    external_frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        frame = variant_frames[symbol]
        frame.to_parquet(FEATURE_DIR / f"{symbol.lower().replace('-', '_')}_live_ready_signals.parquet", index=False)
        external_frames.append(to_external_signal_frame(frame, symbol))
        backtest = run_backtest(frame, backtest_config=backtest_config, risk_config=risk_config)
        rows.append({"symbol": symbol, **backtest["summary"]})

    write_signal_frame(pd.concat(external_frames, ignore_index=True))
    portfolio_backtest = run_portfolio_backtest(variant_frames, backtest_config=backtest_config, risk_config=risk_config)
    equity_curve = portfolio_backtest["equity_curve"].copy()
    trades = portfolio_backtest["trades"].copy()
    if not equity_curve.empty:
        equity_curve.to_csv(GENERATED_EQUITY_CSV_PATH, index=False)
        _render_equity_png(equity_curve, GENERATED_EQUITY_PNG_PATH)

    payload = {
        "generated_at": _utc_now_iso(),
        "source_summary": str(Path(summary_path).resolve()),
        "symbols": symbols,
        "best_params": summary["best_params"],
        "rows": rows,
        "portfolio": portfolio_backtest["summary"],
        "artifacts": {
            "signal_store": str((RUNTIME_ROOT / "freqtrade_signals.parquet").resolve()),
            "equity_csv": str(GENERATED_EQUITY_CSV_PATH.resolve()) if GENERATED_EQUITY_CSV_PATH.exists() else None,
            "equity_png": str(GENERATED_EQUITY_PNG_PATH.resolve()) if GENERATED_EQUITY_PNG_PATH.exists() else None,
        },
    }
    GENERATED_PROFILE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    GENERATED_REPORT_PATH.write_text(json.dumps(portfolio_backtest["summary"], indent=2), encoding="utf-8")
    if not trades.empty:
        trades.to_csv(LIVE_ROOT / "prepared_strategy_trades.csv", index=False)
    return payload


def _freqtrade_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [
        str((PROJECT_ROOT / "freqtrade-develop").resolve()),
        str(PROJECT_ROOT.resolve()),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _freqtrade_config(settings: dict[str, Any], prepared_profile: dict[str, Any]) -> dict[str, Any]:
    exchange_settings = settings["exchange"]
    telegram_settings = settings["telegram"]
    freqtrade_settings = settings["freqtrade"]
    symbols = prepared_profile["symbols"]
    port = int(freqtrade_settings.get("api_port", 8080))
    api_password = str(freqtrade_settings.get("api_password") or secrets.token_urlsafe(18))

    return {
        "$schema": "https://schema.freqtrade.io/schema.json",
        "max_open_trades": int(freqtrade_settings.get("max_open_trades", 3)),
        "stake_currency": "USDT",
        "stake_amount": freqtrade_settings.get("stake_amount", "unlimited"),
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "timeframe": "5m",
        "dry_run": bool(freqtrade_settings.get("dry_run", True)),
        "dry_run_wallet": 100000,
        "cancel_open_orders_on_exit": False,
        "trading_mode": "spot",
        "margin_mode": "",
        "db_url": _to_sqlite_url(GENERATED_DB_PATH),
        "exchange": {
            "name": exchange_settings.get("name", "okx"),
            "key": exchange_settings.get("key", ""),
            "secret": exchange_settings.get("secret", ""),
            "password": exchange_settings.get("password", ""),
            "enable_ws": True,
            "ccxt_config": {"enableRateLimit": True},
            "ccxt_async_config": {"enableRateLimit": True},
            "pair_whitelist": [as_freqtrade_pair(symbol) for symbol in symbols],
            "pair_blacklist": [],
        },
        "pairlists": [{"method": "StaticPairList"}],
        "telegram": {
            "enabled": bool(telegram_settings.get("enabled", False)),
            "token": telegram_settings.get("token", ""),
            "chat_id": telegram_settings.get("chat_id", ""),
            "allow_custom_messages": True,
        },
        "api_server": {
            "enabled": True,
            "listen_ip_address": "127.0.0.1",
            "listen_port": port,
            "verbosity": "error",
            "enable_openapi": False,
            "jwt_secret_key": secrets.token_hex(16),
            "ws_token": secrets.token_urlsafe(24),
            "CORS_origins": [],
            "username": str(freqtrade_settings.get("api_username", "qlib")),
            "password": api_password,
        },
        "bot_name": str(freqtrade_settings.get("bot_name", "qlib-ai-trade-live")),
        "initial_state": "running",
        "force_entry_enable": False,
        "strategy": "QlibExternalSignalStrategy",
        "internals": {"process_throttle_secs": 5},
    }


def _freqtrade_command() -> list[str]:
    return [
        str((PROJECT_ROOT / ".venv" / "Scripts" / "python.exe").resolve()),
        "-m",
        "freqtrade",
        "trade",
        "--config",
        str(GENERATED_CONFIG_PATH.resolve()),
        "--user-data-dir",
        str((PROJECT_ROOT / "freqtrade-develop" / "user_data").resolve()),
        "--strategy-path",
        str((PROJECT_ROOT / "freqtrade-develop" / "user_data" / "strategies").resolve()),
        "--strategy",
        "QlibExternalSignalStrategy",
    ]


def _pid_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    command = f"try {{ Get-Process -Id {int(pid)} | Out-Null; exit 0 }} catch {{ exit 1 }}"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def stop_bot() -> dict[str, Any]:
    state = load_state()
    pid = state.get("pid")
    if pid and _pid_is_running(int(pid)):
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {int(pid)} -Force"],
            capture_output=True,
            text=True,
        )
    state.update(
        {
            "running": False,
            "pid": None,
            "started_at": None,
        }
    )
    save_state(state)
    return state


def start_bot() -> dict[str, Any]:
    settings = load_settings()
    state = load_state()
    if state.get("pid") and _pid_is_running(int(state["pid"])):
        state["running"] = True
        save_state(state)
        return state

    selected_summary = str(Path(settings["selection"]["strategy_summary"]).resolve())
    prepared_profile = None
    if GENERATED_PROFILE_PATH.exists():
        try:
            cached_profile = json.loads(GENERATED_PROFILE_PATH.read_text(encoding="utf-8"))
            if str(cached_profile.get("source_summary")) == selected_summary:
                prepared_profile = cached_profile
        except Exception:
            prepared_profile = None
    if prepared_profile is None:
        prepared_profile = prepare_strategy_signals(selected_summary)
    config = _freqtrade_config(settings, prepared_profile)
    GENERATED_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    settings["freqtrade"]["api_password"] = config["api_server"]["password"]
    save_settings(settings)

    log_handle = GENERATED_LOG_PATH.open("a", encoding="utf-8")
    process = subprocess.Popen(
        _freqtrade_command(),
        cwd=str(PROJECT_ROOT),
        env=_freqtrade_env(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    state.update(
        {
            "running": True,
            "pid": process.pid,
            "started_at": _utc_now_iso(),
            "strategy_summary": selected_summary,
            "generated_config_path": str(GENERATED_CONFIG_PATH.resolve()),
            "generated_profile_path": str(GENERATED_PROFILE_PATH.resolve()),
            "last_prepare": _utc_now_iso(),
            "last_prepare_error": None,
            "last_start_error": None,
        }
    )
    save_state(state)
    return state


def tail_log(lines: int = 120) -> list[str]:
    if not GENERATED_LOG_PATH.exists():
        return []
    content = GENERATED_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    return content[-lines:]


def _api_base_url(settings: dict[str, Any]) -> str:
    port = int(settings["freqtrade"].get("api_port", 8080))
    return f"http://127.0.0.1:{port}/api/v1"


def _api_auth(settings: dict[str, Any]) -> tuple[str, str]:
    return (
        str(settings["freqtrade"].get("api_username", "qlib")),
        str(settings["freqtrade"].get("api_password", "")),
    )


def query_live_api() -> dict[str, Any]:
    settings = load_settings()
    state = load_state()
    if not state.get("pid") or not _pid_is_running(int(state["pid"])):
        return {}

    base_url = _api_base_url(settings)
    auth = _api_auth(settings)
    snapshot: dict[str, Any] = {}
    for name, endpoint in {
        "count": "/count",
        "profit": "/profit",
        "status": "/status",
    }.items():
        try:
            response = requests.get(f"{base_url}{endpoint}", auth=auth, timeout=5)
            response.raise_for_status()
            snapshot[name] = response.json()
        except Exception as exc:
            snapshot[name] = {"error": str(exc)}
    return snapshot


def live_snapshot() -> dict[str, Any]:
    settings = load_settings()
    state = load_state()
    running = bool(state.get("pid") and _pid_is_running(int(state["pid"])))
    if state.get("running") != running:
        state["running"] = running
        if not running:
            state["pid"] = None
            state["started_at"] = None
        save_state(state)

    prepared_profile = None
    if GENERATED_PROFILE_PATH.exists():
        prepared_profile = json.loads(GENERATED_PROFILE_PATH.read_text(encoding="utf-8"))

    return {
        "settings": settings,
        "state": state,
        "profiles": list_strategy_profiles(),
        "prepared_profile": prepared_profile,
        "live_api": query_live_api(),
        "log_tail": tail_log(),
        "artifacts": {
            "config": str(GENERATED_CONFIG_PATH.resolve()) if GENERATED_CONFIG_PATH.exists() else None,
            "profile": str(GENERATED_PROFILE_PATH.resolve()) if GENERATED_PROFILE_PATH.exists() else None,
            "equity_csv": str(GENERATED_EQUITY_CSV_PATH.resolve()) if GENERATED_EQUITY_CSV_PATH.exists() else None,
            "equity_png": str(GENERATED_EQUITY_PNG_PATH.resolve()) if GENERATED_EQUITY_PNG_PATH.exists() else None,
            "log": str(GENERATED_LOG_PATH.resolve()),
        },
    }
