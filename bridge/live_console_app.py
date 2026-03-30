from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

from bridge.live_control import (
    GENERATED_EQUITY_PNG_PATH,
    GENERATED_LOG_PATH,
    live_snapshot,
    load_settings,
    prepare_strategy_signals,
    save_settings,
    start_bot,
    stop_bot,
)

APP = FastAPI(title="Qlib AI Trade Live Console")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qlib Live Console</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255,255,255,0.86);
      --ink: #14202b;
      --muted: #5a6772;
      --accent: #0f766e;
      --accent-2: #b45309;
      --danger: #b91c1c;
      --border: rgba(20,32,43,0.14);
      --shadow: 0 22px 60px rgba(20,32,43,0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.15), transparent 30%),
        radial-gradient(circle at top right, rgba(180,83,9,0.18), transparent 26%),
        linear-gradient(180deg, #f7f1e8 0%, #efe5d8 100%);
    }
    .shell {
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px;
    }
    .hero {
      margin-bottom: 22px;
      padding: 24px 28px;
      border: 1px solid var(--border);
      border-radius: 24px;
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(244,239,230,0.92));
      box-shadow: var(--shadow);
    }
    .hero h1 {
      margin: 0 0 10px 0;
      font-size: 2rem;
      letter-spacing: 0.02em;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 880px;
      line-height: 1.5;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 18px;
    }
    .panel {
      grid-column: span 12;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: var(--panel);
      box-shadow: var(--shadow);
      padding: 22px;
      backdrop-filter: blur(14px);
    }
    .panel h2 {
      margin: 0 0 14px 0;
      font-size: 1.1rem;
    }
    .panel.small { grid-column: span 4; }
    .panel.medium { grid-column: span 6; }
    .panel.large { grid-column: span 8; }
    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .field {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 12px;
    }
    label {
      color: var(--muted);
      font-size: 0.92rem;
      font-weight: 600;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid rgba(20,32,43,0.18);
      border-radius: 12px;
      padding: 11px 12px;
      background: rgba(255,255,255,0.94);
      color: var(--ink);
      font: inherit;
    }
    textarea {
      min-height: 180px;
      resize: vertical;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 8px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 11px 18px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.15s ease, opacity 0.15s ease;
    }
    button:hover { transform: translateY(-1px); }
    button.primary { background: var(--accent); color: white; }
    button.secondary { background: #112e4b; color: white; }
    button.warn { background: var(--accent-2); color: white; }
    button.danger { background: var(--danger); color: white; }
    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .stat {
      border: 1px solid rgba(20,32,43,0.12);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.9);
    }
    .stat .k {
      color: var(--muted);
      font-size: 0.84rem;
      margin-bottom: 4px;
    }
    .stat .v {
      font-size: 1.2rem;
      font-weight: 700;
    }
    .hint {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }
    .codebox {
      border-radius: 16px;
      background: #12202d;
      color: #dbe7f3;
      padding: 14px;
      font-family: "Consolas", "Courier New", monospace;
      white-space: pre-wrap;
      max-height: 360px;
      overflow: auto;
    }
    .image-wrap {
      border: 1px solid rgba(20,32,43,0.12);
      border-radius: 18px;
      overflow: hidden;
      background: white;
      min-height: 300px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .image-wrap img {
      max-width: 100%;
      display: block;
    }
    .status-dot {
      display: inline-flex;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 8px;
      background: #94a3b8;
    }
    .status-dot.on { background: #16a34a; }
    .status-dot.off { background: #dc2626; }
    @media (max-width: 1024px) {
      .panel.small, .panel.medium, .panel.large { grid-column: span 12; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Qlib Live Console</h1>
      <p>Configure your OKX and Telegram secrets, choose the strategy profile to run, prepare live signals, then start or stop the Freqtrade process without leaving this page.</p>
    </section>

    <div class="grid">
      <section class="panel medium">
        <h2>Strategy Selection</h2>
        <div class="field">
          <label for="strategy_summary">Strategy Profile</label>
          <select id="strategy_summary"></select>
        </div>
        <div class="cards" id="profile_cards"></div>
        <p class="hint">Profiles are discovered from <code>bridge/runtime/reports/*search_summary.json</code>. Start will automatically prepare the chosen profile before launching the bot.</p>
      </section>

      <section class="panel medium">
        <h2>Runtime Controls</h2>
        <div class="cards">
          <div class="stat"><div class="k">Bot Status</div><div class="v" id="bot_status">Unknown</div></div>
          <div class="stat"><div class="k">PID</div><div class="v" id="bot_pid">-</div></div>
          <div class="stat"><div class="k">Selected Summary</div><div class="v" id="bot_summary">-</div></div>
          <div class="stat"><div class="k">Prepared At</div><div class="v" id="bot_prepare">-</div></div>
        </div>
        <div class="actions">
          <button class="secondary" onclick="refreshState()">Refresh</button>
          <button class="warn" onclick="prepareProfile()">Prepare Signals</button>
          <button class="primary" onclick="startBot()">Start Strategy</button>
          <button class="danger" onclick="stopBot()">Stop Strategy</button>
        </div>
      </section>

      <section class="panel large">
        <h2>Exchange / Telegram / Bot Settings</h2>
        <div class="row">
          <div>
            <div class="field"><label for="okx_key">OKX API Key</label><input id="okx_key" type="password" /></div>
            <div class="field"><label for="okx_secret">OKX API Secret</label><input id="okx_secret" type="password" /></div>
            <div class="field"><label for="okx_password">OKX Passphrase</label><input id="okx_password" type="password" /></div>
            <div class="field"><label for="telegram_token">Telegram Bot Token</label><input id="telegram_token" type="password" /></div>
            <div class="field"><label for="telegram_chat_id">Telegram Chat ID</label><input id="telegram_chat_id" type="text" /></div>
          </div>
          <div>
            <div class="field"><label for="dry_run">Dry Run</label><select id="dry_run"><option value="true">true</option><option value="false">false</option></select></div>
            <div class="field"><label for="stake_amount">Stake Amount</label><input id="stake_amount" type="text" placeholder="unlimited" /></div>
            <div class="field"><label for="max_open_trades">Max Open Trades</label><input id="max_open_trades" type="number" min="1" step="1" /></div>
            <div class="field"><label for="api_port">API Port</label><input id="api_port" type="number" min="1024" step="1" /></div>
            <div class="field"><label for="bot_name">Bot Name</label><input id="bot_name" type="text" /></div>
            <div class="actions">
              <button class="primary" onclick="saveConfig()">Save Settings</button>
            </div>
            <p class="hint">Telegram open/close notifications are handed off to Freqtrade's native Telegram integration. If token and chat id are set, live trade notifications will be sent automatically.</p>
          </div>
        </div>
      </section>

      <section class="panel small">
        <h2>Prepared Profile Snapshot</h2>
        <div class="cards">
          <div class="stat"><div class="k">Portfolio Return</div><div class="v" id="prepared_return">-</div></div>
          <div class="stat"><div class="k">Annualized</div><div class="v" id="prepared_ann">-</div></div>
          <div class="stat"><div class="k">Drawdown</div><div class="v" id="prepared_dd">-</div></div>
          <div class="stat"><div class="k">Trades</div><div class="v" id="prepared_trades">-</div></div>
        </div>
      </section>

      <section class="panel small">
        <h2>Freqtrade API Snapshot</h2>
        <div class="cards">
          <div class="stat"><div class="k">Open Trades</div><div class="v" id="api_open">-</div></div>
          <div class="stat"><div class="k">Closed Trades</div><div class="v" id="api_closed">-</div></div>
          <div class="stat"><div class="k">Profit</div><div class="v" id="api_profit">-</div></div>
          <div class="stat"><div class="k">Status Rows</div><div class="v" id="api_status_rows">-</div></div>
        </div>
      </section>

      <section class="panel small">
        <h2>Prepared Equity Curve</h2>
        <div class="image-wrap">
          <img id="equity_png" alt="Prepared equity curve" />
        </div>
      </section>

      <section class="panel large">
        <h2>Log Tail</h2>
        <div class="codebox" id="log_tail">Loading...</div>
      </section>

      <section class="panel medium">
        <h2>Prepared Profile JSON</h2>
        <div class="codebox" id="prepared_json">Loading...</div>
      </section>

      <section class="panel medium">
        <h2>Live API JSON</h2>
        <div class="codebox" id="api_json">Loading...</div>
      </section>
    </div>
  </div>

  <script>
    function fmtPct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
      return `${Number(value).toFixed(2)}%`;
    }
    function fmtInt(value) {
      if (value === null || value === undefined || value === '') return '-';
      return String(value);
    }
    function renderProfiles(profiles, selectedPath) {
      const select = document.getElementById('strategy_summary');
      select.innerHTML = '';
      profiles.forEach((profile) => {
        const option = document.createElement('option');
        option.value = profile.path;
        option.textContent = `${profile.name} | holdout ${fmtPct(profile.holdout_return_pct)} | dd ${fmtPct(profile.holdout_drawdown_pct)}`;
        if (profile.path === selectedPath) option.selected = true;
        select.appendChild(option);
      });

      const selected = profiles.find((item) => item.path === select.value) || profiles[0];
      const cards = document.getElementById('profile_cards');
      cards.innerHTML = '';
      if (!selected) return;
      [
        ['Holdout Return', fmtPct(selected.holdout_return_pct)],
        ['Holdout DD', fmtPct(selected.holdout_drawdown_pct)],
        ['Weekly Positive', fmtPct(Number(selected.holdout_weekly_positive_ratio || 0) * 100)],
        ['Range Mode', selected.enable_range_reversion ? 'on' : 'off'],
      ].forEach(([k, v]) => {
        const div = document.createElement('div');
        div.className = 'stat';
        div.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
        cards.appendChild(div);
      });
    }

    async function refreshState() {
      const resp = await fetch('/api/state');
      const data = await resp.json();
      const settings = data.settings;
      renderProfiles(data.profiles, settings.selection.strategy_summary);
      document.getElementById('okx_key').value = settings.exchange.key || '';
      document.getElementById('okx_secret').value = settings.exchange.secret || '';
      document.getElementById('okx_password').value = settings.exchange.password || '';
      document.getElementById('telegram_token').value = settings.telegram.token || '';
      document.getElementById('telegram_chat_id').value = settings.telegram.chat_id || '';
      document.getElementById('dry_run').value = String(settings.freqtrade.dry_run);
      document.getElementById('stake_amount').value = settings.freqtrade.stake_amount || 'unlimited';
      document.getElementById('max_open_trades').value = settings.freqtrade.max_open_trades || 3;
      document.getElementById('api_port').value = settings.freqtrade.api_port || 8080;
      document.getElementById('bot_name').value = settings.freqtrade.bot_name || '';

      document.getElementById('bot_status').innerHTML = `<span class="status-dot ${data.state.running ? 'on' : 'off'}"></span>${data.state.running ? 'Running' : 'Stopped'}`;
      document.getElementById('bot_pid').textContent = fmtInt(data.state.pid);
      document.getElementById('bot_summary').textContent = data.state.strategy_summary ? data.state.strategy_summary.split(/[\\\\/]/).pop() : '-';
      document.getElementById('bot_prepare').textContent = data.state.last_prepare || '-';
      document.getElementById('log_tail').textContent = (data.log_tail || []).join('\\n') || 'No log output yet.';
      document.getElementById('prepared_json').textContent = JSON.stringify(data.prepared_profile || {}, null, 2);
      document.getElementById('api_json').textContent = JSON.stringify(data.live_api || {}, null, 2);

      const prepared = (data.prepared_profile || {}).portfolio || {};
      document.getElementById('prepared_return').textContent = fmtPct(prepared.return_pct);
      document.getElementById('prepared_ann').textContent = fmtPct(prepared.return_ann_pct);
      document.getElementById('prepared_dd').textContent = fmtPct(prepared.max_drawdown_pct);
      document.getElementById('prepared_trades').textContent = fmtInt(prepared.trades);

      const liveCount = (data.live_api || {}).count || {};
      const liveProfit = (data.live_api || {}).profit || {};
      const liveStatus = (data.live_api || {}).status || [];
      document.getElementById('api_open').textContent = fmtInt(liveCount.current);
      document.getElementById('api_closed').textContent = fmtInt(liveCount.total);
      document.getElementById('api_profit').textContent = liveProfit.profit_closed_coin !== undefined ? String(liveProfit.profit_closed_coin) : '-';
      document.getElementById('api_status_rows').textContent = Array.isArray(liveStatus) ? String(liveStatus.length) : '-';

      const img = document.getElementById('equity_png');
      img.src = '/artifacts/prepared-equity.png?t=' + Date.now();
    }

    async function saveConfig() {
      const payload = {
        selection: {
          strategy_summary: document.getElementById('strategy_summary').value,
        },
        exchange: {
          name: 'okx',
          key: document.getElementById('okx_key').value,
          secret: document.getElementById('okx_secret').value,
          password: document.getElementById('okx_password').value,
        },
        telegram: {
          enabled: Boolean(document.getElementById('telegram_token').value && document.getElementById('telegram_chat_id').value),
          token: document.getElementById('telegram_token').value,
          chat_id: document.getElementById('telegram_chat_id').value,
        },
        freqtrade: {
          dry_run: document.getElementById('dry_run').value === 'true',
          stake_amount: document.getElementById('stake_amount').value || 'unlimited',
          max_open_trades: Number(document.getElementById('max_open_trades').value || 3),
          api_port: Number(document.getElementById('api_port').value || 8080),
          bot_name: document.getElementById('bot_name').value || 'qlib-ai-trade-live',
        },
      };
      await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      await refreshState();
    }

    async function prepareProfile() {
      await saveConfig();
      await fetch('/api/prepare', { method: 'POST' });
      await refreshState();
    }

    async function startBot() {
      await saveConfig();
      await fetch('/api/start', { method: 'POST' });
      await refreshState();
    }

    async function stopBot() {
      await fetch('/api/stop', { method: 'POST' });
      await refreshState();
    }

    refreshState();
    setInterval(refreshState, 15000);
  </script>
</body>
</html>
"""


@APP.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@APP.get("/api/state")
def api_state() -> JSONResponse:
    return JSONResponse(live_snapshot())


@APP.post("/api/settings")
def api_settings(payload: dict[str, Any] = Body(...)) -> JSONResponse:
    settings = load_settings()
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key].update(value)
        else:
            settings[key] = value
    save_settings(settings)
    return JSONResponse({"ok": True, "settings": settings})


@APP.post("/api/prepare")
def api_prepare() -> JSONResponse:
    settings = load_settings()
    summary_path = settings["selection"]["strategy_summary"]
    if not Path(summary_path).exists():
        raise HTTPException(status_code=404, detail=f"Strategy summary not found: {summary_path}")
    payload = prepare_strategy_signals(summary_path)
    return JSONResponse({"ok": True, "prepared_profile": payload})


@APP.post("/api/start")
def api_start() -> JSONResponse:
    state = start_bot()
    return JSONResponse({"ok": True, "state": state})


@APP.post("/api/stop")
def api_stop() -> JSONResponse:
    state = stop_bot()
    return JSONResponse({"ok": True, "state": state})


@APP.get("/artifacts/prepared-equity.png")
def prepared_equity_png() -> FileResponse:
    if not GENERATED_EQUITY_PNG_PATH.exists():
        raise HTTPException(status_code=404, detail="Prepared equity curve not found.")
    return FileResponse(GENERATED_EQUITY_PNG_PATH)


def main() -> None:
    uvicorn.run(APP, host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
