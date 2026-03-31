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

APP = FastAPI(title="Qlib 实盘控制台")


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qlib 实盘控制台</title>
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
      min-width: 0;
      min-height: 90px;
    }
    .stat .k {
      color: var(--muted);
      font-size: 0.84rem;
      margin-bottom: 4px;
    }
    .stat .v {
      font-size: 1rem;
      font-weight: 700;
      line-height: 1.35;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
    }
    .hint {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }
    .kv-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .kv-item {
      border: 1px solid rgba(20,32,43,0.12);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.92);
      min-height: 72px;
    }
    .kv-item .k {
      color: var(--muted);
      font-size: 0.82rem;
      margin-bottom: 6px;
      font-weight: 600;
    }
    .kv-item .v {
      font-size: 0.98rem;
      font-weight: 700;
      line-height: 1.4;
      word-break: break-all;
      white-space: pre-wrap;
    }
    .mono {
      font-family: "Consolas", "Courier New", monospace;
      font-size: 0.9rem;
    }
    .table-wrap {
      border: 1px solid rgba(20,32,43,0.12);
      border-radius: 18px;
      overflow: auto;
      background: rgba(255,255,255,0.92);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 640px;
    }
    th, td {
      padding: 11px 12px;
      border-bottom: 1px solid rgba(20,32,43,0.08);
      text-align: left;
      vertical-align: top;
    }
    th {
      background: rgba(20,32,43,0.06);
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    td {
      font-size: 0.95rem;
      line-height: 1.45;
      word-break: break-word;
    }
    tbody tr:last-child td {
      border-bottom: 0;
    }
    .chart-shell {
      border: 1px solid rgba(20,32,43,0.12);
      border-radius: 18px;
      background: rgba(255,255,255,0.95);
      padding: 14px;
    }
    .chart-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      color: var(--muted);
      font-size: 0.88rem;
      margin-bottom: 10px;
    }
    .chart-box {
      width: 100%;
      height: 300px;
      border-radius: 14px;
      background:
        linear-gradient(180deg, rgba(15,118,110,0.08), rgba(15,118,110,0.01)),
        linear-gradient(90deg, rgba(20,32,43,0.04) 1px, transparent 1px),
        linear-gradient(180deg, rgba(20,32,43,0.04) 1px, transparent 1px);
      background-size: auto, 60px 100%, 100% 60px;
      display: flex;
      align-items: stretch;
      justify-content: stretch;
      overflow: hidden;
    }
    .chart-box svg {
      width: 100%;
      height: 100%;
      display: block;
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
      .kv-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Qlib 实盘控制台</h1>
      <p>在这里配置 OKX 和 Telegram 密钥，选择要运行的策略，准备实时信号，并直接启动或停止 Freqtrade 实盘进程。</p>
    </section>

    <div class="grid">
      <section class="panel medium">
        <h2>策略选择</h2>
        <div class="field">
          <label for="strategy_summary">策略配置</label>
          <select id="strategy_summary"></select>
        </div>
        <div class="cards" id="profile_cards"></div>
        <p class="hint">系统会自动扫描 <code>bridge/runtime/reports/*search_summary.json</code> 中的策略。点击启动前，会先自动准备当前选中的策略信号。</p>
      </section>

      <section class="panel medium">
        <h2>当前策略详情</h2>
        <div class="kv-grid" id="profile_meta"></div>
      </section>

      <section class="panel medium">
        <h2>运行控制</h2>
        <div class="cards">
          <div class="stat"><div class="k">机器人状态</div><div class="v" id="bot_status">未知</div></div>
          <div class="stat"><div class="k">PID</div><div class="v" id="bot_pid">-</div></div>
          <div class="stat"><div class="k">当前策略</div><div class="v" id="bot_summary">-</div></div>
          <div class="stat"><div class="k">最近准备时间</div><div class="v" id="bot_prepare">-</div></div>
        </div>
        <div class="actions">
          <button class="secondary" onclick="refreshState()">刷新状态</button>
          <button class="warn" onclick="prepareProfile()">准备信号</button>
          <button class="primary" onclick="startBot()">启动策略</button>
          <button class="danger" onclick="stopBot()">停止策略</button>
        </div>
      </section>

      <section class="panel large">
        <h2>交易所 / Telegram / 机器人设置</h2>
        <div class="row">
          <div>
            <div class="field"><label for="okx_key">OKX API 密钥</label><input id="okx_key" type="password" /></div>
            <div class="field"><label for="okx_secret">OKX API 私钥</label><input id="okx_secret" type="password" /></div>
            <div class="field"><label for="okx_password">OKX API 口令</label><input id="okx_password" type="password" /></div>
            <div class="field"><label for="telegram_token">Telegram 机器人令牌</label><input id="telegram_token" type="password" /></div>
            <div class="field"><label for="telegram_chat_id">Telegram 聊天 ID</label><input id="telegram_chat_id" type="text" /></div>
          </div>
          <div>
            <div class="field"><label for="dry_run">模拟盘</label><select id="dry_run"><option value="true">开启</option><option value="false">关闭</option></select></div>
            <div class="field"><label for="stake_amount">单笔资金</label><input id="stake_amount" type="text" placeholder="unlimited / 不限" /></div>
            <div class="field"><label for="max_open_trades">最大持仓数</label><input id="max_open_trades" type="number" min="1" step="1" /></div>
            <div class="field"><label for="api_port">API 端口</label><input id="api_port" type="number" min="1024" step="1" /></div>
            <div class="field"><label for="bot_name">机器人名称</label><input id="bot_name" type="text" /></div>
            <div class="actions">
              <button class="primary" onclick="saveConfig()">保存设置</button>
            </div>
            <p class="hint">开平仓通知走 Freqtrade 原生 Telegram 集成。只要填写 Telegram Token 和 Chat ID，实盘成交提醒就会自动发送。</p>
          </div>
        </div>
      </section>

      <section class="panel medium">
        <h2>当前设置摘要</h2>
        <div class="kv-grid" id="settings_overview"></div>
      </section>

      <section class="panel small">
        <h2>实盘预览概览</h2>
        <div class="cards">
          <div class="stat"><div class="k">全历史预览收益</div><div class="v" id="prepared_return">-</div></div>
          <div class="stat"><div class="k">全历史预览年化</div><div class="v" id="prepared_ann">-</div></div>
          <div class="stat"><div class="k">全历史预览回撤</div><div class="v" id="prepared_dd">-</div></div>
          <div class="stat"><div class="k">全历史预览交易数</div><div class="v" id="prepared_trades">-</div></div>
        </div>
        <p class="hint">这里显示的是为了实盘准备信号后，对全历史做的一次预览回放，不等同于留出集收益，也不等同于搜索阶段的全测试 walk-forward 指标。</p>
      </section>

      <section class="panel small">
        <h2>Freqtrade API 快照</h2>
        <div class="cards">
          <div class="stat"><div class="k">当前持仓</div><div class="v" id="api_open">-</div></div>
          <div class="stat"><div class="k">累计交易</div><div class="v" id="api_closed">-</div></div>
          <div class="stat"><div class="k">累计收益</div><div class="v" id="api_profit">-</div></div>
          <div class="stat"><div class="k">状态条目</div><div class="v" id="api_status_rows">-</div></div>
        </div>
      </section>

      <section class="panel medium">
        <h2>实时收益统计</h2>
        <div class="kv-grid" id="live_stats_overview"></div>
      </section>

      <section class="panel small">
        <h2>已准备收益曲线</h2>
        <div class="image-wrap">
          <img id="equity_png" alt="已准备收益曲线" />
        </div>
      </section>

      <section class="panel large">
        <h2>实时收益曲线</h2>
        <div class="chart-shell">
          <div class="chart-meta" id="live_chart_meta"></div>
          <div class="chart-box">
            <svg id="live_profit_chart" viewBox="0 0 1000 300" preserveAspectRatio="none"></svg>
          </div>
        </div>
      </section>

      <section class="panel medium">
        <h2>运行状态详情</h2>
        <div class="kv-grid" id="api_detail"></div>
      </section>

      <section class="panel large">
        <h2>策略参数明细</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>字段</th>
                <th>值</th>
              </tr>
            </thead>
            <tbody id="params_table"></tbody>
          </table>
        </div>
      </section>

      <section class="panel large">
        <h2>币种表现明细</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>币种</th>
                <th>收益</th>
                <th>年化</th>
                <th>回撤</th>
                <th>胜率</th>
                <th>盈亏比</th>
                <th>交易次数</th>
              </tr>
            </thead>
            <tbody id="rows_table"></tbody>
          </table>
        </div>
      </section>

      <section class="panel large">
        <h2>实时收益明细表</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>时间</th>
                <th>总收益率</th>
                <th>已平仓收益率</th>
                <th>当前持仓</th>
                <th>累计已平仓</th>
                <th>胜率</th>
                <th>最大回撤</th>
              </tr>
            </thead>
            <tbody id="live_metrics_table"></tbody>
          </table>
        </div>
      </section>

      <section class="panel large">
        <h2>运行日志</h2>
        <div class="codebox" id="log_tail">加载中...</div>
      </section>

      <section class="panel medium">
        <h2>策略配置 JSON</h2>
        <div class="codebox" id="prepared_json">加载中...</div>
      </section>

      <section class="panel medium">
        <h2>实盘 API JSON</h2>
        <div class="codebox" id="api_json">加载中...</div>
      </section>
    </div>
  </div>

  <script>
    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('\"', '&quot;')
        .replaceAll("'", '&#39;');
    }
    function fmtPct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
      return `${Number(value).toFixed(2)}%`;
    }
    function fmtInt(value) {
      if (value === null || value === undefined || value === '') return '-';
      return String(value);
    }
    function fmtValue(value) {
      if (value === null || value === undefined || value === '') return '-';
      if (typeof value === 'boolean') return value ? '是' : '否';
      if (Array.isArray(value)) return value.join(', ');
      if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4);
      if (typeof value === 'object') return JSON.stringify(value, null, 2);
      return String(value);
    }
    function renderKvGrid(targetId, items) {
      const el = document.getElementById(targetId);
      el.innerHTML = '';
      items.forEach(([k, v, extraClass]) => {
        const div = document.createElement('div');
        div.className = 'kv-item';
        const cls = extraClass ? `v ${extraClass}` : 'v';
        div.innerHTML = `<div class="k">${escapeHtml(k)}</div><div class="${cls}">${escapeHtml(fmtValue(v))}</div>`;
        el.appendChild(div);
      });
    }
    function renderParamsTable(params) {
      const body = document.getElementById('params_table');
      body.innerHTML = '';
      Object.entries(params || {}).forEach(([k, v]) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td class="mono">${escapeHtml(k)}</td><td class="mono">${escapeHtml(fmtValue(v))}</td>`;
        body.appendChild(tr);
      });
    }
    function renderRowsTable(rows) {
      const body = document.getElementById('rows_table');
      body.innerHTML = '';
      (rows || []).forEach((row) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${escapeHtml(row.symbol)}</td>
          <td>${fmtPct(row.return_pct)}</td>
          <td>${fmtPct(row.return_ann_pct)}</td>
          <td>${fmtPct(row.max_drawdown_pct)}</td>
          <td>${fmtPct(row.win_rate_pct)}</td>
          <td>${fmtValue(row.payoff_ratio)}</td>
          <td>${fmtInt(row.trades)}</td>
        `;
        body.appendChild(tr);
      });
    }
    function renderLiveMetricsTable(rows) {
      const body = document.getElementById('live_metrics_table');
      body.innerHTML = '';
      (rows || []).slice().reverse().slice(0, 24).forEach((row) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="mono">${escapeHtml(String(row.timestamp || '').replace('T', ' ').replace('+00:00', ' UTC'))}</td>
          <td>${fmtPct(row.profit_all_percent)}</td>
          <td>${fmtPct(row.profit_closed_percent)}</td>
          <td>${fmtInt(row.open_trades)}</td>
          <td>${fmtInt(row.closed_trades)}</td>
          <td>${fmtPct(row.winrate)}</td>
          <td>${fmtPct(row.max_drawdown_percent)}</td>
        `;
        body.appendChild(tr);
      });
    }
    function renderLiveChart(rows) {
      const svg = document.getElementById('live_profit_chart');
      const meta = document.getElementById('live_chart_meta');
      const points = (rows || []).filter((row) => row.profit_all_percent !== null && row.profit_all_percent !== undefined);
      if (!points.length) {
        svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="#5a6772" font-size="20">暂无实时收益曲线数据</text>`;
        meta.innerHTML = '<span>等待策略启动并产生 API 收益快照后，这里会自动更新。</span>';
        return;
      }
      const values = points.map((row) => Number(row.profit_all_percent || 0));
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = Math.max(max - min, 1);
      const width = 1000;
      const height = 300;
      const pad = 24;
      const coords = points.map((row, idx) => {
        const x = pad + (idx / Math.max(points.length - 1, 1)) * (width - pad * 2);
        const y = height - pad - ((Number(row.profit_all_percent || 0) - min) / span) * (height - pad * 2);
        return `${x},${y}`;
      }).join(' ');
      const last = points[points.length - 1];
      const zeroY = height - pad - ((0 - min) / span) * (height - pad * 2);
      svg.innerHTML = `
        <line x1="${pad}" y1="${Math.max(pad, Math.min(height - pad, zeroY))}" x2="${width - pad}" y2="${Math.max(pad, Math.min(height - pad, zeroY))}" stroke="rgba(180,83,9,0.55)" stroke-dasharray="6 6" />
        <polyline fill="none" stroke="#0f766e" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" points="${coords}" />
        <circle cx="${coords.split(' ').slice(-1)[0].split(',')[0]}" cy="${coords.split(' ').slice(-1)[0].split(',')[1]}" r="6" fill="#b45309" />
      `;
      meta.innerHTML = `
        <span>最新总收益率：<strong>${fmtPct(last.profit_all_percent)}</strong></span>
        <span>最新已平仓收益率：<strong>${fmtPct(last.profit_closed_percent)}</strong></span>
        <span>数据点：<strong>${points.length}</strong></span>
        <span>区间：<strong>${fmtPct(min)} ~ ${fmtPct(max)}</strong></span>
      `;
    }
    function renderProfiles(profiles, selectedPath) {
      const select = document.getElementById('strategy_summary');
      select.innerHTML = '';
      profiles.forEach((profile) => {
        const option = document.createElement('option');
        option.value = profile.path;
        option.textContent = `${profile.name} | 留出收益 ${fmtPct(profile.holdout_return_pct)} | 回撤 ${fmtPct(profile.holdout_drawdown_pct)}`;
        if (profile.path === selectedPath) option.selected = true;
        select.appendChild(option);
      });

      const selected = profiles.find((item) => item.path === select.value) || profiles[0];
      const cards = document.getElementById('profile_cards');
      cards.innerHTML = '';
      if (!selected) {
        renderKvGrid('profile_meta', []);
        return;
      }
      [
        ['留出收益', fmtPct(selected.holdout_return_pct)],
        ['留出回撤', fmtPct(selected.holdout_drawdown_pct)],
        ['周正收益占比', fmtPct(Number(selected.holdout_weekly_positive_ratio || 0) * 100)],
        ['震荡切换', selected.enable_range_reversion ? '开启' : '关闭'],
      ].forEach(([k, v]) => {
        const div = document.createElement('div');
        div.className = 'stat';
        div.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
        cards.appendChild(div);
      });
      renderKvGrid('profile_meta', [
        ['策略名称', selected.name],
        ['策略文件', selected.path, 'mono'],
        ['更新时间', selected.updated_at],
        ['币种范围', selected.symbols || []],
        ['留出年化', fmtPct(selected.holdout_ann_pct)],
        ['全测试收益', fmtPct(selected.best_test_return_pct)],
        ['全测试年化', fmtPct(selected.best_test_ann_pct)],
        ['全测试回撤', fmtPct(selected.best_test_drawdown_pct)],
        ['周正收益占比', fmtPct(Number(selected.holdout_weekly_positive_ratio || 0) * 100)],
        ['月正收益占比', fmtPct(Number(selected.holdout_monthly_positive_ratio || 0) * 100)],
      ]);
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
      renderKvGrid('settings_overview', [
        ['选中策略文件', settings.selection.strategy_summary, 'mono'],
        ['交易所', settings.exchange.name],
        ['模拟盘', settings.freqtrade.dry_run],
        ['单笔资金', settings.freqtrade.stake_amount],
        ['最大持仓数', settings.freqtrade.max_open_trades],
        ['API 端口', settings.freqtrade.api_port],
        ['机器人名称', settings.freqtrade.bot_name],
        ['Telegram 已配置', Boolean(settings.telegram.token && settings.telegram.chat_id)],
      ]);

      document.getElementById('bot_status').innerHTML = `<span class="status-dot ${data.state.running ? 'on' : 'off'}"></span>${data.state.running ? '运行中' : '已停止'}`;
      document.getElementById('bot_pid').textContent = fmtInt(data.state.pid);
      document.getElementById('bot_summary').textContent = data.state.strategy_summary ? data.state.strategy_summary.split(/[\\\\/]/).pop() : '-';
      document.getElementById('bot_prepare').textContent = data.state.last_prepare || '-';
      document.getElementById('log_tail').textContent = (data.log_tail || []).join('\\n') || '暂时还没有日志输出。';
      document.getElementById('prepared_json').textContent = JSON.stringify(data.prepared_profile || {}, null, 2);
      document.getElementById('api_json').textContent = JSON.stringify(data.live_api || {}, null, 2);

      const prepared = (data.prepared_profile || {}).portfolio || {};
      document.getElementById('prepared_return').textContent = fmtPct(prepared.return_pct);
      document.getElementById('prepared_ann').textContent = fmtPct(prepared.return_ann_pct);
      document.getElementById('prepared_dd').textContent = fmtPct(prepared.max_drawdown_pct);
      document.getElementById('prepared_trades').textContent = fmtInt(prepared.trades);
      renderParamsTable((data.prepared_profile || {}).best_params || {});
      renderRowsTable((data.prepared_profile || {}).rows || []);

      const liveCount = (data.live_api || {}).count || {};
      const liveProfit = (data.live_api || {}).profit || {};
      const liveStatus = (data.live_api || {}).status || [];
      const liveMetric = data.live_metric || {};
      const liveMetricsHistory = data.live_metrics_history || [];
      document.getElementById('api_open').textContent = fmtInt(liveCount.current);
      document.getElementById('api_closed').textContent = fmtInt(liveCount.total);
      document.getElementById('api_profit').textContent = liveProfit.profit_closed_coin !== undefined ? String(liveProfit.profit_closed_coin) : '-';
      document.getElementById('api_status_rows').textContent = Array.isArray(liveStatus) ? String(liveStatus.length) : '-';
      renderKvGrid('api_detail', [
        ['可用交易位', liveCount.max_open_trades ?? '-'],
        ['当前开启交易', liveCount.current ?? '-'],
        ['累计已关闭交易', liveCount.total ?? '-'],
        ['已关闭收益', liveProfit.profit_closed_coin ?? '-'],
        ['已关闭收益率', liveProfit.profit_closed_ratio ?? '-'],
        ['实盘状态条目数', Array.isArray(liveStatus) ? liveStatus.length : 0],
        ['当前进程 PID', data.state.pid ?? '-'],
        ['日志文件', (data.artifacts || {}).log || '-', 'mono'],
      ]);
      renderKvGrid('live_stats_overview', [
        ['总收益率', fmtPct(liveMetric.profit_all_percent)],
        ['已平仓收益率', fmtPct(liveMetric.profit_closed_percent)],
        ['当前持仓数', liveMetric.open_trades ?? '-'],
        ['累计已平仓', liveMetric.closed_trades ?? '-'],
        ['胜率', fmtPct(liveMetric.winrate)],
        ['盈利因子', fmtValue(liveMetric.profit_factor)],
        ['最大回撤', fmtPct(liveMetric.max_drawdown_percent)],
        ['最新快照时间', liveMetric.timestamp ? String(liveMetric.timestamp).replace('T', ' ').replace('+00:00', ' UTC') : '-'],
      ]);
      renderLiveMetricsTable(liveMetricsHistory);
      renderLiveChart(liveMetricsHistory);

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
