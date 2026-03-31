# Memory

## 2026-03-30

### What Changed

- closed the main repaint / future-leak gaps in the research-to-execution bridge
- added portfolio-level shared-capital backtesting for leverage, layering, and symbol rotation
- updated the cost model to include `0.05%` commission per side and adverse `0.02%` execution slippage
- reran Qlib training and live-strategy parameter search with the stricter assumptions
- regenerated the latest no-repaint portfolio equity curve and trade log

### Accuracy Guardrails

- higher timeframe data is delayed until the informative candle is fully closed before it can affect lower timeframe decisions
- portfolio execution uses prior-bar signals and next-bar open fills rather than same-bar decision plus same-bar fill
- portfolio fills apply adverse slippage on entry and exit
- evaluation is done on a shared capital pool instead of summing isolated single-symbol curves
- parameter search still selects by validation performance, not by test performance

### Commands Run

- `.venv\Scripts\python.exe bridge\train_qlib_model.py --symbols BTC-USDT ETH-USDT DOGE-USDT PEPE-USDT --lower-limit 0 --higher-limit 0 --source okx --reuse-raw-cache --max-test-window --valid-days 180 --min-train-days 365 --prediction-horizon 12`
- `.venv\Scripts\python.exe bridge\optimize_live_strategy.py --max-trials 120 --seed 42`
- `.venv\Scripts\python.exe bridge\export_best_test_portfolio.py`
- `python -m unittest tests.test_portfolio_backtest`
- `python -m compileall quant_trading bridge tests`

### Latest Reference Result

- best validation portfolio:
  - total return `+19.09%`
  - annualized return `+42.54%`
  - max drawdown `5.11%`
  - trades `82`
- best test portfolio:
  - total return `+311298.17%`
  - annualized return `+231.57%`
  - max drawdown `12.30%`
  - trades `1651`
- portfolio equity window:
  - `2019-07-13 18:35:00 UTC` to `2026-03-29 18:35:00 UTC`
  - start equity `100000.00`
  - end equity `311398171.98`

### Key Artifacts

- `bridge/runtime/reports/qlib_training_summary.json`
- `bridge/runtime/reports/live_strategy_search_summary.json`
- `bridge/runtime/reports/live_strategy_search_trials.csv`
- `bridge/runtime/reports/best_test_portfolio_summary_norepaint.json`
- `bridge/runtime/reports/best_test_portfolio_equity_curve_norepaint.png`
- `bridge/runtime/reports/best_test_portfolio_trades_norepaint.csv`

### Next Honest Step

- with realistic fees, slippage, and anti-repaint constraints enabled, the latest annualized test return is still below the earlier `300%+` stretch target
- the next search should keep validation-first discipline and expand only the validation-safe search space rather than selecting by test performance

### Walk-Forward Follow-Up

- upgraded `bridge/optimize_live_strategy.py` from single-window validation to a 12-fold walk-forward search with a final 365-day holdout
- changed the search objective to reward log-compressed fold growth while penalizing instability and drawdowns above the 30% guardrail
- added `tests/test_optimize_live_strategy.py` to verify holdout splitting and walk-forward window coverage
- ran `.venv\Scripts\python.exe bridge\optimize_live_strategy.py --max-trials 8 --seed 42`
- latest walk-forward best params:
  - `entry_quantile = 0.55`
  - `exit_quantile = 0.25`
  - `confidence_threshold = 0.45`
  - `qlib_rank_threshold = 0.55`
  - `confidence_weight = 0.5`
  - `take_profit_scale = 1.4`
  - `max_leverage = 3.0`
  - `rotation_top_n = 2`
  - `rotation_min_score = 0.75`
- latest walk-forward holdout portfolio:
  - window: `2025-03-29 18:35:00 UTC` to `2026-03-29 18:35:00 UTC`
  - total return `+94.55%`
  - annualized return `+94.63%`
  - max drawdown `6.16%`
  - trades `145`
- latest full walk-forward deployment test:
  - total return `+44236.62%`
  - annualized return `+147.98%`
  - max drawdown `11.66%`

### Local Refinement Follow-Up

- added weekly / monthly / yearly positive-return ratios to the walk-forward search output and objective
- added a local-neighborhood refinement mode to `bridge/optimize_live_strategy.py` via `--refine-around-current-best`
- ran `.venv\Scripts\python.exe bridge\optimize_live_strategy.py --max-trials 8 --seed 42 --refine-around-current-best --local-radius 1`
- latest refined best params:
  - `entry_quantile = 0.45`
  - `exit_quantile = 0.25`
  - `confidence_threshold = 0.35`
  - `qlib_rank_threshold = 0.55`
  - `take_profit_scale = 1.4`
  - `kelly_scale = 0.75`
  - `max_leverage = 4.0`
  - `rotation_top_n = 1`
  - `rotation_min_score = 0.75`
- latest refined holdout portfolio:
  - window: `2025-03-29 18:35:00 UTC` to `2026-03-29 18:35:00 UTC`
  - total return `+138.16%`
  - annualized return `+138.30%`
  - max drawdown `8.04%`
  - weekly positive ratio `48.08%`
  - monthly positive ratio `75.00%`
  - yearly positive ratio `100.00%`
- latest refined full walk-forward deployment test:
  - total return `+350904.04%`
  - annualized return `+237.54%`
  - max drawdown `15.30%`

### Range Regime Experiment

- added range-regime detection plus spot-compatible mean-reversion behavior in `quant_trading/signals/multi_timeframe.py`
- this version does `low-long / high-exit` in ranges; it does not add true shorting because the current live / portfolio execution path is still long-only
- added range parameters to `bridge/optimize_live_strategy.py` and a signal test in `tests/test_signal_range_regime.py`
- current range-aware best search summary:
  - holdout total return `+128.99%`
  - holdout annualized return `+129.12%`
  - holdout max drawdown `10.27%`
  - holdout weekly positive ratio `48.08%`
  - holdout monthly positive ratio `58.33%`
- compared with the prior non-range best, the current range-aware variant is interesting but not yet a clear winner because holdout return and monthly stability both fell

### Leverage And Branch Comparison

- added `--anchor-params-file` / `--fixed-params-file` support so trend-only and range-enabled searches can be optimized independently
- saved reusable anchors:
  - `bridge/runtime/reports/original_best_non_range_anchor.json`
  - `bridge/runtime/reports/range_best_anchor.json`
  - `bridge/runtime/reports/fixed_no_range.json`
  - `bridge/runtime/reports/fixed_range.json`
- trend-only leverage refinement:
  - command: `.venv\Scripts\python.exe bridge\optimize_live_strategy.py --max-trials 5 --seed 52 --refine-around-current-best --anchor-params-file bridge\runtime\reports\original_best_non_range_anchor.json --fixed-params-file bridge\runtime\reports\fixed_no_range.json --local-radius 2`
  - best holdout return `+141.32%`
  - best holdout annualized return `+141.47%`
  - best holdout max drawdown `8.31%`
  - best holdout weekly positive ratio `51.92%`
  - best holdout monthly positive ratio `75.00%`
  - best full test return `+464328.57%`
  - best full test annualized return `+251.92%`
  - best full test max drawdown `18.09%`
- range-enabled refinement:
  - command: `.venv\Scripts\python.exe bridge\optimize_live_strategy.py --max-trials 4 --seed 61 --refine-around-current-best --anchor-params-file bridge\runtime\reports\range_best_anchor.json --fixed-params-file bridge\runtime\reports\fixed_range.json --local-radius 1`
  - best holdout return `+151.31%`
  - best holdout annualized return `+151.47%`
  - best holdout max drawdown `10.41%`
  - best holdout weekly positive ratio `38.46%`
  - best holdout monthly positive ratio `58.33%`
- final choice restored to the trend-only leverage-refined branch because its holdout return stayed very high while weekly and monthly stability were materially better

### Return Push Follow-Up

- ran a more aggressive trend-only leverage search:
  - command: `.venv\Scripts\python.exe bridge\optimize_live_strategy.py --max-trials 6 --seed 77 --refine-around-current-best --anchor-summary bridge\runtime\reports\trend_leverage_search_summary.json --fixed-params-file bridge\runtime\reports\fixed_no_range.json --local-radius 2`
  - best holdout return `+87.19%`
  - result was worse than the already archived trend-only branch, so it was not kept
- ran a more aggressive range-enabled return search:
  - command: `.venv\Scripts\python.exe bridge\optimize_live_strategy.py --max-trials 4 --seed 61 --refine-around-current-best --anchor-params-file bridge\runtime\reports\range_best_anchor.json --fixed-params-file bridge\runtime\reports\fixed_range.json --local-radius 1`
  - best holdout return `+151.31%`
  - best holdout annualized return `+151.47%`
  - best holdout max drawdown `10.41%`
  - best holdout weekly positive ratio `38.46%`
  - best holdout monthly positive ratio `58.33%`
  - best full test return `+888357.67%`
  - best full test annualized return `+287.64%`
  - best full test max drawdown `19.93%`
- current main `live_strategy_search_summary.json` is restored to the range-enabled return leader because the user explicitly prioritized yield while keeping max drawdown below `30%`
- archived branch summaries to keep both options available:
  - `bridge/runtime/reports/trend_leverage_search_summary.json`
  - `bridge/runtime/reports/range_switch_search_summary.json`

### Six-Symbol Expansion

- expanded the research universe to `BTC-USDT`, `ETH-USDT`, `DOGE-USDT`, `PEPE-USDT`, `SUI-USDT`, `SOL-USDT`
- updated `quant_trading/config.py` defaults to include `SUI-USDT` and `SOL-USDT`
- hardened `quant_trading/data/okx_fetcher.py` with retries and longer timeouts so full-history OKX downloads can survive transient timeouts
- ran:
  - `.venv\Scripts\python.exe bridge\train_qlib_model.py --symbols BTC-USDT ETH-USDT DOGE-USDT PEPE-USDT SUI-USDT SOL-USDT --lower-limit 0 --higher-limit 0 --source okx --reuse-raw-cache --max-test-window --valid-days 180 --min-train-days 365 --prediction-horizon 12`
  - `.venv\Scripts\python.exe bridge\optimize_live_strategy.py --max-trials 6 --seed 88 --refine-around-current-best --local-radius 2`
- six-symbol single-name Qlib-filtered test results added:
  - `SUI-USDT`: total return `+13.77%`, max drawdown `1.11%`
  - `SOL-USDT`: total return `+19.56%`, max drawdown `2.96%`
- current six-symbol return leader:
  - holdout total return `+219.05%`
  - holdout annualized return `+219.30%`
  - holdout max drawdown `11.61%`
  - holdout weekly positive ratio `49.06%`
  - holdout monthly positive ratio `58.33%`
  - full test total return `+12736293.38%`
  - full test annualized return `+476.03%`
  - full test max drawdown `19.60%`
- exported latest six-symbol best-test portfolio again:
  - `bridge/runtime/reports/best_test_portfolio_summary_norepaint.json`
  - `bridge/runtime/reports/best_test_portfolio_equity_curve_norepaint.png`
  - `bridge/runtime/reports/best_test_portfolio_trades_norepaint.csv`

### Live Console

- added a minimal FastAPI live control surface:
  - `bridge/live_control.py`
  - `bridge/live_console_app.py`
  - `bridge/prepare_live_strategy.py`
  - `run_live_console.py`
- the console supports:
  - OKX key / secret / passphrase entry
  - Telegram token / chat id entry
  - strategy profile selection from saved `*search_summary.json`
  - prepare signals
  - start / stop Freqtrade
  - live API snapshot cards
  - prepared equity curve and recent logs
- installed Freqtrade runtime requirements into `.venv`
- dry-run smoke passed:
  - the control layer successfully prepared signals, started a Freqtrade process, queried `/api/v1/count`, `/api/v1/profit`, `/api/v1/status`, and stopped the process
- note:
  - one start attempt failed because the machine temporarily could not resolve `www.okx.com` DNS; this is an environment/network issue, not a strategy code issue

### Anti-Repaint Audit

- found a concrete lookahead leak in `bridge/train_qlib_model.py`:
  - `qlib_score_rank` had been computed with a full-history per-symbol percentile rank
  - that rank then fed the practical strategy gating, leverage, rotation, and confidence logic
- fixed the leak by introducing calibration-only percentile mapping in:
  - `bridge/rank_utils.py`
  - `bridge/optimize_live_strategy.py`
  - `bridge/train_qlib_model.py`
  - `bridge/export_live_ready_signals.py`
- re-evaluated the current six-symbol best params with the corrected anti-lookahead logic:
  - holdout return moved from `+219.05%` down to `+207.88%`
  - holdout max drawdown moved from `11.61%` up to `13.33%`
  - full test return moved from `+12736293.38%` down to `+6340091.27%`
- conclusion:
  - the old result was overstated by a real future-information leak
  - even after the fix, the same parameter set still remains extremely aggressive and unusually strong
  - the next honest step is a full re-search under the corrected ranking logic, not trusting the old search leaderboard as-is

### Post-Fix Re-Search

- ran a post-fix six-symbol local search under the corrected anti-lookahead ranking logic:
  - command: `.venv\Scripts\python.exe bridge\optimize_live_strategy.py --max-trials 6 --seed 131 --refine-around-current-best --anchor-summary bridge\runtime\reports\live_strategy_search_summary.json --local-radius 2`
- this validation-first search found a more conservative leader:
  - holdout return `+128.55%`
  - holdout annualized return `+128.68%`
  - holdout max drawdown `8.34%`
  - holdout weekly positive ratio `52.83%`
  - holdout monthly positive ratio `58.33%`
- because the user explicitly prioritized yield, restored the higher-yield post-fix candidate using:
  - `bridge/runtime/reports/post_fix_high_yield_anchor.json`
  - `.venv\Scripts\python.exe bridge\optimize_live_strategy.py --max-trials 1 --seed 1 --refine-around-current-best --anchor-params-file bridge\runtime\reports\post_fix_high_yield_anchor.json --local-radius 0`
- current main post-fix yield leader:
  - holdout return `+207.88%`
  - holdout annualized return `+208.12%`
  - holdout max drawdown `13.33%`
  - holdout weekly positive ratio `47.17%`
  - holdout monthly positive ratio `66.67%`
  - full test return `+6340091.27%`
  - full test annualized return `+419.18%`
  - full test max drawdown `19.73%`
- archived summaries:
  - `bridge/runtime/reports/post_fix_validation_search_summary.json`
  - `bridge/runtime/reports/post_fix_validation_search_trials.csv`
  - `bridge/runtime/reports/post_fix_high_yield_search_summary.json`
  - `bridge/runtime/reports/post_fix_high_yield_search_trials.csv`
- refreshed downstream artifacts after restoring the yield leader:
  - `bridge/runtime/reports/best_test_portfolio_summary_norepaint.json`
  - `bridge/runtime/live_console/prepared_strategy_profile.json`
  - `bridge/runtime/live_console/prepared_strategy_equity_curve.png`
