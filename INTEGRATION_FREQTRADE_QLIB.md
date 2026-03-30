# Freqtrade + Qlib Integration Notes

## Goal

Combine:

- `freqtrade-develop`: exchange connectivity, live trading, risk control, dry-run/backtesting, strategy execution
- `qlib-main`: factor research, model training, rolling retraining, prediction generation

Recommended direction:

- use **Qlib as the research / training / prediction engine**
- use **Freqtrade as the execution engine**

This is the cleanest split because both projects are strong in different layers.

## Key Finding

The two projects do **not** share the same native abstraction:

- `Qlib` naturally outputs **prediction scores/signals** by instrument and timestamp
- `Freqtrade` naturally consumes **per-pair candle dataframes** and expects strategy columns such as `enter_long`, `enter_short`, `exit_long`, `exit_short`

So the best integration is not "merge the whole frameworks together" but to build a **bridge layer** between:

- `qlib prediction output`
- `freqtrade strategy input`

## Best Architecture

### Option A: External Signal Bridge

This is the recommended first implementation.

Flow:

1. Freqtrade downloads and stores crypto OHLCV data
2. A bridge script converts that data into a Qlib-usable dataset
3. Qlib trains a model and produces predictions
4. Predictions are exported into a shared store:
   - parquet
   - sqlite
   - redis
   - or a small local API service
5. A custom Freqtrade strategy loads the latest score for the current pair/timeframe in `populate_indicators()`
6. The strategy converts score/rank/probability into `enter_long` / `enter_short` / exits

Why this is best:

- lowest coupling
- easiest to debug
- preserves Qlib workflow almost intact
- preserves Freqtrade live execution and risk management almost intact
- easiest to run in dry-run first

### Option B: Wrap Qlib inside a custom FreqAI model

Possible, but not the first choice.

Implementation idea:

- create `user_data/freqaimodels/QlibFreqaiModel.py`
- subclass `IFreqaiModel`
- inside `fit()` / `train()` / `predict()`, call a Qlib model and dataset

Pros:

- fits into native FreqAI lifecycle
- can reuse `self.freqai.start()`
- predictions come back directly inside Freqtrade dataframe

Cons:

- much more coupling
- FreqAI feature pipeline and Qlib dataset pipeline overlap
- Qlib is often cross-sectional, while FreqAI is heavily pair-centric
- much harder to keep training windows, metadata, and persistence clean

Recommendation:

- only do this after Option A is stable

## Why Option A is Better Than Immediate FreqAI Wrapping

`FreqAI` already wants to own:

- feature engineering
- training window management
- model persistence
- prediction lifecycle

`Qlib` also wants to own:

- data preparation
- dataset definition
- model training
- signal generation
- online update / rolling training

If both own the same layer, engineering gets messy fast.

So the better split is:

- Qlib owns model science
- Freqtrade owns market execution

## Data Alignment Problems We Must Solve

### 1. Symbol mapping

We need a canonical mapping between:

- Qlib instrument IDs
- Freqtrade pair names like `BTC/USDT:USDT`
- exchange symbols like `BTCUSDT`

Create one shared mapping file.

### 2. Timestamp alignment

Predictions must be aligned to the exact candle close used by Freqtrade.

Rule:

- only allow Qlib to use candles that were fully closed before Freqtrade makes the decision
- never use future bars accidentally

### 3. Target horizon alignment

If Qlib predicts "future return over next N candles", then Freqtrade entry/exit logic must use the same horizon assumption.

### 4. Fee / slippage / futures assumptions

Qlib research metrics are not enough by themselves.

Before trusting any model, replay its signals through Freqtrade backtesting or dry-run with:

- actual exchange fee model
- leverage rules
- stoploss logic
- order fill assumptions

### 5. Crypto data quality

Local Qlib crypto collector is not enough for this project because it is daily and does not provide the full OHLC-based backtest path we need.

For production work, the source of truth should be exchange OHLCV data, ideally the same data family Freqtrade already uses.

## Practical MVP

### Phase 1: Shared Data Layer

Build:

- `bridge/etl_freqtrade_to_qlib.py`

Responsibilities:

- read Freqtrade OHLCV data from `freqtrade-develop/user_data/data/...`
- normalize pair names
- export a Qlib-friendly dataset or direct training dataframe

### Phase 2: Qlib Training Job

Build:

- `bridge/train_qlib_model.py`

Responsibilities:

- train one Qlib model on chosen timeframe
- save model artifact
- save prediction output for the latest candles

### Phase 3: Prediction Export

Build:

- `bridge/export_qlib_signals.py`

Responsibilities:

- write latest predictions into:
  - sqlite table
  - or parquet
  - or json cache

Suggested output fields:

- `timestamp`
- `pair`
- `score`
- `direction`
- `rank`
- `confidence`
- `model_version`

### Phase 4: Freqtrade Strategy Consumer

Build:

- `freqtrade-develop/user_data/strategies/QlibExternalSignalStrategy.py`

Responsibilities:

- read latest external signal for `metadata["pair"]`
- attach columns such as:
  - `qlib_score`
  - `qlib_direction`
  - `qlib_rank`
  - `qlib_confidence`
- convert these into:
  - `enter_long`
  - `enter_short`
  - optional exits

### Phase 5: Dry-Run Validation

Run the whole flow in:

- backtesting
- dry-run

Only after that should we consider live trading.

## Best Signal Form For Freqtrade

Do not start with complex portfolio instructions.

Start with a simple per-pair output:

- `score > long_threshold` => long candidate
- `score < short_threshold` => short candidate
- `abs(score) < neutral_threshold` => no trade / exit bias

Then add:

- rank filters
- confidence filters
- regime filters
- dynamic position size from score magnitude

## How Qlib Output Should Be Used

Best early use:

- Qlib predicts future return or probability of up/down move
- Freqtrade still handles:
  - trade entry
  - stoploss
  - ROI / exits
  - leverage
  - max open trades
  - position sizing

This avoids giving the ML model too much direct control too early.

## Concrete Recommendation

Start with this sequence:

1. Use Freqtrade data as the canonical candle source
2. Build a Qlib training script against those candles
3. Export latest predictions to sqlite/parquet
4. Build a Freqtrade strategy that reads those predictions
5. Validate in backtesting and dry-run
6. Only then decide whether a custom `QlibFreqaiModel` is worth building

## Current Local Pipeline

The repo now contains a working bridge that follows the recommended architecture above.

Implemented pieces:

- `run_multicycle_training.py`
  - builds the multi-timeframe heuristic feature stack
  - supports `BTC-USDT`, `ETH-USDT`, `DOGE-USDT`, and `PEPE-USDT` by default
  - writes external signals into `bridge/runtime/freqtrade_signals.parquet`
- `bridge/train_qlib_model.py`
  - loads `5m + 1H` OHLCV from OKX or local Freqtrade history
  - builds Chan / Wyckoff / Brooks / breakout / Kelly features
  - trains a Qlib `LGBModel` on pooled cross-symbol intraday data
  - calibrates entry / exit thresholds from the validation score distribution
  - scores the latest closed candles too, so the tail of the signal file is usable by Freqtrade
  - backtests both the raw heuristic signals and the Qlib-filtered signals on the test segment
- `bridge/optimize_live_strategy.py`
  - searches a practical signal profile on the validation segment
  - optimizes toward higher annualized return with drawdown and trade-count penalties
- `bridge/export_live_ready_signals.py`
  - applies the chosen live profile to the cached feature frames
  - writes per-symbol live-ready signal parquet files
  - refreshes `bridge/runtime/freqtrade_signals.parquet` for Freqtrade consumption
- `freqtrade-develop/user_data/strategies/QlibExternalSignalStrategy.py`
  - reads `entry_long_signal`, `exit_long_signal`, `signal_confidence`, `qlib_score`, and `qlib_score_rank`

Example command for the current four-symbol universe:

```powershell
.venv\Scripts\python.exe bridge/train_qlib_model.py `
  --symbols BTC-USDT ETH-USDT DOGE-USDT PEPE-USDT `
  --lower-limit 12000 `
  --higher-limit 3000 `
  --source okx `
  --prediction-horizon 12
```

Artifacts are written under:

- `bridge/runtime/qlib/`
- `bridge/runtime/models/`
- `bridge/runtime/reports/`

Current optimization / export flow:

```powershell
.venv\Scripts\python.exe bridge/optimize_live_strategy.py --max-trials 200
.venv\Scripts\python.exe bridge/export_live_ready_signals.py
```

## Maximum Backtest Window

The training script now supports automatically stretching the out-of-sample test segment to the largest window that still leaves the requested validation and minimum training spans:

```powershell
.venv\Scripts\python.exe bridge/train_qlib_model.py `
  --symbols BTC-USDT ETH-USDT DOGE-USDT PEPE-USDT `
  --lower-limit 0 `
  --higher-limit 0 `
  --source okx `
  --reuse-raw-cache `
  --max-test-window `
  --valid-days 180 `
  --min-train-days 365 `
  --prediction-horizon 12
```

Latest max-window run:

- available OKX history:
  - `BTC/ETH`: about `2999` days
  - `DOGE`: about `2446` days
  - `PEPE`: about `1063` days
- effective out-of-sample test window:
  - `2019-07-13 18:35:00 UTC` to `2026-03-29 18:35:00 UTC`
  - `effective_test_days = 2451`
- pooled Qlib model metrics on that test window:
  - `Pearson IC ~= 0.5534`
  - `directional accuracy ~= 70.18%`

Raw Qlib-filtered test results on the max window:

- `BTC-USDT`: `+33.27%`, annualized `+4.37%`, max drawdown `2.05%`
- `ETH-USDT`: `+2.99%`, annualized `+0.44%`, max drawdown `0.05%`
- `DOGE-USDT`: `+35.19%`, annualized `+4.60%`, max drawdown `3.46%`
- `PEPE-USDT`: `-4.10%`, annualized `-1.43%`, max drawdown `6.56%`

Latest live-ready profile test results on the same max window:

- `BTC-USDT`: `+54.59%`, annualized `+6.70%`, max drawdown `1.02%`
- `ETH-USDT`: `+7.92%`, annualized `+1.14%`, max drawdown `0.97%`
- `DOGE-USDT`: `+102.31%`, annualized `+11.08%`, max drawdown `6.36%`
- `PEPE-USDT`: `+65.96%`, annualized `+18.98%`, max drawdown `4.73%`

Notes:

- `DOGE` and `PEPE` start later than `BTC/ETH`, so the max-window validation segment has no symbol-local validation data for those pairs.
- `bridge/export_live_ready_signals.py` now falls back to the pooled global validation score distribution for threshold calibration when a symbol-local validation slice is empty.
- The long-window profile is materially safer than the user's `max drawdown < 30%` target, but it is still far below the requested `300%+ annualized return` target, so more aggressive alpha / leverage / portfolio rotation work is still required before calling this live-ready in that sense.

## Leverage, Layering, Rotation

The bridge now carries practical execution controls all the way from research frames into Freqtrade:

- `quant_trading/signals/multi_timeframe.py`
  - adds `target_leverage`
  - adds `entry_stake_fraction`, `layer_stake_fraction`, `reduce_stake_fraction`
  - adds `add_position_signal`, `reduce_position_signal`
  - adds `partial_take_profit_pct`
  - adds cross-symbol `rotation_score`, `rotation_rank`, `rotation_weight`
  - adds `apply_rotation_overlay(...)` so only the highest-ranked symbols keep the final entry signal
- `quant_trading/backtest/engine.py`
  - supports layered entries
  - supports partial exits
  - approximates per-signal leverage using a max-leverage account margin model
  - now also exposes `run_portfolio_backtest(...)` for shared-cash cross-symbol evaluation
- `bridge/signal_store.py`
  - exports the new execution-control columns to `bridge/runtime/freqtrade_signals.parquet`
- `bridge/export_live_ready_signals.py`
  - now writes `live_ready_portfolio_summary.json`
  - now writes `live_ready_portfolio_equity.csv`
  - now writes `live_ready_portfolio_trades.csv`
- `bridge/optimize_live_strategy.py`
  - now scores candidate profiles on portfolio-level return / drawdown / trade-count first
  - keeps per-symbol rows as diagnostics, instead of treating them as the main objective
- `freqtrade-develop/user_data/strategies/QlibExternalSignalStrategy.py`
  - enables `position_adjustment_enable = True`
  - implements `leverage(...)`
  - implements `custom_stake_amount(...)`
  - implements `adjust_trade_position(...)`
  - prevents repeated add/reduce orders on the same candle via trade custom-data

Current default exported live profile:

- profile: `practical_live_v3_search_best`
- leverage / layering:
  - `max_leverage = 4.0`
  - `initial_entry_fraction = 0.45`
  - `layer_stake_fraction = 0.30`
  - `max_entry_layers = 3`
  - `reduce_stake_fraction = 0.35`
  - `partial_take_profit_scale = 0.55`
- rotation:
  - `rotation_top_n = 1`
  - `rotation_min_score = 0.55`

Latest max-window results for `practical_live_v3_search_best`:

- `BTC-USDT`: `+590.71%`, annualized `+33.33%`, max drawdown `8.93%`
- `ETH-USDT`: `+12106.20%`, annualized `+104.46%`, max drawdown `6.04%`
- `DOGE-USDT`: `+10869.88%`, annualized `+101.52%`, max drawdown `10.12%`
- `PEPE-USDT`: `+2013.40%`, annualized `+184.79%`, max drawdown `11.24%`

Important caveat:

- on the max-window split, the validation segment is `2019-01-14` to `2019-07-13`
- `DOGE` starts near the end of July 2019, and `PEPE` starts in 2023
- this means validation-time parameter search is dominated by `BTC/ETH`
- the exported `practical_live_v3_search_best` profile looks strong on the full test, but it should still be treated as a high-aggression candidate pending a more realistic rolling / recent validation scheme
- the very large per-symbol max-window returns shown above are still useful as single-symbol diagnostics, but they are not a true shared-capital portfolio result
- after re-running the export / search scripts, the files under `bridge/runtime/reports/live_ready_portfolio_*` should be treated as the primary rotation-aware evaluation output

## Current Cost-Aware No-Repaint Reference

The latest trustworthy reference run now uses the stricter rules below:

- no higher-timeframe repaint:
  - informative candles are only merged after the higher timeframe bar is fully closed
- no same-bar execution:
  - the portfolio backtest consumes the prior bar signal and fills at the next bar open
- realistic costs:
  - `commission = 0.0005` per side (`0.05%`)
  - adverse execution slippage uses `spread = 0.0002` (`0.02%`)
- shared capital portfolio evaluation:
  - symbols compete for the same cash and margin pool

Latest full search rerun on `2026-03-30`:

- command:
  - `.venv\Scripts\python.exe bridge/optimize_live_strategy.py --max-trials 120 --seed 42`
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
- equity path:
  - start `100000.00`
  - end `311398171.98`
  - window `2019-07-13 18:35:00 UTC` to `2026-03-29 18:35:00 UTC`

Reference artifacts:

- summary:
  - `bridge/runtime/reports/live_strategy_search_summary.json`
  - `bridge/runtime/reports/best_test_portfolio_summary_norepaint.json`
- equity / trades:
  - `bridge/runtime/reports/best_test_portfolio_equity_curve_norepaint.png`
  - `bridge/runtime/reports/best_test_portfolio_trades_norepaint.csv`

This cost-aware no-repaint result is the current primary reference. Earlier numbers elsewhere in this document that were produced before the anti-repaint and execution-cost fixes should not be treated as the latest truth.

## Local Code References

Useful files already present in this repo:

- `freqtrade-develop/freqtrade/freqai/freqai_interface.py`
- `freqtrade-develop/freqtrade/templates/FreqaiExampleStrategy.py`
- `freqtrade-develop/docs/freqai-developers.md`
- `freqtrade-develop/docs/freqai-configuration.md`
- `qlib-main/examples/workflow_by_code.py`
- `qlib-main/qlib/workflow/record_temp.py`
- `qlib-main/docs/component/online.rst`
- `qlib-main/qlib/contrib/online/online_model.py`
- `qlib-main/scripts/data_collector/crypto/README.md`

## Final Judgment

The two projects can absolutely be combined.

But the correct combination is:

- **Qlib for alpha/model generation**
- **Freqtrade for execution**
- **a custom bridge in the middle**

That is the shortest path to something both powerful and maintainable.
