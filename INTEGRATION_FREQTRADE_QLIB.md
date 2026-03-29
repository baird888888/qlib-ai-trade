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
