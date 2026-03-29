# Recovery Summary

Recovery target: `G:\金融量化工程`

Recovery destination: `G:\qlib-ai-trade\imported_assets\freqtrade_extract`

Recovery date: `2026-03-29`

## Recovered Assets

The following files were successfully extracted into this directory:

- `freqtrade_commit`
- `tradesv3_fourcoin_awl_chan_ml_okx_dryrun.sqlite`
- `tradesv3_fourcoin_awl_chan_ml_okx_live.sqlite`
- `runtime_capture/host_server.log`
- `runtime_capture/host_server.err.log`

Also added for handoff and rebuild reference:

- `docker_runtime_notes.md`
- `docker-compose.recovered.yml`
- `runtime_capture/docker_ps.txt`

## Main Result

No trained model artifact was found in the recoverable sources.

I specifically did not find any recoverable files matching the usual model/output locations such as:

- `*.pkl`
- `*.joblib`
- `*.pt`
- `*.pth`
- `*.onnx`
- `user_data/models`
- `user_data/freqai`
- `user_data/strategies`
- `backtest_results`

What was recoverable is runtime evidence, version fingerprints, historical database files, and logs.

Note:

- The final recovery directory contains the 2 main sqlite database files themselves.
- Temporary sqlite sidecar files such as `-shm` and `-wal` were not present in the final directory snapshot.

## Database Check

The two historical sqlite files recovered from the running `freqtrade` container both contain table structure only.

`tradesv3_fourcoin_awl_chan_ml_okx_dryrun.sqlite`

- `KeyValueStore = 0`
- `orders = 0`
- `pairlocks = 0`
- `trade_custom_data = 0`
- `trades = 0`

`tradesv3_fourcoin_awl_chan_ml_okx_live.sqlite`

- `KeyValueStore = 0`
- `orders = 0`
- `pairlocks = 0`
- `trade_custom_data = 0`
- `trades = 0`

This means these 2 databases are useful as project fingerprints, but not as actual trade history payloads.

## Confirmed Runtime Clues

- `freqtrade_commit` contains commit `0e2313be7b610540fe7ef215542bd212d0e131fb`
- Running bot version observed from logs: `2026.2`
- Current running command observed from container inspect:
  `freqtrade trade --logfile /freqtrade/user_data/logs/freqtrade.log --db-url sqlite:////freqtrade/user_data/tradesv3.sqlite --config /freqtrade/user_data/config.json --strategy SampleStrategy`
- Docker compose project metadata points back to `G:\金融量化工程`
- Historical database filenames show an older project/strategy naming pattern:
  `fourcoin_awl_chan_ml_okx`
- Runtime logs show Binance Futures `5m` market requests and a 20-pair whitelist

## Important Missing Pieces

These files or directories were referenced by the live containers, but were not recoverable from the host path anymore:

- `G:\金融量化工程\user_data\config.json`
- `G:\金融量化工程\manage_api\server.py`
- `G:\金融量化工程\docker-compose.yml`
- `G:\金融量化工程\ui\...`
- the effective contents of `G:\金融量化工程\user_data\tradesv3.sqlite`
- the effective contents of `G:\金融量化工程\user_data\logs\freqtrade.log`

## Practical Conclusion

This recovery package preserves the core evidence that still existed on `2026-03-29`, but it does not contain a usable trained model or the original strategy source code.

If you want to continue searching for the missing model, the next best places are:

- Docker Desktop internal storage / volume cache
- browser cache or local storage from the UI machine
- any backup of `config.json`, `server.py`, `ui`, or `user_data/freqai`
