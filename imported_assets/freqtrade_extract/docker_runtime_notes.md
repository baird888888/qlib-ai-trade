# Docker Runtime Notes

Observed on `2026-03-29`.

## Relevant Containers

### `freqtrade`

- image: `ghcr.io/freqtrade/freqtrade:stable`
- version seen in logs: `2026.2`
- bind mount:
  `G:\金融量化工程\user_data -> /freqtrade/user_data`
- published port:
  `127.0.0.1:18888 -> 8080/tcp`
- startup args:
  - `trade`
  - `--logfile /freqtrade/user_data/logs/freqtrade.log`
  - `--db-url sqlite:////freqtrade/user_data/tradesv3.sqlite`
  - `--config /freqtrade/user_data/config.json`
  - `--strategy SampleStrategy`

Useful files visible inside the container:

- `/freqtrade/tradesv3_fourcoin_awl_chan_ml_okx_dryrun.sqlite`
- `/freqtrade/tradesv3_fourcoin_awl_chan_ml_okx_live.sqlite`
- `/freqtrade/freqtrade_commit`

No recoverable trained model files were found during container-wide search.

### `freqtrade-mode-api`

- image: `python:3.13-alpine`
- command: `python /app/server.py`
- bind mounts:
  - `G:\金融量化工程\manage_api -> /app`
  - `G:\金融量化工程\user_data -> /workspace/user_data`
  - `G:\金融量化工程\docker-compose.yml -> /workspace/docker-compose.yml`
- important env:
  - `LISTEN_PORT=8090`
  - `LISTEN_HOST=0.0.0.0`
  - `CONFIG_PATH=/workspace/user_data/config.json`
  - `FT_API_BASE=http://freqtrade:8080/api/v1`
  - `COMPOSE_PATH=/workspace/docker-compose.yml`
  - `LIVE_CONFIRM_PHRASE=LIVE`

Important finding:

- `/app` currently contains only:
  - `host_server.log`
  - `host_server.err.log`
- `server.py` is no longer present in the mounted host path, so the original mode-api source was not recoverable.

Historical host log clue:

- `runtime_capture/host_server.log` says:
  `Freqtrade mode API listening on 0.0.0.0:18090`
- This suggests there was also a host-side or older run variant using port `18090`, while the current container configuration uses `8090`.

### `freqtrade-cn-ui`

- image: `nginx:1.27-alpine`
- bind mounts:
  - `G:\金融量化工程\ui -> /usr/share/nginx/html`
  - `G:\金融量化工程\ui\nginx.conf -> /etc/nginx/conf.d/default.conf`
- published port:
  `127.0.0.1:8899 -> 80/tcp`

Important finding:

- Earlier access logs prove `/`, `/styles.css`, and `/app.js` returned `200` on `2026-03-29`.
- Later, requests started failing with nginx `500` and a rewrite cycle toward `/index.html`.
- The UI bind-mounted source files are not recoverable from the host path now.

## Runtime Log Clues

The `freqtrade` logs showed:

- steady bot heartbeats with state `RUNNING`
- Binance Futures OHLCV fetch attempts
- timeframe clue: `5m`
- whitelist with 20 pairs:
  - `BTC/USDT:USDT`
  - `ETH/USDT:USDT`
  - `SOL/USDT:USDT`
  - `SIREN/USDT:USDT`
  - `PAXG/USDT:USDT`
  - `XRP/USDT:USDT`
  - `DOGE/USDT:USDT`
  - `RIVER/USDT:USDT`
  - `HYPE/USDT:USDT`
  - `BNB/USDT:USDT`
  - `BR/USDT:USDT`
  - `ZEC/USDT:USDT`
  - `TAO/USDT:USDT`
  - `1000PEPE/USDT:USDT`
  - `ADA/USDT:USDT`
  - `SUI/USDT:USDT`
  - `JCT/USDT:USDT`
  - `BANANAS31/USDT:USDT`
  - `LINK/USDT:USDT`
  - `AVAX/USDT:USDT`

The UI access logs showed:

- successful `GET /strategy` and `GET /mode` requests under user `admin`
- later `401` responses under user `freqtrader`

That is valuable for reconstructing how the UI and mode-api were wired together, even though the source files are gone.
