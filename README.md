# paper-trading-observatory

Python CLI for paper trading observability, execution, and analysis.

## Commands

- `python main.py collect --symbols AAPL MSFT NVDA --start 2024-01-01 --end 2024-12-31`
- `python main.py run --symbols AAPL MSFT NVDA --mode backtest --bars data/cache/`
- `python main.py run --symbols AAPL MSFT NVDA --mode paper --bars data/cache/ --model mistral --timeout 3000`
- `python main.py analyze`

## Notes

- `collect` stores cached minute bars in `data/cache/` as parquet files.
- `run --mode backtest` uses cached bars and simulates fills at the close price.
- `run --mode paper` submits Alpaca paper orders.
- `analyze` reads `instrumentation/traces.db` and writes CSV/HTML outputs to `paper/`.

## Troubleshooting

- Install the project dependencies before running the CLI or tests.
- If parquet loading fails, verify the local `pandas` / `pyarrow` / `numpy` versions are compatible.
- If paper mode fails, confirm the Alpaca SDK is installed and `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and `ALPACA_BASE_URL` are set.
