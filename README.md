# paper-trading-observatory

Paper Trading Observatory is a Python CLI for collecting minute bars, generating trading signals, executing paper orders, tracing latency, and analyzing results.

## What It Does

- Downloads and caches Alpaca minute bars for a configurable universe.
- Computes lightweight market features such as returns, RSI, MACD, realized volatility, and intrabar range.
- Produces deterministic heuristic signals and optional LLM-backed signals.
- Routes signal generation through an escalation policy that can skip LLM calls when latency is degraded or market conditions are unsuitable.
- Executes simulated or paper trades and logs every decision to SQLite.
- Generates latency and trading reports with CSV exports and Plotly HTML artifacts.

## Project Layout

- `agent/` contains the deterministic and LLM signal agents.
- `broker/` contains the paper executor and trading loop.
- `data/` contains Alpaca ingestion and feature engineering.
- `instrumentation/` contains span tracing, metrics, and the SQLite trace store.
- `scheduler/` contains the escalation policy.
- `analysis/` contains reporting and chart generation.
- `paper/` stores generated reports and HTML plots.
- `tests/` contains the pytest suite.

## CLI

### Collect bars

```bash
python main.py collect --symbols AAPL MSFT NVDA --start 2024-01-01 --end 2024-12-31
```

Downloads minute bars for each symbol and caches them under `data/cache/` as parquet files.

### Run the trading loop

```bash
python main.py run --symbols AAPL MSFT NVDA --mode backtest --bars data/cache/
```

Backtest mode stays fully local: it reads cached bars, computes features, applies the heuristic path, simulates fills at the close price, and logs results to SQLite.

```bash
python main.py run --symbols AAPL MSFT NVDA --mode paper --bars data/cache/ --model mistral --timeout 3000
```

Paper mode uses Alpaca paper trading and can call Ollama through the LLM agent when the escalation policy allows it.

### Analyze results

```bash
python main.py analyze
```

Prints latency and trading summaries in the terminal and writes:

- `paper/latency_report.csv`
- `paper/trading_report.csv`
- `paper/latency_dist.html`
- `paper/rolling_p99.html`

## Runtime Data Flow

1. `collect` loads minute bars from Alpaca and writes parquet caches.
2. `run` loads cached bars, computes features, and creates a signal per bar.
3. The escalation policy decides whether the heuristic result is enough or whether the LLM should be called.
4. The broker layer submits paper orders or simulates a backtest fill.
5. Traces and trades are written to SQLite.
6. `analyze` reads the SQLite database and produces reports.

## Environment Variables

Copy `.env.example` to `.env` and populate:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_BASE_URL`
- `OLLAMA_HOST`
- `OLLAMA_MODEL`

## Dependencies

The project targets Python 3.11+ and uses:

- `alpaca-py`
- `ollama`
- `pandas`
- `numpy`
- `scipy`
- `pydantic`
- `httpx`
- `rich`
- `plotly`
- `pytest`

## Testing

Run the suite with:

```bash
pytest
```

The current tests cover:

- tracing span persistence and latency metrics
- feature engineering invariants
- escalation policy behavior

## Notes

- Backtest mode is designed to work without Alpaca or Ollama installed.
- Paper mode requires Alpaca credentials and the Alpaca SDK.
- Optional dependencies are loaded lazily so the CLI fails with clearer messages in minimal environments.

## Troubleshooting

- Install the project dependencies before running the CLI or tests.
- If parquet loading fails, verify that `pandas`, `pyarrow`, and `numpy` are compatible in your environment.
- If paper mode fails, confirm that `alpaca-py` is installed and the Alpaca environment variables are set.
- If LLM calls fail, confirm that `ollama` is installed and `OLLAMA_HOST` / `OLLAMA_MODEL` are configured.
