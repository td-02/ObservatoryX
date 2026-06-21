from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from analysis.profiler import ResultsProfiler
from broker.loop import TradingLoop
from data.feed import AlpacaFeed
from instrumentation.tracer import SpanTracer


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data" / "cache"
DEFAULT_DB_PATH = BASE_DIR / "instrumentation" / "traces.db"


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_cached_bars(symbols: list[str], cache_dir: Path) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        bars[symbol] = pd.read_parquet(cache_dir / f"{symbol}.parquet")
    return bars


def _build_latency_table(report: dict) -> Table:
    table = Table(title="Latency Report")
    table.add_column("operation")
    table.add_column("p50")
    table.add_column("p95")
    table.add_column("p99")
    table.add_column("mean")
    table.add_column("std")
    table.add_column("timeout_rate")
    table.add_column("error_rate")

    for operation, metrics in report.get("per_operation", {}).items():
        table.add_row(
            operation,
            f"{metrics['p50']:.2f}",
            f"{metrics['p95']:.2f}",
            f"{metrics['p99']:.2f}",
            f"{metrics['mean']:.2f}",
            f"{metrics['std']:.2f}",
            f"{metrics['timeout_rate']:.2%}",
            f"{metrics['error_rate']:.2%}",
        )
    table.add_row(
        "overall",
        "-",
        "-",
        f"{report.get('overall_pipeline_p99', 0.0):.2f}",
        "-",
        "-",
        "-",
        "-",
    )
    return table


def _build_trading_table(report: dict) -> Table:
    table = Table(title="Trading Report")
    table.add_column("metric")
    table.add_column("value")
    for key in [
        "sharpe_ratio",
        "max_drawdown",
        "total_trades",
        "escalation_rate",
        "llm_call_count",
        "heuristic_only_count",
    ]:
        value = report.get(key, 0.0)
        if key == "escalation_rate":
            display = f"{float(value):.2%}"
        elif isinstance(value, float):
            display = f"{value:.4f}"
        else:
            display = str(value)
        table.add_row(key, display)
    return table


def cmd_collect(args: argparse.Namespace) -> None:
    _ensure_cache_dir()
    feed = AlpacaFeed()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    for symbol in args.symbols:
        bars = feed.get_bars(symbol, start, end)
        bars.to_parquet(CACHE_DIR / f"{symbol}.parquet", index=False)


def cmd_run(args: argparse.Namespace) -> None:
    cache_dir = Path(args.bars)
    bars_df = _load_cached_bars(args.symbols, cache_dir)
    tracer = SpanTracer()
    loop = TradingLoop()
    loop.run(
        symbols=args.symbols,
        bars_df=bars_df,
        tracer=tracer,
        mode=args.mode,
        model=args.model,
        timeout_ms=float(args.timeout),
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    profiler = ResultsProfiler(str(DEFAULT_DB_PATH))
    console = Console()
    latency_report = profiler.latency_report()
    trading_report = profiler.trading_report()
    profiler.plot_latency_distribution()
    profiler.plot_rolling_p99()
    console.print(_build_latency_table(latency_report))
    console.print(_build_trading_table(trading_report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--symbols", nargs="+", required=True)
    collect.add_argument("--start", required=True)
    collect.add_argument("--end", required=True)
    collect.set_defaults(func=cmd_collect)

    run = subparsers.add_parser("run")
    run.add_argument("--symbols", nargs="+", required=True)
    run.add_argument("--mode", choices=["backtest", "paper"], required=True)
    run.add_argument("--bars", required=True)
    run.add_argument("--model", default="mistral")
    run.add_argument("--timeout", type=float, default=3000.0)
    run.set_defaults(func=cmd_run)

    analyze = subparsers.add_parser("analyze")
    analyze.set_defaults(func=cmd_analyze)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
