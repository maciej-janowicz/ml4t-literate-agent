#!/usr/bin/env python3
"""ML4T Literate Agent - Data Curator Module.

High-performance quantitative market data ingestion, validation, and descriptive
statistics pipeline powered by Polars.

Adheres to reproducible research and literate programming standards.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import yaml

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataCurator")


@dataclass
class CuratorConfig:
    """Configuration container for market data curation."""

    symbols: List[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TLT", "GLD"]
    )
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.0425
    confidence_level: float = 0.95
    cache_dir: Path = Path("outputs/cache")
    stats_csv_path: Path = Path("outputs/summary_stats.csv")
    stats_parquet_path: Path = Path("outputs/summary_stats.parquet")
    stats_markdown_path: Path = Path("outputs/summary_stats.md")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> CuratorConfig:
        """Load configuration from a YAML file."""
        if not yaml_path.exists():
            logger.warning(
                "Config file %s not found. Proceeding with default parameters.", yaml_path
            )
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_cfg: Dict[str, Any] = yaml.safe_load(f) or {}

        universe = raw_cfg.get("universe", {})
        symbols_raw = universe.get("symbols", [])
        if symbols_raw and isinstance(symbols_raw[0], dict):
            symbols = [s["symbol"] for s in symbols_raw if "symbol" in s]
        elif symbols_raw and isinstance(symbols_raw[0], str):
            symbols = symbols_raw
        else:
            symbols = cls().symbols

        data_cfg = raw_cfg.get("data", {})
        analytics_cfg = raw_cfg.get("analytics", {})

        return cls(
            symbols=symbols,
            start_date=data_cfg.get("start_date", "2020-01-01"),
            end_date=data_cfg.get("end_date", "2024-12-31"),
            trading_days_per_year=analytics_cfg.get("trading_days_per_year", 252),
            risk_free_rate=analytics_cfg.get("risk_free_rate", 0.0425),
            confidence_level=analytics_cfg.get("var_confidence_level", 0.95),
            cache_dir=Path(data_cfg.get("cache_dir", "outputs/cache")),
            stats_csv_path=Path(data_cfg.get("stats_csv", "outputs/summary_stats.csv")),
            stats_parquet_path=Path(data_cfg.get("stats_parquet", "outputs/summary_stats.parquet")),
            stats_markdown_path=Path(data_cfg.get("stats_markdown", "outputs/summary_stats.md")),
        )


class MarketDataCurator:
    """Orchestrates data fetching, transformation, and statistical curation using Polars."""

    def __init__(self, config: CuratorConfig) -> None:
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.stats_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def fetch_data(self, force_synthetic: bool = False) -> pl.DataFrame:
        """Fetch market OHLCV and adjusted close prices.

        Tries yfinance if available; falls back smoothly to synthetic calibrated
        Brownian motion with Student-t jumps if offline or yfinance is missing.
        """
        logger.info(
            "Fetching market data for %d symbols [%s] from %s to %s",
            len(self.config.symbols),
            ", ".join(self.config.symbols),
            self.config.start_date,
            self.config.end_date,
        )

        if not force_synthetic:
            try:
                import yfinance as yf

                logger.info("Attempting online market data acquisition via yfinance...")
                raw_dfs: List[pl.DataFrame] = []

                for symbol in self.config.symbols:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=self.config.start_date,
                        end=self.config.end_date,
                        auto_adjust=False,
                    )
                    if not hist.empty:
                        hist = hist.reset_index()
                        hist.rename(
                            columns={
                                "Date": "date",
                                "Open": "open",
                                "High": "high",
                                "Low": "low",
                                "Close": "close",
                                "Adj Close": "adj_close",
                                "Volume": "volume",
                            },
                            inplace=True,
                        )
                        hist["symbol"] = symbol
                        # Convert to Polars DataFrame
                        pldf = pl.from_pandas(hist[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]])
                        # Normalize date to Date type
                        pldf = pldf.with_columns(pl.col("date").cast(pl.Date))
                        raw_dfs.append(pldf)

                if raw_dfs:
                    combined = pl.concat(raw_dfs).sort(["symbol", "date"])
                    logger.info("Successfully acquired %d rows from yfinance.", combined.height)
                    return combined
            except Exception as e:
                logger.warning(
                    "Online fetch failed (%s). Falling back to calibrated synthetic market generator.", e
                )

        return self._generate_synthetic_market_data()

    def _generate_synthetic_market_data(self) -> pl.DataFrame:
        """Generate statistically representative asset price paths.

        Uses Geometric Brownian Motion with Student-t fat-tail innovations
        calibrated to stylized empirical facts of financial asset returns.
        """
        logger.info("Generating calibrated synthetic market series for reproducible testing...")
        np.random.seed(42)  # Strictly deterministic for academic reproducibility

        start = dt.date.fromisoformat(self.config.start_date)
        end = dt.date.fromisoformat(self.config.end_date)
        days = (end - start).days

        # Generate business trading days (excluding weekends)
        date_list: List[dt.date] = [
            start + dt.timedelta(days=i)
            for i in range(days + 1)
            if (start + dt.timedelta(days=i)).weekday() < 5
        ]
        n_obs = len(date_list)

        # Realistic parameter calibration per symbol
        calibrations = {
            "SPY": {"s0": 320.0, "mu": 0.12, "sigma": 0.18, "df": 5},
            "QQQ": {"s0": 210.0, "mu": 0.18, "sigma": 0.23, "df": 4},
            "IWM": {"s0": 165.0, "mu": 0.08, "sigma": 0.24, "df": 4},
            "AAPL": {"s0": 75.0, "mu": 0.22, "sigma": 0.28, "df": 5},
            "MSFT": {"s0": 160.0, "mu": 0.20, "sigma": 0.25, "df": 5},
            "NVDA": {"s0": 60.0, "mu": 0.45, "sigma": 0.48, "df": 3},
            "TLT": {"s0": 138.0, "mu": -0.04, "sigma": 0.15, "df": 6},
            "GLD": {"s0": 142.0, "mu": 0.09, "sigma": 0.14, "df": 6},
        }

        frames: List[pl.DataFrame] = []
        dt_step = 1.0 / self.config.trading_days_per_year

        for symbol in self.config.symbols:
            calib = calibrations.get(
                symbol, {"s0": 100.0, "mu": 0.10, "sigma": 0.20, "df": 5}
            )
            s0 = calib["s0"]
            mu = calib["mu"]
            sigma = calib["sigma"]
            df_t = calib["df"]

            # Student-t standardized innovations for fat tails
            raw_t = np.random.standard_t(df=df_t, size=n_obs)
            std_t = raw_t / np.sqrt(df_t / (df_t - 2.0))

            # Daily drift and volatility
            daily_drift = (mu - 0.5 * sigma**2) * dt_step
            daily_diff = sigma * np.sqrt(dt_step) * std_t
            log_returns = daily_drift + daily_diff
            log_returns[0] = 0.0

            prices = s0 * np.exp(np.cumsum(log_returns))
            volumes = np.random.lognormal(mean=16.0, sigma=0.5, size=n_obs).astype(np.int64)

            df_symbol = pl.DataFrame(
                {
                    "date": date_list,
                    "symbol": [symbol] * n_obs,
                    "open": prices * (1.0 + np.random.normal(0, 0.002, n_obs)),
                    "high": prices * (1.0 + np.abs(np.random.normal(0, 0.008, n_obs))),
                    "low": prices * (1.0 - np.abs(np.random.normal(0, 0.008, n_obs))),
                    "close": prices,
                    "adj_close": prices,
                    "volume": volumes,
                }
            )
            frames.append(df_symbol)

        combined = pl.concat(frames).sort(["symbol", "date"])
        logger.info("Generated %d rows of calibrated synthetic market data.", combined.height)
        return combined

    def compute_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute arithmetic and logarithmic returns using vectorized Polars expressions."""
        logger.info("Computing returns and financial series...")
        return (
            df.sort(["symbol", "date"])
            .with_columns(
                # Simple arithmetic return: (P_t - P_{t-1}) / P_{t-1}
                (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1.0)
                .alias("arithmetic_return"),
                # Continuous log return: ln(P_t / P_{t-1})
                (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol"))
                .log()
                .alias("log_return"),
            )
            .filter(pl.col("log_return").is_not_null())
        )

    def compute_descriptive_statistics(self, df_returns: pl.DataFrame) -> pl.DataFrame:
        """Compute publication-ready quantitative descriptive statistics.

        Leverages Polars aggregation engine for high-throughput calculation of:
        - Observations (N)
        - Mean daily return and annualized return
        - Daily and annualized volatility
        - Skewness and excess kurtosis (stylized fat tails)
        - Historical Value at Risk (VaR 95%)
        - Annualized Sharpe Ratio
        - Maximum Drawdown (MDD)
        """
        logger.info("Computing descriptive statistics and stylized facts...")

        ann_factor = float(self.config.trading_days_per_year)
        sqrt_ann = float(np.sqrt(ann_factor))
        rf = float(self.config.risk_free_rate)
        alpha = float(1.0 - self.config.confidence_level)

        # 1. Base aggregations via Polars native expressions
        stats = (
            df_returns.group_by("symbol", maintain_order=True)
            .agg(
                pl.len().alias("n_obs"),
                pl.col("log_return").mean().alias("mean_daily_return"),
                pl.col("log_return").std().alias("daily_volatility"),
                pl.col("log_return").skew().alias("skewness"),
                pl.col("log_return").kurtosis().alias("excess_kurtosis"),
                pl.col("log_return").min().alias("min_daily_return"),
                pl.col("log_return").max().alias("max_daily_return"),
                pl.col("log_return").quantile(alpha).alias("var_95_daily"),
            )
            .with_columns(
                # Annualized expected return: 252 * mu_daily
                (pl.col("mean_daily_return") * ann_factor).alias("annualized_return"),
                # Annualized volatility: sqrt(252) * sigma_daily
                (pl.col("daily_volatility") * sqrt_ann).alias("annualized_volatility"),
            )
            .with_columns(
                # Annualized Sharpe ratio: (R_ann - R_f) / Sigma_ann
                ((pl.col("annualized_return") - rf) / pl.col("annualized_volatility")).alias(
                    "sharpe_ratio"
                )
            )
        )

        # 2. Maximum Drawdown calculation per symbol
        mdd_list: List[Dict[str, Any]] = []
        for symbol in self.config.symbols:
            sub = df_returns.filter(pl.col("symbol") == symbol).sort("date")
            prices = sub["adj_close"].to_numpy()
            if len(prices) > 0:
                running_max = np.maximum.accumulate(prices)
                drawdowns = (prices - running_max) / running_max
                max_dd = float(np.min(drawdowns))
            else:
                max_dd = 0.0
            mdd_list.append({"symbol": symbol, "max_drawdown": max_dd})

        mdd_df = pl.DataFrame(mdd_list)
        stats = stats.join(mdd_df, on="symbol", how="left")

        # Sort and arrange columns for publication standards
        ordered_cols = [
            "symbol",
            "n_obs",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "skewness",
            "excess_kurtosis",
            "max_drawdown",
            "var_95_daily",
            "min_daily_return",
            "max_daily_return",
        ]
        return stats.select(ordered_cols)

    def export_results(self, stats_df: pl.DataFrame) -> None:
        """Export curated statistics to CSV, Parquet, and Markdown formats."""
        # Export CSV
        stats_df.write_csv(self.config.stats_csv_path)
        logger.info("Exported summary statistics CSV: %s", self.config.stats_csv_path)

        # Export Parquet
        stats_df.write_parquet(self.config.stats_parquet_path)
        logger.info("Exported summary statistics Parquet: %s", self.config.stats_parquet_path)

        # Export formatted Markdown table
        md_content = self.generate_markdown_table(stats_df)
        with open(self.config.stats_markdown_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Exported summary statistics Markdown: %s", self.config.stats_markdown_path)

    @staticmethod
    def generate_markdown_table(stats_df: pl.DataFrame) -> str:
        """Generate publication-ready Markdown table adhering to Literate Programming."""
        header = (
            "| Asset | N | Ann. Return | Ann. Volatility | Sharpe ($R_f=4.25\\%$) | "
            "Skewness | Kurtosis | Max Drawdown | Daily VaR (95%) |\n"
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        rows: List[str] = []
        for row in stats_df.iter_rows(named=True):
            line = (
                f"| **{row['symbol']}** "
                f"| {row['n_obs']:,} "
                f"| {row['annualized_return']:.2%} "
                f"| {row['annualized_volatility']:.2%} "
                f"| {row['sharpe_ratio']:.2f} "
                f"| {row['skewness']:.2f} "
                f"| {row['excess_kurtosis']:.2f} "
                f"| {row['max_drawdown']:.2%} "
                f"| {row['var_95_daily']:.2%} |"
            )
            rows.append(line)
        return header + "\n".join(rows) + "\n"

    def run_pipeline(self, force_synthetic: bool = False) -> pl.DataFrame:
        """Execute end-to-end data curation pipeline."""
        logger.info("Starting Data Curator pipeline execution...")
        raw_data = self.fetch_data(force_synthetic=force_synthetic)
        returns_data = self.compute_returns(raw_data)
        stats = self.compute_descriptive_statistics(returns_data)
        self.export_results(stats)
        logger.info("Data Curator pipeline completed successfully.")
        return stats


def main() -> None:
    """CLI entrypoint for Data Curator."""
    parser = argparse.ArgumentParser(
        description="ML4T Data Curator: Market data acquisition and descriptive statistics with Polars."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic calibrated data generation.",
    )
    args = parser.parse_args()

    config = CuratorConfig.from_yaml(args.config)
    curator = MarketDataCurator(config)
    stats = curator.run_pipeline(force_synthetic=args.synthetic)

    print("\n" + "=" * 80)
    print("CURATED DESCRIPTIVE STATISTICS (POLARS AGGREGATION)")
    print("=" * 80)
    print(curator.generate_markdown_table(stats))


if __name__ == "__main__":
    main()
