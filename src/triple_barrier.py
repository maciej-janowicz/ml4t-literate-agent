#!/usr/bin/env python3
"""ML4T Literate Agent - Triple-Barrier Method Labeling Module.

Implements Marcos López de Prado's Triple-Barrier Method for continuous financial
machine learning label generation with dynamic volatility scaling and strict
temporal / cross-asset leakage prevention.

Reference:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    John Wiley & Sons, Chapter 3: "Labeling".

Author: Quantitative Research Team / agy-cli
Date: 2026-10-11
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TripleBarrier")


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Configuration container for Triple-Barrier Method labeling.

    Parameters
    ----------
    open_col : str
        Column name for open price.
    high_col : str
        Column name for high price.
    low_col : str
        Column name for low price.
    close_col : str
        Column name for close price.
    symbol_col : str
        Column name for asset ticker/symbol partition.
    date_col : str
        Column name for temporal ordering.
    volatility_window : int
        Lookback window k for backward-looking rolling return volatility estimate (default 20).
    pt_multiplier : float
        Upper barrier (Take Profit) multiplier k_pt scaled by local volatility (default 2.0).
    sl_multiplier : float
        Lower barrier (Stop Loss) multiplier k_sl scaled by local volatility (default 2.0).
    max_holding_bars : int
        Vertical barrier / maximum holding horizon h in bars (default 10).
    min_volatility : float
        Floor threshold for volatility to prevent degenerate barriers in flat regimes (default 1e-4).
    conflict_resolution : str
        Policy when both upper and lower barriers are crossed in the same bar:
        'conservative' (stop loss triggered first) or 'distance' (closer to bar open).
    use_sign_on_expiration : bool
        If True, vertical barrier expiration is labeled sign(return); if False, labeled 0.
    """

    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    symbol_col: str = "symbol"
    date_col: str = "date"
    volatility_window: int = 20
    pt_multiplier: float = 2.0
    sl_multiplier: float = 2.0
    max_holding_bars: int = 10
    min_volatility: float = 1e-4
    conflict_resolution: str = "conservative"
    use_sign_on_expiration: bool = False


# ==============================================================================
# VOLATILITY ESTIMATION EXPRESSIONS (STRICTLY CAUSAL / BACKWARD-LOOKING)
# ==============================================================================


def expr_rolling_volatility(
    close_col: str = "close",
    window: int = 20,
    partition_by: Optional[str] = "symbol",
    min_volatility: float = 1e-4,
) -> pl.Expr:
    r"""Compute causal backward-looking daily log-return volatility:

    $$\sigma_t = \sqrt{\frac{1}{k-1} \sum_{i=0}^{k-1} (r_{t-i} - \bar{r}_t)^2}$$

    where $r_t = \ln(Close_t / Close_{t-1})$. Strictly avoids lookahead by utilizing
    only historical observations up to bar $t$.
    """
    # Daily log returns: ln(Close_t / Close_{t-1})
    log_ret = (pl.col(close_col) / pl.col(close_col).shift(1)).log()
    if partition_by:
        log_ret = log_ret.over(partition_by)

    rolling_std = log_ret.rolling_std(window_size=window)
    if partition_by:
        rolling_std = rolling_std.over(partition_by)

    # Floor with min_volatility to avoid zero-barrier degeneracies
    return (
        pl.when(rolling_std.is_null() | (rolling_std < min_volatility))
        .then(pl.lit(min_volatility))
        .otherwise(rolling_std)
        .alias("daily_volatility")
    )


# ==============================================================================
# CORE TRIPLE-BARRIER SCANNING ENGINE
# ==============================================================================


def _scan_barriers_1d(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    opens: np.ndarray,
    entry_prices: np.ndarray,
    volatilities: np.ndarray,
    pt_mult: float,
    sl_mult: float,
    max_h: int,
    conflict_resolution: str = "conservative",
    use_sign_expiration: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute high-speed forward barrier scan over a single contiguous asset time series.

    Guarantees strict temporal causality: for bar t, barriers are monitored only
    across subsequent bars s in [t+1, min(t+max_h, n-1)].

    Returns
    -------
    labels : np.ndarray (int8)
        Label: +1 (Take Profit), -1 (Stop Loss), 0 (Vertical expiration or neutral).
    touch_bars : np.ndarray (int16)
        Holding time in bars until barrier touch (1 to max_h).
    realized_returns : np.ndarray (float64)
        Realized log return at barrier touch: ln(P_tau / P_t).
    barrier_types : np.ndarray (object)
        Touched barrier descriptor: 'upper', 'lower', 'vertical', or 'unresolved'.
    upper_barriers : np.ndarray (float64)
        Price level of upper barrier U_t.
    lower_barriers : np.ndarray (float64)
        Price level of lower barrier L_t.
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    touch_bars = np.zeros(n, dtype=np.int16)
    realized_returns = np.zeros(n, dtype=np.float64)
    barrier_types = np.empty(n, dtype=object)
    upper_barriers = np.zeros(n, dtype=np.float64)
    lower_barriers = np.zeros(n, dtype=np.float64)

    for t in range(n):
        entry_p = entry_prices[t]
        vol = volatilities[t]

        if np.isnan(entry_p) or np.isnan(vol) or entry_p <= 0.0 or vol <= 0.0:
            barrier_types[t] = "unresolved"
            continue

        # Dynamic volatility-scaled barrier thresholds
        upper_b = entry_p * (1.0 + pt_mult * vol)
        lower_b = entry_p * (1.0 - sl_mult * vol)
        upper_barriers[t] = upper_b
        lower_barriers[t] = lower_b

        end_idx = min(t + max_h, n - 1)
        if t >= n - 1:
            # End of series boundary
            barrier_types[t] = "unresolved"
            continue

        hit_found = False
        # Forward scan starting strictly at t + 1 (never inside entry bar t)
        for s in range(t + 1, end_idx + 1):
            h_s = highs[s]
            l_s = lows[s]
            o_s = opens[s]

            hit_upper = h_s >= upper_b
            hit_lower = l_s <= lower_b

            if hit_upper and hit_lower:
                # Conflicting simultaneous cross within bar s
                if conflict_resolution == "conservative":
                    # Conservative risk-first assumption: stop-loss executed
                    labels[t] = -1
                    touch_bars[t] = s - t
                    realized_returns[t] = np.log(lower_b / entry_p)
                    barrier_types[t] = "lower"
                else:
                    # Resolve based on proximity to open price of bar s
                    if abs(o_s - upper_b) < abs(o_s - lower_b):
                        labels[t] = 1
                        touch_bars[t] = s - t
                        realized_returns[t] = np.log(upper_b / entry_p)
                        barrier_types[t] = "upper"
                    else:
                        labels[t] = -1
                        touch_bars[t] = s - t
                        realized_returns[t] = np.log(lower_b / entry_p)
                        barrier_types[t] = "lower"
                hit_found = True
                break

            elif hit_upper:
                labels[t] = 1
                touch_bars[t] = s - t
                realized_returns[t] = np.log(upper_b / entry_p)
                barrier_types[t] = "upper"
                hit_found = True
                break

            elif hit_lower:
                labels[t] = -1
                touch_bars[t] = s - t
                realized_returns[t] = np.log(lower_b / entry_p)
                barrier_types[t] = "lower"
                hit_found = True
                break

        if not hit_found:
            # Vertical barrier expired at end_idx
            holding = end_idx - t
            touch_bars[t] = holding
            exit_price = closes[end_idx]
            ret_at_expiration = np.log(exit_price / entry_p)
            realized_returns[t] = ret_at_expiration
            barrier_types[t] = "vertical"

            if use_sign_expiration:
                if ret_at_expiration > 0:
                    labels[t] = 1
                elif ret_at_expiration < 0:
                    labels[t] = -1
                else:
                    labels[t] = 0
            else:
                labels[t] = 0

    return labels, touch_bars, realized_returns, barrier_types, upper_barriers, lower_barriers


# ==============================================================================
# PIPELINE ORCHESTRATION & POLARS INTEGRATION
# ==============================================================================


def compute_triple_barrier_labels(
    data: Union[pl.DataFrame, pl.LazyFrame],
    config: Optional[TripleBarrierConfig] = None,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """Apply the Triple-Barrier Method across an entire multi-asset universe.

    Guarantees:
    1. Dynamic volatility scaling per observation t based strictly on historical returns.
    2. Zero lookahead bias: barriers evaluated strictly across bars s > t.
    3. Multi-asset partitioning: symbol groups are segmented and computed independently,
       completely eliminating cross-ticker leakage.

    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Input market dataset containing OHLCV and symbol/date columns.
    config : TripleBarrierConfig, optional
        Configuration for barrier parameters and column mappings.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Enriched dataset containing triple-barrier labels, touch horizons, realized returns,
        and barrier thresholds.
    """
    cfg = config or TripleBarrierConfig()
    is_lazy = isinstance(data, pl.LazyFrame)
    df_collected = data.collect() if is_lazy else data

    # Step 1: Ensure sorting by symbol and date
    sort_cols = [c for c in [cfg.symbol_col, cfg.date_col] if c in df_collected.columns]
    if sort_cols:
        df_sorted = df_collected.sort(sort_cols)
    else:
        df_sorted = df_collected

    # Step 2: Compute backward-looking volatility via Polars
    has_symbol = cfg.symbol_col in df_sorted.columns
    partition = cfg.symbol_col if has_symbol else None

    vol_expr = expr_rolling_volatility(
        close_col=cfg.close_col,
        window=cfg.volatility_window,
        partition_by=partition,
        min_volatility=cfg.min_volatility,
    )
    df_with_vol = df_sorted.with_columns(vol_expr)

    # Step 3: Partitioned barrier scanning per asset symbol
    if has_symbol:
        symbols = df_with_vol[cfg.symbol_col].unique(maintain_order=True).to_list()
        partitioned_dfs: List[pl.DataFrame] = []

        for sym in symbols:
            sub = df_with_vol.filter(pl.col(cfg.symbol_col) == sym)
            sub_res = _process_single_asset_barriers(sub, cfg)
            partitioned_dfs.append(sub_res)

        result_df = pl.concat(partitioned_dfs).sort(sort_cols)
    else:
        result_df = _process_single_asset_barriers(df_with_vol, cfg)

    return result_df.lazy() if is_lazy else result_df


def _process_single_asset_barriers(
    sub_df: pl.DataFrame,
    cfg: TripleBarrierConfig,
) -> pl.DataFrame:
    """Helper to execute 1D barrier scan on a single asset partition."""
    highs = sub_df[cfg.high_col].to_numpy()
    lows = sub_df[cfg.low_col].to_numpy()
    closes = sub_df[cfg.close_col].to_numpy()
    opens = sub_df[cfg.open_col].to_numpy()
    entry_prices = closes  # Session close is the decision settlement
    volatilities = sub_df["daily_volatility"].to_numpy()

    labels, touch_bars, realized_rets, barrier_types, upper_b, lower_b = _scan_barriers_1d(
        highs=highs,
        lows=lows,
        closes=closes,
        opens=opens,
        entry_prices=entry_prices,
        volatilities=volatilities,
        pt_mult=cfg.pt_multiplier,
        sl_mult=cfg.sl_multiplier,
        max_h=cfg.max_holding_bars,
        conflict_resolution=cfg.conflict_resolution,
        use_sign_expiration=cfg.use_sign_on_expiration,
    )

    enriched = sub_df.with_columns([
        pl.Series("tb_label", labels, dtype=pl.Int8),
        pl.Series("tb_touch_bars", touch_bars, dtype=pl.Int16),
        pl.Series("tb_realized_return", realized_rets, dtype=pl.Float64),
        pl.Series("tb_barrier_type", barrier_types, dtype=pl.String),
        pl.Series("tb_upper_barrier", upper_b, dtype=pl.Float64),
        pl.Series("tb_lower_barrier", lower_b, dtype=pl.Float64),
    ])
    return enriched


# ==============================================================================
# AUDITABILITY & LABEL DIAGNOSTIC METRICS
# ==============================================================================


def compute_label_distribution(
    df: pl.DataFrame,
    group_by_symbol: bool = True,
) -> pl.DataFrame:
    """Compute empirical distribution and risk-reward diagnostic statistics for triple-barrier labels."""
    group_cols = ["symbol"] if group_by_symbol and "symbol" in df.columns else []

    base = df.filter(pl.col("tb_barrier_type") != "unresolved")

    stats = (
        base.group_by(group_cols + ["tb_label"], maintain_order=True)
        .agg(
            pl.len().alias("count"),
            pl.col("tb_touch_bars").mean().round(2).alias("avg_holding_bars"),
            pl.col("tb_realized_return").mean().round(4).alias("avg_return"),
            pl.col("tb_realized_return").std().round(4).alias("std_return"),
        )
        .sort(group_cols + ["tb_label"])
    )

    # Compute percentages within group
    if group_cols:
        totals = base.group_by(group_cols).agg(pl.len().alias("total_count"))
        stats = stats.join(totals, on=group_cols).with_columns(
            (pl.col("count") / pl.col("total_count") * 100).round(2).alias("pct_share")
        ).drop("total_count")
    else:
        total = base.height
        stats = stats.with_columns(
            (pl.col("count") / total * 100).round(2).alias("pct_share")
        )

    return stats


def compute_summary_table_markdown(df_dist: pl.DataFrame) -> str:
    """Format label distribution DataFrame as a publication-ready Markdown table."""
    has_sym = "symbol" in df_dist.columns

    if has_sym:
        header = (
            "| Asset | Label Class | Interpretation | Count | Share (%) | Avg Bars to Touch | Avg Realized Return | Std Dev |\n"
            "|:---|---:|:---|---:|---:|---:|---:|---:|\n"
        )
    else:
        header = (
            "| Label Class | Interpretation | Count | Share (%) | Avg Bars to Touch | Avg Realized Return | Std Dev |\n"
            "|---:|:---|---:|---:|---:|---:|---:|\n"
        )

    label_desc = {
        1: "Upper Barrier (Take Profit)",
        -1: "Lower Barrier (Stop Loss)",
        0: "Vertical Barrier (Time Expired)",
    }

    lines: List[str] = []
    for row in df_dist.iter_rows(named=True):
        lbl = row["tb_label"]
        desc = label_desc.get(lbl, "Unknown")
        cnt = row["count"]
        pct = row["pct_share"]
        bars = row["avg_holding_bars"]
        ret = row["avg_return"]
        std = row["std_return"]

        if has_sym:
            sym = row["symbol"]
            line = f"| **{sym}** | `{lbl:+d}` | {desc} | {cnt:,} | {pct:.1f}% | {bars:.2f} | {ret:+.2%} | {std:.2%} |"
        else:
            line = f"| `{lbl:+d}` | {desc} | {cnt:,} | {pct:.1f}% | {bars:.2f} | {ret:+.2%} | {std:.2%} |"
        lines.append(line)

    return header + "\n".join(lines) + "\n"


# ==============================================================================
# CLI EXECUTION & OFFLINE VERIFICATION
# ==============================================================================


def main() -> None:
    """CLI execution entrypoint for Triple-Barrier Method labeling."""
    parser = argparse.ArgumentParser(
        description="Compute Triple-Barrier Method labels with dynamic volatility scaling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/curated_market_data.parquet"),
        help="Path to curated OHLCV parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/curated_triple_barrier_labels.parquet"),
        help="Path to export triple-barrier labels parquet file.",
    )
    parser.add_argument(
        "--pt",
        type=float,
        default=2.0,
        help="Take profit multiplier k_pt.",
    )
    parser.add_argument(
        "--sl",
        type=float,
        default=2.0,
        help="Stop loss multiplier k_sl.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Maximum vertical barrier holding horizon in bars.",
    )
    args = parser.parse_args()

    logger.info("Initializing Triple-Barrier Method Labeling Pipeline...")

    if args.input.exists():
        logger.info("Loading input market data from %s", args.input)
        raw_df = pl.read_parquet(args.input)
    else:
        logger.info("Input file not found. Generating calibrated synthetic OHLCV benchmark...")
        # Import synthetic generator from candlestick_features
        from src.candlestick_features import generate_synthetic_ohlcv
        raw_df = generate_synthetic_ohlcv(
            symbols=["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TLT", "GLD"],
            n_days=1258,
            seed=42,
        )

    config = TripleBarrierConfig(
        pt_multiplier=args.pt,
        sl_multiplier=args.sl,
        max_holding_bars=args.horizon,
    )

    logger.info(
        "Executing Triple-Barrier scan (k_pt=%.1f, k_sl=%.1f, horizon=%d bars)...",
        args.pt,
        args.sl,
        args.horizon,
    )
    labeled_df = compute_triple_barrier_labels(raw_df, config=config)

    # Export labeled dataset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.write_parquet(args.output)
    logger.info("Exported labeled dataset to %s (%d rows)", args.output, labeled_df.height)

    # Compute diagnostics
    dist_df = compute_label_distribution(labeled_df, group_by_symbol=True)
    dist_csv = Path("outputs/triple_barrier_distribution.csv")
    dist_df.write_csv(dist_csv)
    logger.info("Exported label distribution statistics to %s", dist_csv)

    print("\n" + "=" * 90)
    print("TRIPLE-BARRIER LABEL DIAGNOSTICS")
    print("=" * 90)
    print(compute_summary_table_markdown(dist_df))


if __name__ == "__main__":
    main()
