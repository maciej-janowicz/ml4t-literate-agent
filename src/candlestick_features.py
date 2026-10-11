#!/usr/bin/env python3
"""ML4T Literate Agent - Continuous Candlestick Feature Engineering Module.

This module replaces heuristic, discrete candlestick pattern matching (e.g., Doji,
Hammer, Engulfing rules) with a continuous, scale-invariant geometric and microstructural
feature representation.

Implemented using Polars vectorized expressions (pl.Expr) for high-throughput
analytical execution on large-scale quantitative market data.

Author: Quantitative Research Team / agy-cli
Date: 2026-10-11
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import polars as pl

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CandlestickFeatures")


@dataclass(frozen=True)
class CandlestickFeatureConfig:
    """Configuration container for continuous candlestick feature extraction.

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
    volume_col : str
        Column name for trading volume.
    symbol_col : str
        Column name for asset ticker/symbol partition.
    date_col : str
        Column name for temporal ordering.
    epsilon : float
        Regularization constant to prevent zero-division on zero-spread candles.
    rolling_windows : Sequence[int]
        Lookback windows for baseline volume and volatility normalization.
    forward_return_horizons : Sequence[int]
        Forward periods h for target log returns calculation: ln(Close_{t+h} / Close_t).
    """

    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    symbol_col: str = "symbol"
    date_col: str = "date"
    epsilon: float = 1e-8
    rolling_windows: Sequence[int] = field(default_factory=lambda: (5, 20))
    forward_return_horizons: Sequence[int] = field(default_factory=lambda: (1, 3, 5, 10))


# ==============================================================================
# ATOMIC POLARS EXPRESSIONS (VECTORIZED GEOMETRIC & VOLUME FORMULATIONS)
# ==============================================================================


def expr_total_range(
    high_col: str = "high",
    low_col: str = "low",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute total candle range with numerical regularization:

    $$\widetilde{R}_t = High_t - Low_t + \epsilon$$
    """
    return (pl.col(high_col) - pl.col(low_col) + epsilon).alias("candle_range")


def expr_relative_body(
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute absolute relative body size:

    $$B_t = \frac{|Close_t - Open_t|}{High_t - Low_t + \epsilon} \in [0, 1]$$
    """
    rng = pl.col(high_col) - pl.col(low_col) + epsilon
    return ((pl.col(close_col) - pl.col(open_col)).abs() / rng).alias("body_rel_size")


def expr_signed_relative_body(
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute signed relative body size (directional conviction):

    $$B_t^{\text{signed}} = \frac{Close_t - Open_t}{High_t - Low_t + \epsilon} \in [-1, 1]$$
    """
    rng = pl.col(high_col) - pl.col(low_col) + epsilon
    return ((pl.col(close_col) - pl.col(open_col)) / rng).alias("body_signed_ratio")


def expr_upper_shadow_ratio(
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute upper shadow (wick) ratio to total range:

    $$U_t = \frac{High_t - \max(Open_t, Close_t)}{High_t - Low_t + \epsilon} \in [0, 1]$$
    """
    rng = pl.col(high_col) - pl.col(low_col) + epsilon
    upper_len = pl.col(high_col) - pl.max_horizontal(open_col, close_col)
    return (upper_len / rng).alias("upper_shadow_ratio")


def expr_lower_shadow_ratio(
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute lower shadow (tail) ratio to total range:

    $$L_t = \frac{\min(Open_t, Close_t) - Low_t}{High_t - Low_t + \epsilon} \in [0, 1]$$
    """
    rng = pl.col(high_col) - pl.col(low_col) + epsilon
    lower_len = pl.min_horizontal(open_col, close_col) - pl.col(low_col)
    return (lower_len / rng).alias("lower_shadow_ratio")


def expr_shadow_asymmetry(
    upper_shadow_col: str = "upper_shadow_ratio",
    lower_shadow_col: str = "lower_shadow_ratio",
) -> pl.Expr:
    r"""Compute shadow asymmetry (overhead supply rejection vs. demand support absorption):

    $$A_t^{\text{shadow}} = U_t - L_t \in [-1, 1]$$

    Positive: upper rejection dominance (bearish wick).
    Negative: lower rejection dominance (bullish tail).
    """
    return (pl.col(upper_shadow_col) - pl.col(lower_shadow_col)).alias("shadow_asymmetry")


def expr_shadow_relative_asymmetry(
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute normalized shadow asymmetry ratio:

    $$A_{t,\text{rel}}^{\text{shadow}} = \frac{U_t - L_t}{U_t + L_t + \epsilon} \in [-1, 1]$$
    """
    upper_len = pl.col(high_col) - pl.max_horizontal(open_col, close_col)
    lower_len = pl.min_horizontal(open_col, close_col) - pl.col(low_col)
    return ((upper_len - lower_len) / (upper_len + lower_len + epsilon)).alias(
        "shadow_rel_asymmetry"
    )


def expr_position_in_range(
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute close location value relative to the session's extremes:

    $$Pos_t^{\text{range}} = \frac{Close_t - Low_t}{High_t - Low_t + \epsilon} \in [0, 1]$$

    $Pos_t = 1$: close at the exact session high (maximum intraday dominance).
    $Pos_t = 0$: close at the exact session low.
    """
    rng = pl.col(high_col) - pl.col(low_col) + epsilon
    return ((pl.col(close_col) - pl.col(low_col)) / rng).alias("close_position_in_range")


def expr_open_position_in_range(
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute open location value relative to the session's extremes:

    $$Pos_t^{\text{open}} = \frac{Open_t - Low_t}{High_t - Low_t + \epsilon} \in [0, 1]$$
    """
    rng = pl.col(high_col) - pl.col(low_col) + epsilon
    return ((pl.col(open_col) - pl.col(low_col)) / rng).alias("open_position_in_range")


def expr_relative_range(
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute relative candle range (spread) normalized by open price:

    $$Range_t^{\text{rel}} = \frac{High_t - Low_t}{Open_t + \epsilon}$$
    """
    return ((pl.col(high_col) - pl.col(low_col)) / (pl.col(open_col) + epsilon)).alias(
        "relative_range"
    )


def expr_normalized_range(
    high_col: str = "high",
    low_col: str = "low",
    window: int = 20,
    partition_by: Optional[str] = "symbol",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute range normalized by rolling average range (volatility expansion factor):

    $$Range_{t,k}^{\text{norm}} = \frac{High_t - Low_t}{\text{SMA}_k(High - Low) + \epsilon}$$
    """
    raw_range = pl.col(high_col) - pl.col(low_col)
    rolling_range = raw_range.rolling_mean(window_size=window)
    if partition_by:
        rolling_range = rolling_range.over(partition_by)
    return (raw_range / (rolling_range + epsilon)).alias(f"normalized_range_w{window}")


def expr_relative_volume(
    volume_col: str = "volume",
    window: int = 20,
    partition_by: Optional[str] = "symbol",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute relative volume (RVOL) against rolling average volume:

    $$V_{t,k}^{\text{rel}} = \frac{Volume_t}{\text{SMA}_k(Volume) + \epsilon}$$
    """
    rolling_vol = pl.col(volume_col).rolling_mean(window_size=window)
    if partition_by:
        rolling_vol = rolling_vol.over(partition_by)
    return (pl.col(volume_col) / (rolling_vol + epsilon)).alias(f"relative_volume_w{window}")


def expr_volume_intensity(
    volume_col: str = "volume",
    high_col: str = "high",
    low_col: str = "low",
    epsilon: float = 1e-8,
) -> pl.Expr:
    r"""Compute volume intensity (volume per unit of price movement):

    $$V_t^{\text{intensity}} = \ln\left(1 + \frac{Volume_t}{High_t - Low_t + \epsilon}\right)$$

    High intensity with narrow range indicates liquidity clustering / institutional churning.
    """
    intensity = pl.col(volume_col) / (pl.col(high_col) - pl.col(low_col) + epsilon)
    return (intensity.log1p()).alias("volume_intensity_log")


def expr_volume_weighted_body(
    body_signed_col: str = "body_signed_ratio",
    relative_volume_col: str = "relative_volume_w20",
) -> pl.Expr:
    r"""Compute volume-weighted directional body momentum (effort vs. result):

    $$V_t^{\text{body}} = B_t^{\text{signed}} \times V_{t,k}^{\text{rel}}$$
    """
    return (pl.col(body_signed_col) * pl.col(relative_volume_col)).alias("vol_weighted_body")


def expr_volume_weighted_shadows(
    upper_shadow_col: str = "upper_shadow_ratio",
    lower_shadow_col: str = "lower_shadow_ratio",
    relative_volume_col: str = "relative_volume_w20",
) -> Sequence[pl.Expr]:
    r"""Compute volume-weighted shadow rejections:

    Overhead supply pressure: $V_t^{\text{upper}} = U_t \times V_{t,k}^{\text{rel}}$
    Support absorption: $V_t^{\text{lower}} = L_t \times V_{t,k}^{\text{rel}}$
    """
    return [
        (pl.col(upper_shadow_col) * pl.col(relative_volume_col)).alias("vol_weighted_upper_shadow"),
        (pl.col(lower_shadow_col) * pl.col(relative_volume_col)).alias("vol_weighted_lower_shadow"),
    ]


def expr_forward_returns(
    close_col: str = "close",
    horizons: Sequence[int] = (1, 3, 5, 10),
    partition_by: Optional[str] = "symbol",
) -> List[pl.Expr]:
    r"""Compute target forward log returns for alpha evaluation:

    $$r_{t \to t+h} = \ln\left(\frac{Close_{t+h}}{Close_t}\right)$$
    """
    exprs: List[pl.Expr] = []
    for h in horizons:
        shifted_close = pl.col(close_col).shift(-h)
        if partition_by:
            shifted_close = shifted_close.over(partition_by)
        ret_expr = (shifted_close / pl.col(close_col)).log().alias(f"fwd_ret_{h}d")
        exprs.append(ret_expr)
    return exprs


# ==============================================================================
# HIGH-PERFORMANCE POLARS FEATURE PIPELINE
# ==============================================================================


def compute_candlestick_features(
    data: Union[pl.DataFrame, pl.LazyFrame],
    config: Optional[CandlestickFeatureConfig] = None,
    include_forward_returns: bool = True,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """Transform raw OHLCV market quotes into continuous geometric and microstructural features.

    Fully vectorized using Polars expressions, guaranteeing zero Python row-level
    iteration overhead and optimal thread pool utilization across symbol groups.

    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Input dataset containing OHLCV columns.
    config : CandlestickFeatureConfig, optional
        Configuration for feature extraction and column mappings.
    include_forward_returns : bool, default True
        Whether to append forward log return columns for target modeling/correlation.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Transformed dataset enriched with continuous candlestick features.
    """
    cfg = config or CandlestickFeatureConfig()
    is_lazy = isinstance(data, pl.LazyFrame)
    lf = data if is_lazy else data.lazy()

    # Step 1: Ensure sorted order by symbol and date
    sort_cols = [c for c in [cfg.symbol_col, cfg.date_col] if c in lf.collect_schema().names()]
    if sort_cols:
        lf = lf.sort(sort_cols)

    # Step 2: Base Geometric Candle Features
    base_geometric_exprs = [
        expr_total_range(cfg.high_col, cfg.low_col, cfg.epsilon),
        expr_relative_body(cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.epsilon),
        expr_signed_relative_body(
            cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.epsilon
        ),
        expr_upper_shadow_ratio(
            cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.epsilon
        ),
        expr_lower_shadow_ratio(
            cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.epsilon
        ),
        expr_shadow_relative_asymmetry(
            cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.epsilon
        ),
        expr_position_in_range(cfg.high_col, cfg.low_col, cfg.close_col, cfg.epsilon),
        expr_open_position_in_range(cfg.open_col, cfg.high_col, cfg.low_col, cfg.epsilon),
        expr_relative_range(cfg.open_col, cfg.high_col, cfg.low_col, cfg.epsilon),
        expr_volume_intensity(cfg.volume_col, cfg.high_col, cfg.low_col, cfg.epsilon),
    ]

    # Include partition-based rolling expressions
    has_symbol = cfg.symbol_col in lf.collect_schema().names()
    partition = cfg.symbol_col if has_symbol else None

    rolling_exprs: List[pl.Expr] = []
    for w in cfg.rolling_windows:
        rolling_exprs.append(
            expr_normalized_range(cfg.high_col, cfg.low_col, window=w, partition_by=partition, epsilon=cfg.epsilon)
        )
        rolling_exprs.append(
            expr_relative_volume(cfg.volume_col, window=w, partition_by=partition, epsilon=cfg.epsilon)
        )

    # Apply base and rolling expressions
    lf = lf.with_columns(base_geometric_exprs + rolling_exprs)

    # Step 3: Composite Interactive Features (Shadow Asymmetry & Volume Interactions)
    primary_window = cfg.rolling_windows[-1] if cfg.rolling_windows else 20
    rel_vol_col = f"relative_volume_w{primary_window}"

    composite_exprs = [
        expr_shadow_asymmetry("upper_shadow_ratio", "lower_shadow_ratio"),
        expr_volume_weighted_body("body_signed_ratio", rel_vol_col),
        *expr_volume_weighted_shadows("upper_shadow_ratio", "lower_shadow_ratio", rel_vol_col),
    ]
    lf = lf.with_columns(composite_exprs)

    # Step 4: Optional Forward Return Targets
    if include_forward_returns:
        fwd_exprs = expr_forward_returns(
            close_col=cfg.close_col,
            horizons=cfg.forward_return_horizons,
            partition_by=partition,
        )
        lf = lf.with_columns(fwd_exprs)

    return lf if is_lazy else lf.collect()


# ==============================================================================
# EMPIRICAL VALIDATION & CORRELATION / INFORMATION COEFFICIENT UTILITIES
# ==============================================================================


CANDLESTICK_FEATURE_NAMES = [
    "body_rel_size",
    "body_signed_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "shadow_asymmetry",
    "shadow_rel_asymmetry",
    "close_position_in_range",
    "open_position_in_range",
    "relative_range",
    "volume_intensity_log",
    "vol_weighted_body",
    "vol_weighted_upper_shadow",
    "vol_weighted_lower_shadow",
]


def compute_feature_correlations(
    df: pl.DataFrame,
    features: Optional[Sequence[str]] = None,
    targets: Optional[Sequence[str]] = None,
    method: str = "pearson",
) -> pl.DataFrame:
    """Compute cross-sectional / time-series correlations between candlestick features and forward returns.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with computed features and forward return columns.
    features : Sequence[str], optional
        List of feature column names. Defaults to standard geometric candlestick features.
    targets : Sequence[str], optional
        List of target forward return column names (e.g. ['fwd_ret_1d', 'fwd_ret_3d', 'fwd_ret_5d']).
    method : str, default 'pearson'
        Correlation method ('pearson' or 'spearman').

    Returns
    -------
    pl.DataFrame
        Polars DataFrame containing feature correlation statistics.
    """
    feature_cols = [f for f in (features or CANDLESTICK_FEATURE_NAMES) if f in df.columns]
    target_cols = [t for t in (targets or ["fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d"]) if t in df.columns]

    if not feature_cols or not target_cols:
        raise ValueError("Missing feature or target columns in input dataframe.")

    # Filter out rows with nulls in features or targets
    valid_df = df.select(feature_cols + target_cols).drop_nulls()

    results: List[Dict[str, Union[str, float]]] = []

    for feat in feature_cols:
        row_dict: Dict[str, Union[str, float]] = {"feature": feat}
        x = valid_df[feat].to_numpy()

        for tgt in target_cols:
            y = valid_df[tgt].to_numpy()
            if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
                corr = 0.0
            elif method == "spearman":
                # Rank correlation (Spearman Information Coefficient)
                rx = np.argsort(np.argsort(x))
                ry = np.argsort(np.argsort(y))
                corr = float(np.corrcoef(rx, ry)[0, 1])
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
            row_dict[f"IC_{tgt}"] = round(corr, 4)

        results.append(row_dict)

    return pl.DataFrame(results)


def compute_feature_summary_statistics(
    df: pl.DataFrame,
    features: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """Compute distributional descriptive statistics for continuous candlestick features."""
    feature_cols = [f for f in (features or CANDLESTICK_FEATURE_NAMES) if f in df.columns]

    agg_exprs: List[pl.Expr] = []
    for f in feature_cols:
        agg_exprs.extend([
            pl.col(f).mean().alias(f"{f}__mean"),
            pl.col(f).std().alias(f"{f}__std"),
            pl.col(f).min().alias(f"{f}__min"),
            pl.col(f).quantile(0.25).alias(f"{f}__q25"),
            pl.col(f).median().alias(f"{f}__median"),
            pl.col(f).quantile(0.75).alias(f"{f}__q75"),
            pl.col(f).max().alias(f"{f}__max"),
            pl.col(f).skew().alias(f"{f}__skew"),
            pl.col(f).kurtosis().alias(f"{f}__kurt"),
        ])

    aggregated = df.select(agg_exprs).to_dicts()[0]

    rows: List[Dict[str, Union[str, float]]] = []
    for f in feature_cols:
        rows.append({
            "feature": f,
            "mean": round(float(aggregated[f"{f}__mean"] or 0.0), 4),
            "std": round(float(aggregated[f"{f}__std"] or 0.0), 4),
            "min": round(float(aggregated[f"{f}__min"] or 0.0), 4),
            "q25": round(float(aggregated[f"{f}__q25"] or 0.0), 4),
            "median": round(float(aggregated[f"{f}__median"] or 0.0), 4),
            "q75": round(float(aggregated[f"{f}__q75"] or 0.0), 4),
            "max": round(float(aggregated[f"{f}__max"] or 0.0), 4),
            "skewness": round(float(aggregated[f"{f}__skew"] or 0.0), 4),
            "kurtosis": round(float(aggregated[f"{f}__kurt"] or 0.0), 4),
        })

    return pl.DataFrame(rows)


def generate_synthetic_ohlcv(
    symbols: Sequence[str] = ("SPY", "QQQ", "AAPL", "NVDA"),
    n_days: int = 1258,
    start_date: str = "2020-01-01",
    seed: int = 42,
) -> pl.DataFrame:
    """Generate deterministic, statistically calibrated synthetic OHLCV data for offline verification."""
    np.random.seed(seed)
    start = dt.date.fromisoformat(start_date)

    # Generate business days
    dates: List[dt.date] = []
    cur = start
    while len(dates) < n_days:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += dt.timedelta(days=1)

    dfs: List[pl.DataFrame] = []
    for sym in symbols:
        drift = 0.0004
        vol = 0.015
        log_rets = np.random.normal(drift, vol, n_days)
        prices = 100.0 * np.exp(np.cumsum(log_rets))

        # Realistic intraday noise
        intraday_vol = np.random.gamma(shape=2.0, scale=0.008, size=n_days)
        opens = prices * (1.0 + np.random.normal(0, 0.003, n_days))
        closes = prices
        highs = np.maximum(opens, closes) + prices * np.abs(np.random.normal(0, intraday_vol, n_days))
        lows = np.minimum(opens, closes) - prices * np.abs(np.random.normal(0, intraday_vol, n_days))
        volumes = np.random.lognormal(mean=16.0, sigma=0.5, size=n_days).astype(np.int64)

        df_sym = pl.DataFrame({
            "date": dates,
            "symbol": [sym] * n_days,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "adj_close": closes,
            "volume": volumes,
        })
        dfs.append(df_sym)

    return pl.concat(dfs).sort(["symbol", "date"])


# ==============================================================================
# MAIN ENTRYPOINT / SELF-VERIFICATION
# ==============================================================================


def main() -> None:
    """CLI / Execution entrypoint for continuous candlestick feature generation."""
    parser = argparse.ArgumentParser(
        description="Compute continuous geometric and microstructural candlestick features using Polars."
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
        default=Path("outputs/curated_candlestick_features.parquet"),
        help="Path to export features parquet file.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force generation on synthetic calibrated OHLCV data.",
    )
    args = parser.parse_args()

    logger.info("Initializing Continuous Candlestick Feature Engineering Pipeline...")

    if not args.synthetic and args.input.exists():
        logger.info("Loading input market data from %s", args.input)
        raw_df = pl.read_parquet(args.input)
    else:
        logger.info("Input file not found or --synthetic specified. Generating calibrated synthetic OHLCV universe...")
        raw_df = generate_synthetic_ohlcv()

    cfg = CandlestickFeatureConfig()
    logger.info("Computing vectorized candlestick features on %d rows...", raw_df.height)
    features_df = compute_candlestick_features(raw_df, config=cfg)

    # Export enriched dataset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features_df.write_parquet(args.output)
    logger.info("Successfully exported candlestick features to %s (%d rows)", args.output, features_df.height)

    # Compute correlation analysis
    corr_df = compute_feature_correlations(features_df, method="spearman")
    corr_csv_path = Path("outputs/candlestick_correlations.csv")
    corr_df.write_csv(corr_csv_path)
    logger.info("Exported feature correlation table to %s", corr_csv_path)

    # Compute descriptive statistics
    stats_df = compute_feature_summary_statistics(features_df)
    stats_csv_path = Path("outputs/candlestick_feature_stats.csv")
    stats_df.write_csv(stats_csv_path)
    logger.info("Exported feature summary statistics to %s", stats_csv_path)

    print("\n" + "=" * 90)
    print("CONTINUOUS CANDLESTICK FEATURES: INFORMATION COEFFICIENT (SPEARMAN CORRELATION)")
    print("=" * 90)
    print(corr_df)
    print("\n" + "=" * 90)
    print("DISTRIBUTIONAL SUMMARY STATISTICS")
    print("=" * 90)
    print(stats_df)


if __name__ == "__main__":
    main()
