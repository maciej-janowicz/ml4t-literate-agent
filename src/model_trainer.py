#!/usr/bin/env python3
"""ML4T Literate Agent - Predictive Modeling & Purged Cross-Validation Engine.

This module implements a rigorous machine learning training and evaluation pipeline
for quantitative finance, incorporating:
1. Purged K-Fold Cross-Validation (eliminating label overlap leakage).
2. Post-test Embargo mechanism (mitigating short-term autoregressive correlation).
3. LightGBM multi-class gradient boosted decision trees for Triple-Barrier target classification.
4. Out-of-sample evaluation metrics (Precision, Recall, F1-Score, Multi-Class Log Loss, Balanced Accuracy).
5. Cross-fold Feature Importance analysis (split count and gain attribution).

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    John Wiley & Sons, Chapter 7: "Cross-Validation in Finance".
    Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
    Advances in Neural Information Processing Systems (NeurIPS 30).

Author: Quantitative Research Team / agy-cli
Date: 2026-10-17
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

# Optional LightGBM import with graceful fallback
try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ModelTrainer")


# ==============================================================================
# CONFIGURATION CONTAINER
# ==============================================================================


@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration container for predictive modeling and Purged K-Fold validation.

    Parameters
    ----------
    n_splits : int
        Number of cross-validation folds K (default 5).
    pct_embargo : float
        Percentage of sample size applied as post-test embargo buffer (default 0.01 = 1%).
    embargo_bars : Optional[int]
        Fixed number of bars/days for post-test embargo buffer. If specified, overrides pct_embargo.
    target_col : str
        Target column name representing Triple-Barrier labels {-1, 0, +1}.
    date_col : str
        Column name for temporal ordering and fold partitioning.
    symbol_col : str
        Column name for multi-asset segmentation.
    touch_bars_col : str
        Column name indicating number of bars until barrier touch (event horizon).
    barrier_type_col : str
        Column name specifying touched barrier ('upper', 'lower', 'vertical', 'unresolved').
    feature_cols : Optional[Sequence[str]]
        Explicit list of feature columns. If None, default candlestick features are selected.
    n_estimators : int
        Number of gradient boosted trees (default 150).
    learning_rate : float
        Boosting learning rate shrinkage (default 0.03).
    max_depth : int
        Maximum tree depth constraint to prevent memorizing financial noise (default 4).
    num_leaves : int
        Maximum number of leaves per tree (default 15).
    min_child_samples : int
        Minimum number of data points required in a leaf (default 40).
    subsample : float
        Subsample ratio of training instances for bagging (default 0.8).
    colsample_bytree : float
        Subsample ratio of features per tree split (default 0.8).
    reg_alpha : float
        L1 regularization penalty on weights (default 0.1).
    reg_lambda : float
        L2 regularization penalty on weights (default 1.0).
    class_weight : str
        Class weighting strategy: 'balanced' or None (default 'balanced').
    random_state : int
        Random seed for reproducible partition and model initializations (default 42).
    """

    n_splits: int = 5
    pct_embargo: float = 0.01
    embargo_bars: Optional[int] = None
    target_col: str = "tb_label"
    date_col: str = "date"
    symbol_col: str = "symbol"
    touch_bars_col: str = "tb_touch_bars"
    barrier_type_col: str = "tb_barrier_type"
    feature_cols: Optional[Sequence[str]] = None
    n_estimators: int = 150
    learning_rate: float = 0.03
    max_depth: int = 4
    num_leaves: int = 15
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    class_weight: str = "balanced"
    random_state: int = 42


# ==============================================================================
# DEFAULT FEATURE TAXONOMY
# ==============================================================================

DEFAULT_CONTINUOUS_FEATURES: List[str] = [
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
    "normalized_range_w5",
    "relative_volume_w5",
    "normalized_range_w20",
    "relative_volume_w20",
    "vol_weighted_body",
    "vol_weighted_upper_shadow",
    "vol_weighted_lower_shadow",
    "daily_volatility",
]

# Mapping between Triple-Barrier targets {-1, 0, 1} and multi-class indices {0, 1, 2}
LABEL_ENCODING: Dict[int, int] = {-1: 0, 0: 1, 1: 2}
LABEL_DECODING: Dict[int, int] = {0: -1, 1: 0, 2: 1}
CLASS_NAMES: List[str] = ["Stop Loss (-1)", "Vertical Exp. (0)", "Take Profit (+1)"]


# ==============================================================================
# PURGED K-FOLD CROSS-VALIDATION WITH EMBARGO
# ==============================================================================


class PurgedKFoldCV:
    """Purged K-Fold Cross-Validation with Post-Test Embargo.

    Addresses two structural vulnerabilities of standard K-Fold in financial time series:
    1. Label Overlap Leakage: Samples in the training set whose label evaluation horizon
       overlaps with the test fold window are purged.
    2. Autoregressive Contamination: Samples immediately following the test fold window
       are embargoed to prevent information leakage through serial correlation.

    Parameters
    ----------
    n_splits : int, default 5
        Number of cross-validation folds.
    t1 : Sequence or np.ndarray
        Array of event end dates/timestamps (stopping times tau*) corresponding
        to each observation's entry time.
    pct_embargo : float, default 0.01
        Proportion of total sample size applied as post-test embargo buffer.
    embargo_bars : Optional[int], default None
        Explicit integer bar count for post-test embargo buffer.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[Sequence[Any]] = None,
        pct_embargo: float = 0.01,
        embargo_bars: Optional[int] = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.t1 = np.asarray(t1) if t1 is not None else None
        self.pct_embargo = pct_embargo
        self.embargo_bars = embargo_bars
        self.fold_diagnostics_: List[Dict[str, Any]] = []

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
        times: Optional[Sequence[Any]] = None,
        t1: Optional[Sequence[Any]] = None,
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
        groups : array-like, optional
            Ignored; present for scikit-learn API compatibility.
        times : Sequence of shape (n_samples,), optional
            Observation start dates/timestamps. If None, observation row indices are assumed.
        t1 : Sequence of shape (n_samples,), optional
            Observation end dates/timestamps (stopping times tau*). If provided, overrides self.t1.

        Yields
        ------
        train_indices : np.ndarray
            The training set indices for that fold (purged and embargoed).
        test_indices : np.ndarray
            The testing set indices for that fold.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Resolve observation entry times t
        if times is None:
            times_arr = indices
        else:
            times_arr = np.asarray(times)

        # Resolve observation exit/stopping times t1
        effective_t1 = np.asarray(t1) if t1 is not None else self.t1
        if effective_t1 is None:
            # Fallback: point-in-time observations without overlap
            effective_t1 = times_arr
        else:
            effective_t1 = np.asarray(effective_t1)

        if len(times_arr) != n_samples or len(effective_t1) != n_samples:
            raise ValueError(
                f"Mismatch in sample lengths: X={n_samples}, times={len(times_arr)}, t1={len(effective_t1)}"
            )

        # Partition timeline into K contiguous chronological blocks based on unique dates/times
        unique_times = np.unique(times_arr)
        unique_times.sort()
        n_unique = len(unique_times)

        if n_unique < self.n_splits:
            raise ValueError(
                f"Not enough unique time periods ({n_unique}) for {self.n_splits} splits."
            )

        time_splits = np.array_split(unique_times, self.n_splits)
        self.fold_diagnostics_ = []

        # Calculate embargo size in terms of unique dates/bars
        if self.embargo_bars is not None:
            embargo_step = int(self.embargo_bars)
        else:
            embargo_step = max(1, int(n_unique * self.pct_embargo))

        for fold_idx, test_times_chunk in enumerate(time_splits):
            test_start_time = test_times_chunk[0]
            test_end_time = test_times_chunk[-1]

            # 1. Identify Test Set observations
            test_mask = (times_arr >= test_start_time) & (times_arr <= test_end_time)
            test_idx = indices[test_mask]

            if len(test_idx) == 0:
                logger.warning("Fold %d has empty test set!", fold_idx)
                continue

            # Earliest entry time in test set and maximum label resolution time in test set
            test_min_t = times_arr[test_idx].min()
            test_max_t1 = effective_t1[test_idx].max()

            # 2. Initial Training Candidates: all non-test observations
            candidate_train_mask = ~test_mask

            # 3. Purging:
            # Type A Purging (Left of test): Training samples with entry before test start,
            # but whose label evaluation horizon t1 extends into or across test:
            # t_i < test_min_t AND t1_i >= test_min_t
            purged_left_mask = candidate_train_mask & (times_arr < test_min_t) & (effective_t1 >= test_min_t)

            # Type B Purging (Right of test): Training samples with entry after test_end_time,
            # but before test events have fully concluded (t_i <= test_max_t1):
            purged_right_mask = candidate_train_mask & (times_arr > test_end_time) & (times_arr <= test_max_t1)

            purged_mask = purged_left_mask | purged_right_mask

            # 4. Embargo:
            # Determine embargo cutoff time following the latest test event resolution
            # Find the position of test_max_t1 in unique_times array
            max_t1_pos = np.searchsorted(unique_times, test_max_t1, side="right")
            embargo_idx_pos = min(n_unique - 1, max_t1_pos + embargo_step)
            embargo_cutoff_time = unique_times[embargo_idx_pos]

            embargo_mask = (
                candidate_train_mask
                & ~purged_mask
                & (times_arr > test_max_t1)
                & (times_arr <= embargo_cutoff_time)
            )

            # 5. Final Purged & Embargoed Training Set
            final_train_mask = candidate_train_mask & ~purged_mask & ~embargo_mask
            train_idx = indices[final_train_mask]

            # Diagnostic telemetry recording
            diag = {
                "fold": fold_idx,
                "test_start": str(test_start_time),
                "test_end": str(test_end_time),
                "test_max_t1": str(test_max_t1),
                "embargo_cutoff": str(embargo_cutoff_time),
                "n_test": int(len(test_idx)),
                "n_train_raw": int(candidate_train_mask.sum()),
                "n_purged_left": int(purged_left_mask.sum()),
                "n_purged_right": int(purged_right_mask.sum()),
                "n_purged_total": int(purged_mask.sum()),
                "n_embargoed": int(embargo_mask.sum()),
                "n_train_final": int(len(train_idx)),
                "train_retention_pct": float(len(train_idx) / max(1, candidate_train_mask.sum()) * 100.0),
            }
            self.fold_diagnostics_.append(diag)

            logger.info(
                "Fold %d: Test [%s to %s] | N_test=%d | Purged=%d | Embargoed=%d | Final Train=%d (%.1f%%)",
                fold_idx,
                test_start_time,
                test_end_time,
                diag["n_test"],
                diag["n_purged_total"],
                diag["n_embargoed"],
                diag["n_train_final"],
                diag["train_retention_pct"],
            )

            yield train_idx, test_idx

    def get_n_splits(self, X: Optional[Any] = None, y: Optional[Any] = None, groups: Optional[Any] = None) -> int:
        """Return number of splitting iterations in the cross-validator."""
        return self.n_splits


# ==============================================================================
# MACHINE LEARNING TRAINING & EVALUATION ENGINE
# ==============================================================================


class PredictiveModelTrainer:
    """Orchestrates model training, Purged K-Fold validation, and alpha metric extraction.

    Parameters
    ----------
    config : ModelTrainerConfig, optional
        Configuration holding hyperparameters and feature specifications.
    """

    def __init__(self, config: Optional[ModelTrainerConfig] = None) -> None:
        self.config = config or ModelTrainerConfig()
        self.models_: List[Any] = []
        self.feature_names_: List[str] = []
        self.cv_results_: Dict[str, Any] = {}
        self.feature_importance_df_: Optional[pl.DataFrame] = None
        self.cv_diagnostics_: List[Dict[str, Any]] = []

    def prepare_dataset(
        self,
        df: pl.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], pl.DataFrame]:
        """Validate schema, filter unresolved rows, encode targets, and derive event timestamps.

        Parameters
        ----------
        df : pl.DataFrame
            Enriched dataset containing candlestick features and triple-barrier columns.

        Returns
        -------
        X : np.ndarray (float64)
            Feature design matrix.
        y : np.ndarray (int64)
            Encoded multi-class target labels in {0, 1, 2}.
        times : np.ndarray
            Observation start dates/timestamps.
        t1 : np.ndarray
            Observation barrier touch/expiration dates (stopping times).
        feature_names : List[str]
            Names of feature columns included in X.
        cleaned_df : pl.DataFrame
            Cleaned and aligned dataframe.
        """
        cfg = self.config
        cols = df.columns

        # Verify required columns exist
        for req in [cfg.date_col, cfg.symbol_col, cfg.target_col, cfg.touch_bars_col]:
            if req not in cols:
                raise ValueError(f"Required column '{req}' missing from input dataset.")

        # Filter out unresolved rows if barrier_type column is present
        filtered_df = df
        if cfg.barrier_type_col in cols:
            filtered_df = filtered_df.filter(pl.col(cfg.barrier_type_col) != "unresolved")

        # Filter out null targets
        filtered_df = filtered_df.filter(pl.col(cfg.target_col).is_not_null())

        # Ensure strict sort by symbol and date
        filtered_df = filtered_df.sort([cfg.symbol_col, cfg.date_col])

        # Compute exact event end date (t1) per symbol using touch_bars offset
        # For each symbol, entry date is t, and stopping time date is t + touch_bars
        t1_dates: List[Any] = []
        symbols = filtered_df[cfg.symbol_col].unique().to_list()

        # Build list of end dates aligned with filtered_df rows
        symbol_dfs: List[pl.DataFrame] = []
        for sym in symbols:
            sym_df = filtered_df.filter(pl.col(cfg.symbol_col) == sym).sort(cfg.date_col)
            dates = sym_df[cfg.date_col].to_numpy()
            touch_bars = sym_df[cfg.touch_bars_col].to_numpy()
            n = len(dates)
            end_indices = np.minimum(np.arange(n) + touch_bars, n - 1)
            end_dates = dates[end_indices]
            sym_df = sym_df.with_columns(pl.Series("event_end_date", end_dates))
            symbol_dfs.append(sym_df)

        aligned_df = pl.concat(symbol_dfs).sort([cfg.date_col, cfg.symbol_col])

        # Select feature columns
        available_cols = aligned_df.columns
        if cfg.feature_cols:
            selected_features = [c for c in cfg.feature_cols if c in available_cols]
        else:
            selected_features = [c for c in DEFAULT_CONTINUOUS_FEATURES if c in available_cols]

        if not selected_features:
            raise ValueError("No valid feature columns found in dataset.")

        # Filter out any rows with NaNs in selected features
        for f in selected_features:
            aligned_df = aligned_df.filter(pl.col(f).is_not_null() & ~pl.col(f).is_nan())

        # Extract numpy arrays
        X = aligned_df.select(selected_features).to_numpy()
        raw_y = aligned_df[cfg.target_col].to_numpy()
        times = aligned_df[cfg.date_col].to_numpy()
        t1 = aligned_df["event_end_date"].to_numpy()

        # Map {-1, 0, 1} to {0, 1, 2}
        y = np.vectorize(LABEL_ENCODING.get)(raw_y)

        logger.info(
            "Prepared dataset: %d samples, %d features, Class distribution: %s",
            len(y),
            len(selected_features),
            {name: int((raw_y == val).sum()) for val, name in [(-1, "StopLoss"), (0, "VerticalExp"), (1, "TakeProfit")]},
        )

        return X, y, times, t1, selected_features, aligned_df

    def _build_model(self) -> Any:
        """Construct a configured classifier instance with regularization."""
        cfg = self.config
        if HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                num_leaves=cfg.num_leaves,
                min_child_samples=cfg.min_child_samples,
                subsample=cfg.subsample,
                subsample_freq=1,
                colsample_bytree=cfg.colsample_bytree,
                reg_alpha=cfg.reg_alpha,
                reg_lambda=cfg.reg_lambda,
                class_weight=cfg.class_weight,
                random_state=cfg.random_state,
                verbosity=-1,
                n_jobs=-1,
            )
        else:
            logger.warning("LightGBM unavailable; falling back to HistGradientBoostingClassifier.")
            return HistGradientBoostingClassifier(
                max_iter=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_child_samples,
                l2_regularization=cfg.reg_lambda,
                random_state=cfg.random_state,
            )

    def fit_cv(
        self,
        df: pl.DataFrame,
    ) -> Dict[str, Any]:
        """Execute Purged K-Fold Cross-Validation training and out-of-sample evaluation.

        Parameters
        ----------
        df : pl.DataFrame
            Dataset containing features, dates, and triple-barrier target labels.

        Returns
        -------
        results : Dict[str, Any]
            Comprehensive evaluation metrics, out-of-sample predictions, and diagnostics.
        """
        cfg = self.config
        X, y, times, t1, feature_names, aligned_df = self.prepare_dataset(df)
        self.feature_names_ = feature_names

        cv = PurgedKFoldCV(
            n_splits=cfg.n_splits,
            pct_embargo=cfg.pct_embargo,
            embargo_bars=cfg.embargo_bars,
        )

        n_samples = len(y)
        oof_predictions = np.zeros(n_samples, dtype=np.int64)
        oof_probabilities = np.zeros((n_samples, 3), dtype=np.float64)
        test_indices_all: List[int] = []

        fold_metrics: List[Dict[str, float]] = []
        feature_importances_gain: List[np.ndarray] = []
        feature_importances_split: List[np.ndarray] = []
        self.models_ = []

        logger.info(
            "Executing Purged K-Fold CV (K=%d, Embargo=%.1f%%) on %d observations...",
            cfg.n_splits,
            cfg.pct_embargo * 100.0 if cfg.embargo_bars is None else cfg.embargo_bars,
            n_samples,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(
            cv.split(X, y, times=times, t1=t1)
        ):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            model = self._build_model()
            model.fit(X_train, y_train)
            self.models_.append(model)

            # Out-of-sample inferences
            probs = model.predict_proba(X_test)
            preds = model.predict(X_test)

            oof_predictions[test_idx] = preds
            oof_probabilities[test_idx] = probs
            test_indices_all.extend(test_idx.tolist())

            # Evaluate fold metrics
            fold_acc = accuracy_score(y_test, preds)
            fold_bal_acc = balanced_accuracy_score(y_test, preds)
            fold_f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
            fold_logloss = log_loss(y_test, probs, labels=[0, 1, 2])

            fold_metrics.append({
                "fold": fold_idx,
                "accuracy": fold_acc,
                "balanced_accuracy": fold_bal_acc,
                "f1_macro": fold_f1_macro,
                "log_loss": fold_logloss,
            })

            # Feature importance extraction
            if HAS_LIGHTGBM and hasattr(model, "booster_"):
                gain_imp = model.booster_.feature_importance(importance_type="gain")
                split_imp = model.booster_.feature_importance(importance_type="split")
            elif hasattr(model, "feature_importances_"):
                gain_imp = model.feature_importances_
                split_imp = model.feature_importances_
            else:
                gain_imp = np.zeros(len(feature_names))
                split_imp = np.zeros(len(feature_names))

            feature_importances_gain.append(gain_imp)
            feature_importances_split.append(split_imp)

        self.cv_diagnostics_ = cv.fold_diagnostics_

        # Calculate Global Out-of-Sample Metrics
        test_eval_mask = np.zeros(n_samples, dtype=bool)
        test_eval_mask[test_indices_all] = True

        y_true_eval = y[test_eval_mask]
        y_pred_eval = oof_predictions[test_eval_mask]
        y_prob_eval = oof_probabilities[test_eval_mask]

        global_metrics = self.compute_out_of_sample_metrics(y_true_eval, y_pred_eval, y_prob_eval)

        # Feature Importance Summary Table
        feat_imp_df = self._compute_feature_importance_dataframe(
            feature_names, feature_importances_gain, feature_importances_split
        )
        self.feature_importance_df_ = feat_imp_df

        self.cv_results_ = {
            "global_metrics": global_metrics,
            "fold_metrics": fold_metrics,
            "cv_diagnostics": self.cv_diagnostics_,
            "feature_importance": feat_imp_df,
            "oof_predictions": y_pred_eval,
            "oof_probabilities": y_prob_eval,
            "oof_true": y_true_eval,
            "aligned_df": aligned_df,
            "feature_names": feature_names,
        }

        logger.info(
            "Out-of-Sample Results: Accuracy=%.2f%% | Balanced Acc=%.2f%% | F1 Macro=%.4f | Log Loss=%.4f",
            global_metrics["accuracy"] * 100.0,
            global_metrics["balanced_accuracy"] * 100.0,
            global_metrics["f1_macro"],
            global_metrics["log_loss"],
        )

        return self.cv_results_

    @staticmethod
    def compute_out_of_sample_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute exhaustive classification performance metrics on Out-Of-Sample predictions.

        Parameters
        ----------
        y_true : np.ndarray
            True encoded labels in {0, 1, 2}.
        y_pred : np.ndarray
            Predicted encoded labels in {0, 1, 2}.
        y_prob : np.ndarray
            Predicted class probabilities of shape (n_samples, 3).

        Returns
        -------
        metrics : Dict[str, Any]
            Scalar scores, per-class metrics, and confusion matrix.
        """
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        loss = log_loss(y_true, y_prob, labels=[0, 1, 2])

        # Macro and weighted metrics
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)

        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class scores
        prec_per_class = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        rec_per_class = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

        conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        return {
            "n_samples": int(len(y_true)),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "log_loss": float(loss),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_mac),
            "precision_weighted": float(prec_weighted),
            "recall_weighted": float(rec_weighted),
            "f1_weighted": float(f1_wt),
            "per_class": {
                "stop_loss": {
                    "label": -1,
                    "precision": float(prec_per_class[0]),
                    "recall": float(rec_per_class[0]),
                    "f1": float(f1_per_class[0]),
                    "support": int((y_true == 0).sum()),
                },
                "vertical_expiration": {
                    "label": 0,
                    "precision": float(prec_per_class[1]),
                    "recall": float(rec_per_class[1]),
                    "f1": float(f1_per_class[1]),
                    "support": int((y_true == 1).sum()),
                },
                "take_profit": {
                    "label": 1,
                    "precision": float(prec_per_class[2]),
                    "recall": float(rec_per_class[2]),
                    "f1": float(f1_per_class[2]),
                    "support": int((y_true == 2).sum()),
                },
            },
            "confusion_matrix": conf_mat,
        }

    @staticmethod
    def _compute_feature_importance_dataframe(
        feature_names: List[str],
        gain_importances: List[np.ndarray],
        split_importances: List[np.ndarray],
    ) -> pl.DataFrame:
        """Aggregate cross-fold feature importance into an auditable Polars DataFrame."""
        gain_arr = np.array(gain_importances)
        split_arr = np.array(split_importances)

        gain_mean = gain_arr.mean(axis=0)
        gain_std = gain_arr.std(axis=0)
        split_mean = split_arr.mean(axis=0)
        split_std = split_arr.std(axis=0)

        total_gain = gain_mean.sum()
        gain_pct = (gain_mean / total_gain * 100.0) if total_gain > 0 else np.zeros_like(gain_mean)

        df = pl.DataFrame({
            "feature": feature_names,
            "importance_gain_mean": gain_mean,
            "importance_gain_std": gain_std,
            "importance_gain_pct": gain_pct,
            "importance_split_mean": split_mean,
            "importance_split_std": split_std,
        }).sort("importance_gain_mean", descending=True)

        return df

    # ==========================================================================
    # PUBLICATION-GRADE MARKDOWN EXPORTERS & VISUALIZATIONS
    # ==========================================================================

    @staticmethod
    def generate_metrics_markdown_table(metrics: Dict[str, Any]) -> str:
        """Render publication-ready Markdown table summarizing OOS evaluation metrics."""
        pc = metrics["per_class"]
        header = (
            "| Class Objective | Encoded Label | Support | Precision | Recall | F1-Score |\n"
            "|:---|:---:|---:|---:|---:|---:|\n"
        )
        row_sl = (
            f"| **Stop Loss (Barrier Hit)** | -1 | {pc['stop_loss']['support']:,} "
            f"| {pc['stop_loss']['precision']:.2%} | {pc['stop_loss']['recall']:.2%} | {pc['stop_loss']['f1']:.4f} |\n"
        )
        row_vert = (
            f"| **Vertical Expiration (Time Out)** | 0 | {pc['vertical_expiration']['support']:,} "
            f"| {pc['vertical_expiration']['precision']:.2%} | {pc['vertical_expiration']['recall']:.2%} | {pc['vertical_expiration']['f1']:.4f} |\n"
        )
        row_tp = (
            f"| **Take Profit (Barrier Hit)** | +1 | {pc['take_profit']['support']:,} "
            f"| {pc['take_profit']['precision']:.2%} | {pc['take_profit']['recall']:.2%} | {pc['take_profit']['f1']:.4f} |\n"
        )
        divider = "|:---|:---:|---:|---:|---:|---:|\n"
        summary_macro = (
            f"| **Macro Average** | — | {metrics['n_samples']:,} "
            f"| {metrics['precision_macro']:.2%} | {metrics['recall_macro']:.2%} | **{metrics['f1_macro']:.4f}** |\n"
        )
        summary_weighted = (
            f"| **Weighted Average** | — | {metrics['n_samples']:,} "
            f"| {metrics['precision_weighted']:.2%} | {metrics['recall_weighted']:.2%} | {metrics['f1_weighted']:.4f} |\n"
        )
        summary_overall = (
            f"| **Overall Accuracy** | — | — | — | — | **{metrics['accuracy']:.2%}** |\n"
            f"| **Balanced Accuracy** | — | — | — | — | **{metrics['balanced_accuracy']:.2%}** |\n"
            f"| **Multi-Class Log Loss** | — | — | — | — | **{metrics['log_loss']:.4f}** |\n"
        )

        return header + row_sl + row_vert + row_tp + divider + summary_macro + summary_weighted + summary_overall

    @staticmethod
    def generate_cv_diagnostics_markdown_table(cv_diagnostics: List[Dict[str, Any]]) -> str:
        """Render publication-ready Markdown table summarizing Purged CV partition dynamics."""
        header = (
            "| Fold | Test Period | Test Obs | Pre-Purge Train | Purged Obs | Embargoed Obs | Final Train Obs | Retention |\n"
            "|:---:|:---|---:|---:|---:|---:|---:|---:|\n"
        )
        rows: List[str] = []
        for d in cv_diagnostics:
            period = f"{d['test_start'][:10]} to {d['test_end'][:10]}"
            row = (
                f"| **Fold {d['fold'] + 1}** | {period} | {d['n_test']:,} | {d['n_train_raw']:,} "
                f"| {d['n_purged_total']:,} | {d['n_embargoed']:,} | {d['n_train_final']:,} | {d['train_retention_pct']:.1f}% |"
            )
            rows.append(row)
        return header + "\n".join(rows) + "\n"

    @staticmethod
    def generate_feature_importance_markdown_table(
        feat_df: pl.DataFrame,
        top_n: int = 10,
    ) -> str:
        """Render publication-ready Markdown table of top feature importances."""
        header = (
            "| Rank | Predictive Feature | Gain Importance (Mean $\\pm$ Std) | Relative Gain Share | Split Count |\n"
            "|:---:|:---|---:|---:|---:|\n"
        )
        rows: List[str] = []
        for rank, row in enumerate(feat_df.head(top_n).iter_rows(named=True), start=1):
            line = (
                f"| {rank} | `{row['feature']}` | {row['importance_gain_mean']:.2f} $\\pm$ {row['importance_gain_std']:.2f} "
                f"| **{row['importance_gain_pct']:.1f}%** | {row['importance_split_mean']:.1f} |"
            )
            rows.append(line)
        return header + "\n".join(rows) + "\n"

    @staticmethod
    def plot_feature_importance(
        feat_df: pl.DataFrame,
        top_n: int = 12,
        title: str = "Feature Importance Attribution (Cross-Fold Gain)",
        figsize: Tuple[float, float] = (8.5, 5.0),
    ) -> plt.Figure:
        """Generate horizontal bar chart of feature importances with cross-fold error bars."""
        top_df = feat_df.head(top_n).reverse()
        features = top_df["feature"].to_list()
        gain_means = top_df["importance_gain_pct"].to_numpy()
        gain_stds = (
            top_df["importance_gain_std"].to_numpy()
            / max(1e-8, feat_df["importance_gain_mean"].sum())
            * 100.0
        )

        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        y_pos = np.arange(len(features))

        colors = ["#1f77b4" if i < len(features) - 3 else "#2ca02c" for i in range(len(features))]

        bars = ax.barh(
            y_pos,
            gain_means,
            xerr=gain_stds,
            align="center",
            alpha=0.85,
            color=colors,
            capsize=3,
            edgecolor="#333333",
            linewidth=0.6,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9, fontweight="medium")
        ax.set_xlabel("Relative Gain Importance Share (%)", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.4, axis="x")

        # Add percentage labels to bars
        for bar, val in zip(bars, gain_means):
            ax.text(
                val + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_xlim(0, max(gain_means) * 1.25)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_confusion_matrix(
        conf_mat: np.ndarray,
        class_names: Sequence[str] = ("Stop Loss (-1)", "Vertical Exp. (0)", "Take Profit (+1)"),
        title: str = "Out-of-Sample Confusion Matrix (Normalized)",
        figsize: Tuple[float, float] = (6.0, 5.0),
    ) -> plt.Figure:
        """Generate annotated heatmap of Out-of-Sample classification confusion matrix."""
        norm_cm = conf_mat.astype(float) / np.maximum(1, conf_mat.sum(axis=1)[:, np.newaxis])

        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        cax = ax.imshow(norm_cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_ylabel("True Market Event", fontsize=10, fontweight="bold")
        ax.set_xlabel("Model Predicted Class", fontsize=10, fontweight="bold")

        # Annotate text
        thresh = norm_cm.max() / 2.0
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                raw_cnt = conf_mat[i, j]
                pct = norm_cm[i, j] * 100.0
                ax.text(
                    j,
                    i,
                    f"{pct:.1f}%\n({raw_cnt:,})",
                    ha="center",
                    va="center",
                    color="white" if norm_cm[i, j] > thresh else "black",
                    fontsize=9,
                    fontweight="bold",
                )

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_cv_architecture(
        cv_diagnostics: List[Dict[str, Any]],
        figsize: Tuple[float, float] = (10.0, 4.0),
    ) -> plt.Figure:
        """Visualize timeline partition of Purged K-Fold CV with Embargo."""
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        n_splits = len(cv_diagnostics)

        y_positions = np.arange(n_splits)

        for d in cv_diagnostics:
            fold = d["fold"]
            y = n_splits - 1 - fold

            n_train = d["n_train_final"]
            n_purged = d["n_purged_total"]
            n_embargo = d["n_embargoed"]
            n_test = d["n_test"]
            total = n_train + n_purged + n_embargo + n_test

            pct_train = n_train / total * 100
            pct_test = n_test / total * 100
            pct_purge = n_purged / total * 100
            pct_embargo = n_embargo / total * 100

            # Horizontal stacked bar
            left = 0
            ax.barh(y, pct_train, left=left, color="#2b5c8f", alpha=0.8, height=0.55, label="Effective Train" if fold == 0 else "")
            left += pct_train

            ax.barh(y, pct_purge, left=left, color="#d62728", alpha=0.8, height=0.55, label="Purged (Overlap)" if fold == 0 else "")
            left += pct_purge

            ax.barh(y, pct_test, left=left, color="#2ca02c", alpha=0.9, height=0.55, label="Test Fold (OOS)" if fold == 0 else "")
            left += pct_test

            ax.barh(y, pct_embargo, left=left, color="#ff7f0e", alpha=0.8, height=0.55, label="Embargo (Buffer)" if fold == 0 else "")

        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"Fold {n_splits - i}" for i in range(n_splits)], fontsize=9, fontweight="bold")
        ax.set_xlabel("Partition Share of Historical Observations (%)", fontsize=10, fontweight="bold")
        ax.set_title("Purged K-Fold Cross-Validation Architecture with Embargo", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlim(0, 100)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=4, frameon=True, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3, axis="x")

        plt.tight_layout()
        return fig


# ==============================================================================
# PIPELINE ORCHESTRATION & CLI ENTRYPOINT
# ==============================================================================


def run_model_training_pipeline(
    input_path: Optional[Path] = None,
    output_dir: Path = Path("outputs"),
    force_synthetic: bool = False,
    config: Optional[ModelTrainerConfig] = None,
) -> Dict[str, Any]:
    """Execute end-to-end data preparation, feature engineering, and Purged CV training."""
    from src.candlestick_features import compute_candlestick_features, generate_synthetic_ohlcv
    from src.triple_barrier import compute_triple_barrier_labels

    cfg = config or ModelTrainerConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire Market Data
    df_raw: pl.DataFrame
    if input_path and input_path.exists() and not force_synthetic:
        logger.info("Loading market dataset from %s...", input_path)
        df_raw = pl.read_parquet(input_path)
    else:
        logger.info("Generating calibrated synthetic OHLCV universe (8 assets, 1258 sessions)...")
        df_raw = generate_synthetic_ohlcv(
            symbols=["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TLT", "GLD"],
            n_days=1258,
            seed=42,
        )

    # 2. Enrich with Candlestick Features if not present
    if "body_rel_size" not in df_raw.columns:
        logger.info("Computing continuous candlestick and microstructural features...")
        df_enriched = compute_candlestick_features(df_raw, include_forward_returns=False)
    else:
        df_enriched = df_raw

    # 3. Enrich with Triple-Barrier Labels if not present
    if "tb_label" not in df_enriched.columns:
        logger.info("Computing Triple-Barrier labels with dynamic volatility scaling...")
        df_labeled = compute_triple_barrier_labels(df_enriched)
    else:
        df_labeled = df_enriched

    # 4. Train Predictive Model with Purged K-Fold CV & Embargo
    trainer = PredictiveModelTrainer(cfg)
    results = trainer.fit_cv(df_labeled)

    # 5. Export Auditable Visualizations and Tables
    feat_df = results["feature_importance"]
    metrics = results["global_metrics"]
    diag = results["cv_diagnostics"]

    # Export Feature Importance Table to CSV/Parquet
    feat_df.write_csv(output_dir / "feature_importance.csv")
    feat_df.write_parquet(output_dir / "feature_importance.parquet")
    logger.info("Exported feature importance to %s", output_dir / "feature_importance.csv")

    # Generate and save figures
    fig_imp = trainer.plot_feature_importance(feat_df)
    fig_imp.savefig(output_dir / "feature_importance.png", bbox_inches="tight")
    plt.close(fig_imp)

    fig_cm = trainer.plot_confusion_matrix(metrics["confusion_matrix"])
    fig_cm.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig_cm)

    fig_cv = trainer.plot_cv_architecture(diag)
    fig_cv.savefig(output_dir / "purged_cv_architecture.png", bbox_inches="tight")
    plt.close(fig_cv)
    logger.info("Exported diagnostic figures to %s", output_dir)

    return results


def main() -> None:
    """CLI entrypoint for Model Trainer."""
    parser = argparse.ArgumentParser(
        description="ML4T Model Trainer: LightGBM classification with Purged K-Fold CV & Embargo."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/curated_market_data.parquet"),
        help="Path to input curated parquet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to output directory for logs, metrics, and plots.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic calibrated market data generation.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of Purged K-Fold cross-validation splits (default 5).",
    )
    parser.add_argument(
        "--pct-embargo",
        type=float,
        default=0.01,
        help="Post-test embargo proportion (default 0.01 = 1%%).",
    )
    args = parser.parse_args()

    cfg = ModelTrainerConfig(
        n_splits=args.n_splits,
        pct_embargo=args.pct_embargo,
    )

    results = run_model_training_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        force_synthetic=args.synthetic,
        config=cfg,
    )

    trainer = PredictiveModelTrainer(cfg)
    print("\n" + "=" * 80)
    print("PURGED K-FOLD CROSS-VALIDATION DIAGNOSTICS")
    print("=" * 80)
    print(trainer.generate_cv_diagnostics_markdown_table(results["cv_diagnostics"]))

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE EVALUATION METRICS (TRIPLE-BARRIER CLASSIFICATION)")
    print("=" * 80)
    print(trainer.generate_metrics_markdown_table(results["global_metrics"]))

    print("\n" + "=" * 80)
    print("TOP PREDICTIVE FEATURES (CROSS-FOLD GAIN ATTRIBUTION)")
    print("=" * 80)
    print(trainer.generate_feature_importance_markdown_table(results["feature_importance"]))


if __name__ == "__main__":
    main()
