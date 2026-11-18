"""Association rule mining and algorithm benchmarking utilities using mlxtend.

Implements both Apriori and FP-Growth frequent itemset mining along with
association rule extraction (support, confidence, lift, and conviction).
Tracks wall-clock execution time and peak Python memory usage for each run
so both algorithms can be compared directly across threshold parameters.
"""

from __future__ import annotations

import gc
import logging
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

LOGGER = logging.getLogger(__name__)

# Standard columns kept when formatting association rules for export and review
RULE_COLUMNS = [
    "antecedents",
    "consequents",
    "antecedent support",
    "consequent support",
    "support",
    "confidence",
    "lift",
    "conviction",
]


def _extract_rules(
    frequent_itemsets: pd.DataFrame,
    num_transactions: int,
    metric: str = "confidence",
    min_threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate association rules from frequent itemsets and sort by lift."""
    # If no frequent itemsets met min_support, return an empty DataFrame with expected columns
    if frequent_itemsets.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)

    # Newer versions of mlxtend (>= 0.23.2) require num_itemsets, whereas 0.23.0/0.23.1 do not.
    try:
        rules = association_rules(
            frequent_itemsets,
            num_itemsets=num_transactions,
            metric=metric,
            min_threshold=min_threshold,
        )
    except TypeError:
        rules = association_rules(
            frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold,
        )

    if rules.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)

    # Sort rules so the strongest associations (highest lift, then confidence and support) come first
    rules = rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False],
    )
    rules = rules.reset_index(drop=True)
    return rules


def _compute_benchmark_metrics(
    algorithm_name: str,
    basket_matrix: pd.DataFrame,
    min_support: float,
    metric: str,
    min_threshold: float,
    frequent_itemsets: pd.DataFrame,
    rules: pd.DataFrame,
    itemset_time_sec: float,
    rule_time_sec: float,
    peak_memory_mb: float,
) -> dict[str, Any]:
    """Build a summary dictionary of rule metrics, runtime, and peak memory usage."""
    # When confidence == 1.0, conviction is mathematically infinite (division by zero).
    # We exclude infinities when taking the average conviction so the mean remains readable.
    if not rules.empty and "conviction" in rules.columns:
        finite_conviction = rules["conviction"].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite_conviction.empty:
            avg_conviction = float(finite_conviction.mean())
        else:
            avg_conviction = np.nan
    else:
        avg_conviction = np.nan

    if not rules.empty:
        avg_support = float(rules["support"].mean())
        avg_confidence = float(rules["confidence"].mean())
        avg_lift = float(rules["lift"].mean())
        max_lift = float(rules["lift"].max())
    else:
        avg_support = np.nan
        avg_confidence = np.nan
        avg_lift = np.nan
        max_lift = np.nan

    total_time_sec = itemset_time_sec + rule_time_sec

    metrics = {
        "algorithm": algorithm_name,
        "n_transactions": int(basket_matrix.shape[0]),
        "n_items": int(basket_matrix.shape[1]),
        "min_support": float(min_support),
        "metric": metric,
        "min_threshold": float(min_threshold),
        "frequent_itemsets": int(len(frequent_itemsets)),
        "rules_generated": int(len(rules)),
        "avg_support": avg_support,
        "avg_confidence": avg_confidence,
        "avg_lift": avg_lift,
        "max_lift": max_lift,
        "avg_conviction": avg_conviction,
        "itemset_time_sec": round(itemset_time_sec, 6),
        "rule_time_sec": round(rule_time_sec, 6),
        "execution_time_sec": round(total_time_sec, 6),
        "peak_memory_mb": round(peak_memory_mb, 4),
    }
    return metrics


def _run_itemset_miner(
    miner_function: Any,
    basket_matrix: pd.DataFrame,
    min_support: float,
    max_len: int | None = None,
    low_memory: bool | None = None,
) -> pd.DataFrame:
    """Invoke an mlxtend itemset miner, converting sparse DataFrames to dense bool if needed."""
    miner_kwargs: dict[str, Any] = {
        "min_support": min_support,
        "use_colnames": True,
        "max_len": max_len,
    }
    if low_memory is not None:
        miner_kwargs["low_memory"] = low_memory

    try:
        return miner_function(basket_matrix, **miner_kwargs)
    except (TypeError, ValueError):
        # Fallback if a specific mlxtend/pandas version rejects SparseDtype input
        if hasattr(basket_matrix, "sparse"):
            dense_matrix = basket_matrix.sparse.to_dense().astype(bool)
            return miner_function(dense_matrix, **miner_kwargs)
        raise


def run_apriori(
    basket_matrix: pd.DataFrame,
    min_support: float = 0.02,
    metric: str = "confidence",
    min_threshold: float = 0.5,
    max_len: int | None = None,
    low_memory: bool = False,
) -> dict[str, Any]:
    """Run Apriori frequent itemset mining and derive association rules with profiling.

    Returns a dictionary containing:
    - 'frequent_itemsets': DataFrame of itemsets and their support
    - 'rules': DataFrame of association rules (support, confidence, lift, conviction)
    - 'metrics': Dictionary of execution time, peak memory (MB), and average rule scores
    """
    # Collect garbage before starting tracemalloc so peak memory reflects this run cleanly
    gc.collect()
    tracemalloc.start()

    try:
        # Step 1: Mine frequent itemsets with Apriori and measure time
        t_start = time.perf_counter()
        frequent_itemsets = _run_itemset_miner(
            miner_function=apriori,
            basket_matrix=basket_matrix,
            min_support=min_support,
            max_len=max_len,
            low_memory=low_memory,
        )
        t_itemsets_done = time.perf_counter()

        # Step 2: Generate association rules (support, confidence, lift, conviction)
        rules = _extract_rules(
            frequent_itemsets=frequent_itemsets,
            num_transactions=len(basket_matrix),
            metric=metric,
            min_threshold=min_threshold,
        )
        t_rules_done = time.perf_counter()

        # Step 3: Read peak allocated memory during this run
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    if not frequent_itemsets.empty:
        frequent_itemsets = frequent_itemsets.sort_values(
            by="support", ascending=False
        ).reset_index(drop=True)

    metrics = _compute_benchmark_metrics(
        algorithm_name="Apriori",
        basket_matrix=basket_matrix,
        min_support=min_support,
        metric=metric,
        min_threshold=min_threshold,
        frequent_itemsets=frequent_itemsets,
        rules=rules,
        itemset_time_sec=t_itemsets_done - t_start,
        rule_time_sec=t_rules_done - t_itemsets_done,
        peak_memory_mb=peak_bytes / (1024 * 1024),
    )

    LOGGER.info(
        "Apriori (min_support=%.3f, %s>=%.2f): %d itemsets, %d rules in %.4fs (peak %.2f MB).",
        min_support,
        metric,
        min_threshold,
        metrics["frequent_itemsets"],
        metrics["rules_generated"],
        metrics["execution_time_sec"],
        metrics["peak_memory_mb"],
    )

    return {
        "frequent_itemsets": frequent_itemsets,
        "rules": rules,
        "metrics": metrics,
    }


def run_fpgrowth(
    basket_matrix: pd.DataFrame,
    min_support: float = 0.02,
    metric: str = "confidence",
    min_threshold: float = 0.5,
    max_len: int | None = None,
) -> dict[str, Any]:
    """Run FP-Growth frequent itemset mining and derive association rules with profiling.

    Returns a dictionary containing:
    - 'frequent_itemsets': DataFrame of itemsets and their support
    - 'rules': DataFrame of association rules (support, confidence, lift, conviction)
    - 'metrics': Dictionary of execution time, peak memory (MB), and average rule scores
    """
    # Collect garbage before starting tracemalloc for an apples-to-apples memory comparison
    gc.collect()
    tracemalloc.start()

    try:
        # Step 1: Mine frequent itemsets with FP-Growth and measure time
        t_start = time.perf_counter()
        frequent_itemsets = _run_itemset_miner(
            miner_function=fpgrowth,
            basket_matrix=basket_matrix,
            min_support=min_support,
            max_len=max_len,
            low_memory=None,
        )
        t_itemsets_done = time.perf_counter()

        # Step 2: Generate association rules (support, confidence, lift, conviction)
        rules = _extract_rules(
            frequent_itemsets=frequent_itemsets,
            num_transactions=len(basket_matrix),
            metric=metric,
            min_threshold=min_threshold,
        )
        t_rules_done = time.perf_counter()

        # Step 3: Read peak allocated memory during this run
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    if not frequent_itemsets.empty:
        frequent_itemsets = frequent_itemsets.sort_values(
            by="support", ascending=False
        ).reset_index(drop=True)

    metrics = _compute_benchmark_metrics(
        algorithm_name="FP-Growth",
        basket_matrix=basket_matrix,
        min_support=min_support,
        metric=metric,
        min_threshold=min_threshold,
        frequent_itemsets=frequent_itemsets,
        rules=rules,
        itemset_time_sec=t_itemsets_done - t_start,
        rule_time_sec=t_rules_done - t_itemsets_done,
        peak_memory_mb=peak_bytes / (1024 * 1024),
    )

    LOGGER.info(
        "FP-Growth (min_support=%.3f, %s>=%.2f): %d itemsets, %d rules in %.4fs (peak %.2f MB).",
        min_support,
        metric,
        min_threshold,
        metrics["frequent_itemsets"],
        metrics["rules_generated"],
        metrics["execution_time_sec"],
        metrics["peak_memory_mb"],
    )

    return {
        "frequent_itemsets": frequent_itemsets,
        "rules": rules,
        "metrics": metrics,
    }


def format_rules_table(rules: pd.DataFrame, top_n: int | None = 50) -> pd.DataFrame:
    """Convert frozenset item columns to readable comma-separated strings for CSV export."""
    if rules.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)

    formatted = rules.copy()

    # Replace frozenset({'ITEM A', 'ITEM B'}) with readable 'ITEM A, ITEM B' text
    for col in ("antecedents", "consequents"):
        if col in formatted.columns:
            readable_values: list[str] = []
            for item_set in formatted[col]:
                if isinstance(item_set, (frozenset, set, list, tuple)):
                    readable_values.append(", ".join(sorted(str(x) for x in item_set)))
                else:
                    readable_values.append(str(item_set))
            formatted[col] = readable_values

    # Keep only standard evaluation columns that exist in the DataFrame
    available_cols: list[str] = []
    for col in RULE_COLUMNS:
        if col in formatted.columns:
            available_cols.append(col)

    formatted = formatted[available_cols]
    if top_n is not None:
        formatted = formatted.head(top_n)

    return formatted.reset_index(drop=True)
