"""End-to-end orchestration script for Group 6 Market Basket Analysis (SEED = 5018).

Pipeline steps:
1. Fix global random seeds (SEED = 5018) for full reproducibility.
2. Verify UCI Online Retail II dataset in data/ or download it automatically.
3. Clean cancelled/invalid transactions and sample unique invoices (random_state=5018).
4. Construct the boolean invoice-by-item basket matrix and save single-item frequencies.
5. Sequentially benchmark Apriori vs. FP-Growth across support and confidence thresholds,
   recording runtime, peak memory (MB), rule count, support, confidence, lift, and conviction.
6. Export comparison tables to results/tables/ and plots to results/figures/.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import matplotlib

# Use non-interactive 'Agg' backend so figures save cleanly without a GUI display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import GROUP_SEED
from src.association_rules import format_rules_table, run_apriori, run_fpgrowth
from src.download_data import DEFAULT_DATA_DIR, download_dataset
from src.preprocessing import create_basket_matrix, load_and_sample_data

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# Parameter grids evaluated to compare Apriori vs. FP-Growth (RQ1 & RQ2)
DEFAULT_SUPPORT_GRID: tuple[float, ...] = (0.03, 0.02, 0.015, 0.01)
DEFAULT_CONFIDENCE_GRID: tuple[float, ...] = (0.30, 0.50, 0.70)


def set_global_seed(seed: int = GROUP_SEED) -> None:
    """Fix Python and NumPy random seeds so every step is reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def export_frequency_baseline(
    basket_matrix: pd.DataFrame,
    output_path: Path,
    top_n: int = 30,
) -> pd.DataFrame:
    """Calculate single-item support (fraction of invoices) and export top items to CSV."""
    # Count how many invoices (rows) contain each item (column)
    if hasattr(basket_matrix, "sparse"):
        item_counts = np.asarray(basket_matrix.sparse.to_coo().sum(axis=0)).ravel()
    else:
        item_counts = basket_matrix.to_numpy().sum(axis=0)

    total_invoices = len(basket_matrix)
    freq_df = pd.DataFrame(
        {
            "item": basket_matrix.columns,
            "invoice_count": item_counts.astype(int),
            "support": item_counts / total_invoices,
        }
    )

    freq_df = freq_df.sort_values(
        by=["invoice_count", "item"],
        ascending=[False, True],
    )
    freq_df = freq_df.head(top_n).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    freq_df.to_csv(output_path, index=False)
    LOGGER.info("Saved top-%d item frequency baseline to %s", top_n, output_path)
    return freq_df


def save_benchmark_figures(
    benchmark_df: pd.DataFrame,
    reference_rules: pd.DataFrame,
    figures_dir: Path = RESULTS_FIGURES_DIR,
) -> None:
    """Generate and save runtime, memory, and rule-distribution plots to results/figures/."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Filter to reference confidence threshold (0.50) for clean 1-to-1 algorithm bars
    ref_mask = np.isclose(benchmark_df["min_threshold"], 0.50)
    subset_df = benchmark_df.loc[ref_mask]
    if subset_df.empty:
        subset_df = benchmark_df

    # Plot 1: Execution time comparison (Apriori vs FP-Growth)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=subset_df,
        x="min_support",
        y="execution_time_sec",
        hue="algorithm",
        ax=ax,
    )
    ax.set_title("Apriori vs. FP-Growth Execution Time (Confidence >= 0.50)")
    ax.set_xlabel("Minimum Support Threshold")
    ax.set_ylabel("Execution Time (seconds)")
    fig.tight_layout()
    fig.savefig(figures_dir / "algorithm_runtime_comparison.png", dpi=150)
    plt.close(fig)

    # Plot 2: Peak memory usage comparison (Apriori vs FP-Growth)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=subset_df,
        x="min_support",
        y="peak_memory_mb",
        hue="algorithm",
        ax=ax,
    )
    ax.set_title("Apriori vs. FP-Growth Peak Memory Usage (Confidence >= 0.50)")
    ax.set_xlabel("Minimum Support Threshold")
    ax.set_ylabel("Peak Traced Memory (MB)")
    fig.tight_layout()
    fig.savefig(figures_dir / "algorithm_memory_comparison.png", dpi=150)
    plt.close(fig)

    # Plot 3: Heatmap of rule counts across Support x Confidence thresholds
    fp_only = benchmark_df.loc[benchmark_df["algorithm"] == "FP-Growth"]
    if not fp_only.empty:
        rule_pivot = fp_only.pivot(
            index="min_support",
            columns="min_threshold",
            values="rules_generated",
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(rule_pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
        ax.set_title("Generated Association Rules Across Support & Confidence")
        ax.set_xlabel("Minimum Confidence Threshold")
        ax.set_ylabel("Minimum Support Threshold")
        fig.tight_layout()
        fig.savefig(figures_dir / "rule_count_heatmap.png", dpi=150)
        plt.close(fig)

    # Plot 4: Scatter plot of Support vs Confidence colored by Lift for reference rules
    if not reference_rules.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            reference_rules["support"],
            reference_rules["confidence"],
            c=reference_rules["lift"],
            cmap="viridis",
            alpha=0.8,
        )
        fig.colorbar(scatter, ax=ax, label="Lift")
        ax.set_title("Association Rules: Support vs. Confidence (Colored by Lift)")
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        fig.tight_layout()
        fig.savefig(figures_dir / "rules_support_confidence_lift.png", dpi=150)
        plt.close(fig)

    LOGGER.info("Saved diagnostic figures to %s", figures_dir)


def run_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    seed: int = GROUP_SEED,
    n_invoices: int = 5000,
    support_grid: tuple[float, ...] = DEFAULT_SUPPORT_GRID,
    confidence_grid: tuple[float, ...] = DEFAULT_CONFIDENCE_GRID,
) -> pd.DataFrame:
    """Execute the end-to-end Market Basket Analysis workflow sequentially."""
    pipeline_start = time.perf_counter()

    # Step 1: Fix all random seeds (GROUP_SEED = 5018)
    set_global_seed(seed)

    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2: Ensure dataset exists in data/ (or download automatically)
    dataset_path = download_dataset(data_dir=data_dir)

    # Step 3: Load raw data, remove cancelled invoices/invalid rows, and sample unique invoices
    sampled_df = load_and_sample_data(
        filepath=dataset_path,
        seed=seed,
        n_invoices=n_invoices,
    )

    # Step 4: Construct boolean invoice-by-item basket matrix
    basket_matrix = create_basket_matrix(sampled_df, sparse=True)

    # Step 5: Export top-30 single-item support baseline
    export_frequency_baseline(
        basket_matrix=basket_matrix,
        output_path=RESULTS_TABLES_DIR / "top_frequent_items.csv",
        top_n=30,
    )

    # Step 6: Sequentially benchmark Apriori and FP-Growth across threshold combinations
    benchmark_rows: list[dict[str, object]] = []
    reference_rules = pd.DataFrame()

    for min_support in support_grid:
        for min_confidence in confidence_grid:
            # Run Apriori with profiling
            apriori_result = run_apriori(
                basket_matrix=basket_matrix,
                min_support=min_support,
                metric="confidence",
                min_threshold=min_confidence,
                low_memory=False,
            )
            benchmark_rows.append(apriori_result["metrics"])

            # Run FP-Growth with profiling on the exact same basket matrix and thresholds
            fpgrowth_result = run_fpgrowth(
                basket_matrix=basket_matrix,
                min_support=min_support,
                metric="confidence",
                min_threshold=min_confidence,
            )
            benchmark_rows.append(fpgrowth_result["metrics"])

            # Save the rule set at reference thresholds (support=0.015, confidence=0.50) for export
            if np.isclose(min_support, 0.015) and np.isclose(min_confidence, 0.50):
                reference_rules = fpgrowth_result["rules"]
            elif reference_rules.empty and not fpgrowth_result["rules"].empty:
                reference_rules = fpgrowth_result["rules"]

    # Step 7: Export benchmark comparison table and top cross-selling rules to CSV
    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_csv = RESULTS_TABLES_DIR / "algorithm_benchmark.csv"
    benchmark_df.to_csv(benchmark_csv, index=False)
    LOGGER.info("Exported algorithm benchmark table to %s", benchmark_csv)

    top_rules_df = format_rules_table(reference_rules, top_n=50)
    top_rules_csv = RESULTS_TABLES_DIR / "top_association_rules.csv"
    top_rules_df.to_csv(top_rules_csv, index=False)
    LOGGER.info("Exported top association rules to %s", top_rules_csv)

    # Step 8: Generate benchmark plots
    save_benchmark_figures(
        benchmark_df=benchmark_df,
        reference_rules=reference_rules,
        figures_dir=RESULTS_FIGURES_DIR,
    )

    total_elapsed = time.perf_counter() - pipeline_start
    LOGGER.info(
        "Complete pipeline finished in %.2f seconds (seed=%d, sampled_invoices=%d).",
        total_elapsed,
        seed,
        basket_matrix.shape[0],
    )
    return benchmark_df


def parse_args() -> argparse.Namespace:
    """Parse CLI options for running the full pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete Group 6 Market Basket Analysis pipeline."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing (or receiving) Online Retail II data (default: data/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GROUP_SEED,
        help=f"Random seed for deterministic invoice sampling (default: {GROUP_SEED}).",
    )
    parser.add_argument(
        "--n-invoices",
        type=int,
        default=5000,
        help="Number of unique invoices to sample (default: 5000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    cli_args = parse_args()
    summary_df = run_pipeline(
        data_dir=cli_args.data_dir,
        seed=cli_args.seed,
        n_invoices=cli_args.n_invoices,
    )

    print("\n=== Benchmark Summary (Confidence >= 0.50) ===")
    ref_view = summary_df.loc[np.isclose(summary_df["min_threshold"], 0.50)]
    display_cols = [
        "algorithm",
        "min_support",
        "min_threshold",
        "frequent_itemsets",
        "rules_generated",
        "avg_lift",
        "avg_conviction",
        "execution_time_sec",
        "peak_memory_mb",
    ]
    print(ref_view[display_cols].to_string(index=False))
