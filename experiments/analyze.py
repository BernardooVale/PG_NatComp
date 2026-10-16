from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Ensure non-interactive backend for headless plotting
plt.switch_backend("Agg")


def load_experiment_data(exp_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load summary.csv and history.csv from an experiment directory."""
    summary_path = exp_dir / "summary.csv"
    history_path = exp_dir / "history.csv"

    summary_df = pd.read_csv(summary_path) if summary_path.exists() else None
    history_df = pd.read_csv(history_path) if history_path.exists() else None
    return summary_df, history_df


def generate_experiment_tables(
    exp_dirs: List[Path],
    out_dir: Path,
) -> Tuple[pd.DataFrame, str]:
    """Generate consolidated CSV and Markdown tables reporting mean +/- std for all experiments."""
    rows = []
    for ed in exp_dirs:
        summary_df, _ = load_experiment_data(ed)
        if summary_df is None or len(summary_df) == 0:
            continue

        name = ed.name
        n = len(summary_df)
        train_mean = summary_df["train_rmse"].mean()
        train_std = summary_df["train_rmse"].std()
        test_mean = summary_df["test_rmse"].mean()
        test_std = summary_df["test_rmse"].std()
        size_mean = summary_df["size"].mean()
        size_std = summary_df["size"].std()
        time_mean = summary_df["time_sec"].mean()

        rows.append({
            "Experiment": name,
            "Runs": n,
            "Train RMSE (Mean +/- Std)": f"{train_mean:.4f} +/- {train_std:.4f}",
            "Test RMSE (Mean +/- Std)": f"{test_mean:.4f} +/- {test_std:.4f}",
            "Size (Mean +/- Std)": f"{size_mean:.2f} +/- {size_std:.2f}",
            "Avg Time (s)": f"{time_mean:.2f}",
            "train_mean_raw": train_mean,
            "train_std_raw": train_std,
            "test_mean_raw": test_mean,
            "test_std_raw": test_std,
            "size_mean_raw": size_mean,
            "size_std_raw": size_std,
            "time_mean_raw": time_mean,
        })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "experiments_consolidated.csv", index=False)

    display_cols = [
        "Experiment",
        "Runs",
        "Train RMSE (Mean +/- Std)",
        "Test RMSE (Mean +/- Std)",
        "Size (Mean +/- Std)",
        "Avg Time (s)",
    ]
    try:
        md_table = df[display_cols].to_markdown(index=False)
    except Exception:
        header = "| " + " | ".join(display_cols) + " |"
        sep = "| " + " | ".join(["---"] * len(display_cols)) + " |"
        rows_str = []
        for _, r in df[display_cols].iterrows():
            rows_str.append("| " + " | ".join(str(r[c]) for c in display_cols) + " |")
        md_table = "\n".join([header, sep] + rows_str)

    with open(out_dir / "experiments_consolidated.md", "w", encoding="utf-8") as f:
        f.write(md_table)

    return df, md_table


def plot_convergence_curves(
    experiments: Dict[str, Path],
    out_path: Path,
    title: str = "Convergence Curves (Best Training RMSE)",
) -> None:
    """Plot mean best training RMSE across generations with std-dev shading."""
    plt.figure(figsize=(9, 5))

    for label, exp_dir in experiments.items():
        _, history_df = load_experiment_data(exp_dir)
        if history_df is None or "generation" not in history_df.columns:
            continue

        grouped = history_df.groupby("generation")["best_fitness"].agg(["mean", "std"]).reset_index()
        gens = grouped["generation"]
        mean_val = grouped["mean"]
        std_val = grouped["std"].fillna(0.0)

        plt.plot(gens, mean_val, label=label, linewidth=1.8)
        plt.fill_between(gens, np.maximum(0, mean_val - std_val), mean_val + std_val, alpha=0.15)

    plt.xlabel("Generation")
    plt.ylabel("Train RMSE")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_boxplots(
    experiments: Dict[str, Path],
    out_path: Path,
    metric: str = "test_rmse",
    title: str = "Test RMSE Distribution Across Configurations",
) -> None:
    """Plot boxplots of train or test RMSE for compared configurations."""
    plt.figure(figsize=(10, 6))

    labels = []
    data_list = []
    for label, exp_dir in experiments.items():
        summary_df, _ = load_experiment_data(exp_dir)
        if summary_df is not None and metric in summary_df.columns:
            labels.append(label)
            data_list.append(summary_df[metric].values)

    if not data_list:
        plt.close()
        return

    plt.boxplot(data_list, tick_labels=labels, showmeans=True)
    plt.ylabel("RMSE")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_selection_comparison(
    selection_experiments: Dict[str, Path],
    figures_dir: Path,
) -> None:
    """Detailed visual comparison between selection methods: convergence, test error, and diversity."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convergence
    plot_convergence_curves(
        selection_experiments,
        figures_dir / "selection_convergence.png",
        title="Selection Methods: Training Convergence",
    )

    # 2. Test RMSE Boxplot
    plot_boxplots(
        selection_experiments,
        figures_dir / "selection_test_rmse_boxplot.png",
        metric="test_rmse",
        title="Selection Methods: Test RMSE Distribution",
    )

    # 3. Diversity (n_repeated)
    plt.figure(figsize=(9, 5))
    for label, exp_dir in selection_experiments.items():
        _, history_df = load_experiment_data(exp_dir)
        if history_df is not None and "n_repeated" in history_df.columns:
            grouped = history_df.groupby("generation")["n_repeated"].mean()
            plt.plot(grouped.index, grouped.values, label=label, linewidth=1.8)

    plt.xlabel("Generation")
    plt.ylabel("Mean Number of Repeated Individuals")
    plt.title("Population Convergence / Diversity Loss (n_repeated)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "selection_diversity_n_repeated.png", dpi=300)
    plt.close()


def plot_operator_dynamics(
    exp_dir: Path,
    out_path: Path,
    title: str = "Operator Dynamics: Beneficial vs Deleterious Offspring",
) -> None:
    """Plot frequency of offspring better/worse than parents over generations."""
    _, history_df = load_experiment_data(exp_dir)
    if history_df is None:
        return

    # Filter out gen 0 (no offspring)
    h_sub = history_df[history_df["generation"] > 0]
    grouped = h_sub.groupby("generation")[
        ["cx_better_than_mean_parents", "cx_worse_than_mean_parents", "mut_better", "mut_worse"]
    ].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(grouped.index, grouped["cx_better_than_mean_parents"], label="Crossover: Better than Mean Parents", color="green")
    plt.plot(grouped.index, grouped["cx_worse_than_mean_parents"], label="Crossover: Worse than Mean Parents", color="red", linestyle="--")
    plt.plot(grouped.index, grouped["mut_better"], label="Mutation: Better than Parent", color="blue")
    plt.plot(grouped.index, grouped["mut_worse"], label="Mutation: Worse than Parent", color="orange", linestyle="--")

    plt.xlabel("Generation")
    plt.ylabel("Average Number of Offspring")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_bloating_curves(
    experiments: Dict[str, Path],
    out_path: Path,
    title: str = "Bloating Analysis: Population Mean Size vs Generation",
) -> None:
    """Plot average tree size over generations to inspect code bloat."""
    plt.figure(figsize=(9, 5))

    for label, exp_dir in experiments.items():
        _, history_df = load_experiment_data(exp_dir)
        if history_df is not None and "mean_size" in history_df.columns:
            grouped = history_df.groupby("generation")["mean_size"].mean()
            plt.plot(grouped.index, grouped.values, label=label, linewidth=1.8)

    plt.xlabel("Generation")
    plt.ylabel("Mean Node Count")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def compute_statistical_tests(
    exp_a_dir: Path,
    exp_b_dir: Path,
    metric: str = "test_rmse",
) -> Dict[str, float]:
    """Perform two-sided Mann-Whitney U test between two experiment runs."""
    sum_a, _ = load_experiment_data(exp_a_dir)
    sum_b, _ = load_experiment_data(exp_b_dir)

    if sum_a is None or sum_b is None:
        return {"u_stat": float("nan"), "p_value": float("nan")}

    vals_a = sum_a[metric].dropna().values
    vals_b = sum_b[metric].dropna().values

    stat, pval = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    return {"u_stat": float(stat), "p_value": float(pval)}


def get_best_and_median_expressions(exp_dir: Path) -> Dict[str, Any]:
    """Extract best (lowest test RMSE) and median expressions from an experiment."""
    summary_df, _ = load_experiment_data(exp_dir)
    if summary_df is None or len(summary_df) == 0:
        return {}

    sorted_df = summary_df.sort_values(by="test_rmse").reset_index(drop=True)
    best_row = sorted_df.iloc[0]
    median_idx = len(sorted_df) // 2
    median_row = sorted_df.iloc[median_idx]

    return {
        "best": {
            "seed": int(best_row["seed"]),
            "train_rmse": float(best_row["train_rmse"]),
            "test_rmse": float(best_row["test_rmse"]),
            "size": int(best_row["size"]),
            "expression": str(best_row["expression"]),
        },
        "median": {
            "seed": int(median_row["seed"]),
            "train_rmse": float(median_row["train_rmse"]),
            "test_rmse": float(median_row["test_rmse"]),
            "size": int(median_row["size"]),
            "expression": str(median_row["expression"]),
        },
    }
