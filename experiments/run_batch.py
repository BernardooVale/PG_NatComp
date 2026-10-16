from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from gp.config import GPConfig
from gp.data import load_dataset
from gp.engine import run_evolution
from gp.stats import stats_to_dataframe


def _worker_run(task_args: Tuple[GPConfig, int, Tuple[Any, Any, Any, Any]]) -> Dict[str, Any]:
    """Worker function for executing a single run in parallel."""
    base_cfg, seed, data = task_args
    cfg = copy.deepcopy(base_cfg)
    cfg.seed = seed

    res = run_evolution(cfg, data=data, verbose=False)
    history_df = stats_to_dataframe(res.history)
    history_df["run"] = seed
    history_df["seed"] = seed

    summary_row = {
        "run": seed,
        "seed": seed,
        "dataset": cfg.dataset,
        "train_rmse": res.train_rmse,
        "test_rmse": res.test_rmse,
        "size": res.best_individual.size,
        "depth": res.best_individual.depth,
        "expression": res.best_individual.to_string(),
        "canonical": res.best_individual.canonical_repr(),
        "time_sec": res.execution_time,
    }

    return {
        "seed": seed,
        "summary": summary_row,
        "history_df": history_df,
    }


def run_batch(
    config: GPConfig,
    n_runs: Optional[int] = None,
    exp_name: str = "batch_experiment",
    out_dir: str = "results",
    n_jobs: Optional[int] = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute n_runs repetitions of GP evolution in parallel and consolidate results."""
    out_path = Path(out_dir) / exp_name
    summary_path = out_path / "summary.csv"
    history_path = out_path / "history.csv"

    if summary_path.exists() and history_path.exists() and not force:
        print(f"Results for '{exp_name}' already exist. Loading cached results.")
        summary_df = pd.read_csv(summary_path)
        history_df = pd.read_csv(history_path)
        return summary_df, history_df

    out_path.mkdir(parents=True, exist_ok=True)
    config.to_yaml(out_path / "config.yaml")

    runs = n_runs if n_runs is not None else config.n_runs
    jobs = n_jobs if n_jobs is not None else config.n_jobs
    if jobs is None or jobs <= 0 or jobs > 4:
        jobs = min(4, max(1, os.cpu_count() or 1))

    print(f"Loading dataset '{config.dataset}'...")
    data = load_dataset(config.dataset, normalize=config.normalize_concrete)

    print(f"Starting batch '{exp_name}': {runs} runs with {jobs} parallel workers...")
    start_time = time.time()

    tasks = [(config, seed, data) for seed in range(runs)]
    summaries: List[Dict[str, Any]] = []
    histories: List[pd.DataFrame] = []

    if jobs == 1 or runs == 1:
        for t in tqdm(tasks, desc=exp_name):
            res = _worker_run(t)
            summaries.append(res["summary"])
            histories.append(res["history_df"])
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_worker_run, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=exp_name):
                res = fut.result()
                summaries.append(res["summary"])
                histories.append(res["history_df"])

    # Sort by run/seed for consistency
    summaries.sort(key=lambda x: x["run"])
    histories.sort(key=lambda df: int(df["run"].iloc[0]))

    summary_df = pd.DataFrame(summaries)
    history_df = pd.concat(histories, ignore_index=True)

    summary_df.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)

    elapsed = time.time() - start_time
    print(f"Batch '{exp_name}' finished in {elapsed:.2f}s.")
    print(
        f"Train RMSE: {summary_df['train_rmse'].mean():.4f} +/- {summary_df['train_rmse'].std():.4f} | "
        f"Test RMSE: {summary_df['test_rmse'].mean():.4f} +/- {summary_df['test_rmse'].std():.4f}"
    )

    return summary_df, history_df


def main():
    parser = argparse.ArgumentParser(description="Run batch GP experiments across seeds.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to base YAML config")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name override")
    parser.add_argument("--runs", type=int, default=30, help="Number of repetitions (seeds)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel worker count (-1 for all CPUs)")
    parser.add_argument("--exp_name", type=str, default="batch_run", help="Experiment name")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")
    args = parser.parse_args()

    cfg = GPConfig.from_yaml(args.config) if Path(args.config).exists() else GPConfig()
    if args.dataset is not None:
        cfg.dataset = args.dataset
    if args.runs is not None:
        cfg.n_runs = args.runs
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs
    if args.override:
        cfg.override(args.override)

    run_batch(
        config=cfg,
        n_runs=args.runs,
        exp_name=args.exp_name,
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
        force=args.force,
    )


if __name__ == "__main__":
    main()
