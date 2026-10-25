from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from gp.config import GPConfig
from gp.data import load_dataset
from gp.engine import run_evolution
from gp.stats import stats_to_dataframe


def _worker_task(args: Tuple[GPConfig, int, Tuple[Any, Any, Any, Any]]) -> Dict[str, Any]:
    base_cfg, seed, data = args
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


def run_exp_fast(
    config: GPConfig,
    exp_name: str,
    out_dir: Path,
    data_cache: Dict[Tuple[str, bool], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_runs: int = 30,
    n_jobs: int = 12,
    force: bool = False,
) -> pd.DataFrame:
    exp_path = out_dir / exp_name
    summary_path = exp_path / "summary.csv"
    history_path = exp_path / "history.csv"

    if summary_path.exists() and history_path.exists() and not force:
        print(f"[CACHE] {exp_name}")
        return pd.read_csv(summary_path)

    exp_path.mkdir(parents=True, exist_ok=True)
    config.to_yaml(exp_path / "config.yaml")

    key = (config.dataset, config.normalize_concrete)
    if key not in data_cache:
        data_cache[key] = load_dataset(config.dataset, normalize=config.normalize_concrete)
    data = data_cache[key]

    t0 = time.time()
    tasks = [(config, s, data) for s in range(n_runs)]
    summaries = []
    histories = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_worker_task, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            summaries.append(res["summary"])
            histories.append(res["history_df"])

    summaries.sort(key=lambda x: x["run"])
    histories.sort(key=lambda df: int(df["run"].iloc[0]))

    summary_df = pd.DataFrame(summaries)
    history_df = pd.concat(histories, ignore_index=True)

    summary_df.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)

    dt = time.time() - t0
    tr_m, tr_s = summary_df["train_rmse"].mean(), summary_df["train_rmse"].std()
    te_m, te_s = summary_df["test_rmse"].mean(), summary_df["test_rmse"].std()
    te_med = summary_df["test_rmse"].median()
    te_iqr = summary_df["test_rmse"].quantile(0.75) - summary_df["test_rmse"].quantile(0.25)
    sz_m = summary_df["size"].mean()
    print(
        f"[DONE {dt:5.1f}s] {exp_name:42s} | "
        f"Train: {tr_m:7.4f}±{tr_s:6.4f} | "
        f"Test Mean: {te_m:8.4f}±{te_s:8.4f} | "
        f"Test Med(IQR): {te_med:7.4f}({te_iqr:6.4f}) | "
        f"Size: {sz_m:5.1f}"
    )
    return summary_df


def compare_paired(df_base: pd.DataFrame, df_new: pd.DataFrame, metric: str = "test_rmse") -> Dict[str, Any]:
    a = df_base.sort_values("seed")[metric].to_numpy(dtype=np.float64)
    b = df_new.sort_values("seed")[metric].to_numpy(dtype=np.float64)

    _, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    try:
        if np.allclose(a, b):
            p_wilc = 1.0
        else:
            _, p_wilc = stats.wilcoxon(a, b, alternative="two-sided")
    except Exception:
        p_wilc = float("nan")

    wins = int(np.sum(b < a - 1e-9))
    losses = int(np.sum(b > a + 1e-9))
    ties = len(a) - wins - losses

    return {
        "base_mean": float(np.mean(a)),
        "base_std": float(np.std(a, ddof=1)),
        "base_median": float(np.median(a)),
        "base_iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "base_max": float(np.max(a)),
        "new_mean": float(np.mean(b)),
        "new_std": float(np.std(b, ddof=1)),
        "new_median": float(np.median(b)),
        "new_iqr": float(np.percentile(b, 75) - np.percentile(b, 25)),
        "new_max": float(np.max(b)),
        "p_wilcoxon": float(p_wilc),
        "p_mannwhitney": float(p_mw),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def main():
    out_dir = Path("results/phase2_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_cache: Dict[Tuple[str, bool], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    n_runs = 30
    n_jobs = min(12, max(1, (os.cpu_count() or 4) - 2))

    datasets = [
        ("synth1", False),
        ("synth2", False),
        ("concrete", True),
    ]

    print("====================================================================================")
    print(f"PHASE 2 AUDITABLE STUDY: 30 runs per config | {n_jobs} parallel workers")
    print("====================================================================================\n")

    for ds_name, norm in datasets:
        ds_label = "concrete1_norm" if ds_name == "concrete" else ds_name
        print(f"\n==================== DATASET: {ds_label} ====================")

        # Estado Atual (State 0): Epsilon-Lexicase with standard Phase 1 parameters
        base_cfg = GPConfig(
            dataset=ds_name,
            normalize_concrete=norm,
            pop_size=100,
            generations=100,
            max_size=7,
            size_mode="depth",
            selection="lexicase",
            lexicase_epsilon="mad",
            crossover_prob=0.9,
            mutation_prob=0.05,
            mutation_type="subtree",
            elitism=True,
            elite_size=1,
            function_set=["add", "sub", "mul", "div"],
        )
        run_exp_fast(base_cfg, f"{ds_label}_S0_estado_atual_lexicase", out_dir, data_cache, n_runs, n_jobs)

        # H1: Analytic Quotient (Polos/Falésias)
        cfg_h1 = copy.deepcopy(base_cfg)
        cfg_h1.function_set = ["add", "sub", "mul", "aq"]
        run_exp_fast(cfg_h1, f"{ds_label}_H1_analytic_quotient", out_dir, data_cache, n_runs, n_jobs)

        # H2: Rebalanceamento de Operadores + Mutação Mista (Rugosidade/Epistasia)
        cfg_h2 = copy.deepcopy(base_cfg)
        cfg_h2.crossover_prob = 0.7
        cfg_h2.mutation_prob = 0.15
        cfg_h2.mutation_type = "mixed"
        run_exp_fast(cfg_h2, f"{ds_label}_H2_op_rebalance_mixed_mut", out_dir, data_cache, n_runs, n_jobs)

        # H3: Rejeição de Duplicatas / Diversidade (Vales/Barreiras)
        cfg_h3 = copy.deepcopy(base_cfg)
        cfg_h3.dedup_offspring = True
        run_exp_fast(cfg_h3, f"{ds_label}_H3_dedup_diversity", out_dir, data_cache, n_runs, n_jobs)

        # H4: Modelo de Ilhas (Bacias de Atração Isoladas)
        cfg_h4 = copy.deepcopy(base_cfg)
        cfg_h4.n_islands = 4
        cfg_h4.migration_interval = 20
        cfg_h4.migration_size = 2
        run_exp_fast(cfg_h4, f"{ds_label}_H4_island_model", out_dir, data_cache, n_runs, n_jobs)

        # H5: Pressão de Parcimônia Lexicográfica (Redes Neutras / Bloat)
        cfg_h5 = copy.deepcopy(base_cfg)
        cfg_h5.parsimony_tiebreak = True
        run_exp_fast(cfg_h5, f"{ds_label}_H5_parsimony_pressure", out_dir, data_cache, n_runs, n_jobs)

        # H6a: Linear Scaling (Escala e Offset por OLS)
        cfg_h6a = copy.deepcopy(base_cfg)
        cfg_h6a.linear_scaling = True
        run_exp_fast(cfg_h6a, f"{ds_label}_H6a_linear_scaling", out_dir, data_cache, n_runs, n_jobs)

        # H6b: Primitivas Não-Lineares + Linear Scaling
        cfg_h6b = copy.deepcopy(base_cfg)
        cfg_h6b.linear_scaling = True
        cfg_h6b.function_set = ["add", "sub", "mul", "div", "log", "sqrt", "exp"]
        run_exp_fast(cfg_h6b, f"{ds_label}_H6b_nonlinear_plus_scaling", out_dir, data_cache, n_runs, n_jobs)


if __name__ == "__main__":
    main()
