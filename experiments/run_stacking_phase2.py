from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
from pathlib import Path
import sys
from typing import Dict, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_audit_phase2 import compare_paired, run_exp_fast
from gp.config import GPConfig


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
    print("PHASE 2 SEQUENTIAL STACKING LADDER (30 runs per step)")
    print("====================================================================================\n")

    for ds_name, norm in datasets:
        ds_label = "concrete1_norm" if ds_name == "concrete" else ds_name
        print(f"\n==================== STACKING LADDER: {ds_label} ====================")

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

        # Ladder A (Structural/Selective improvements without altering function set / scaling)
        # Step A1: S0 + H3 (Dedup) + H5 (Parsimony)
        cfg_a1 = copy.deepcopy(base_cfg)
        cfg_a1.dedup_offspring = True
        cfg_a1.parsimony_tiebreak = True
        run_exp_fast(cfg_a1, f"{ds_label}_StackA1_H3_dedup_H5_parsimony", out_dir, data_cache, n_runs, n_jobs)

        # Step A2: S0 + H3 (Dedup) + H5 (Parsimony) + H2 (Op Rebalance + Mixed Mut)
        cfg_a2 = copy.deepcopy(cfg_a1)
        cfg_a2.crossover_prob = 0.7
        cfg_a2.mutation_prob = 0.15
        cfg_a2.mutation_type = "mixed"
        run_exp_fast(cfg_a2, f"{ds_label}_StackA2_H3_H5_H2_ops", out_dir, data_cache, n_runs, n_jobs)

        # Ladder B (Full Numerical + Structural Stack: Linear Scaling + AQ + Dedup + Parsimony + Ops + Nonlinear)
        # Step B1: S0 + H6a (Linear Scaling) + H1 (Analytic Quotient)
        cfg_b1 = copy.deepcopy(base_cfg)
        cfg_b1.linear_scaling = True
        cfg_b1.function_set = ["add", "sub", "mul", "aq"]
        run_exp_fast(cfg_b1, f"{ds_label}_StackB1_H6a_scaling_H1_aq", out_dir, data_cache, n_runs, n_jobs)

        # Step B2: Step B1 + H3 (Dedup) + H5 (Parsimony)
        cfg_b2 = copy.deepcopy(cfg_b1)
        cfg_b2.dedup_offspring = True
        cfg_b2.parsimony_tiebreak = True
        run_exp_fast(cfg_b2, f"{ds_label}_StackB2_B1_plus_H3_H5", out_dir, data_cache, n_runs, n_jobs)

        # Step B3: Step B2 + H2 (Op Rebalance + Mixed Mut)
        cfg_b3 = copy.deepcopy(cfg_b2)
        cfg_b3.crossover_prob = 0.7
        cfg_b3.mutation_prob = 0.15
        cfg_b3.mutation_type = "mixed"
        run_exp_fast(cfg_b3, f"{ds_label}_StackB3_B2_plus_H2_ops", out_dir, data_cache, n_runs, n_jobs)

        # Step B4: Step B3 + H6b (Nonlinear primitives with AQ)
        cfg_b4 = copy.deepcopy(cfg_b3)
        cfg_b4.function_set = ["add", "sub", "mul", "aq", "log", "sqrt", "exp"]
        run_exp_fast(cfg_b4, f"{ds_label}_StackB4_Full_Nonlinear_AQ", out_dir, data_cache, n_runs, n_jobs)


if __name__ == "__main__":
    main()
