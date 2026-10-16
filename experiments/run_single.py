from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gp.config import GPConfig, parse_cli_config
from gp.engine import run_evolution
from gp.stats import save_generation_stats


def main():
    parser = argparse.ArgumentParser(description="Run a single GP symbolic regression execution.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to base YAML config")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name override")
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    parser.add_argument("--exp_name", type=str, default="single_run", help="Experiment name / output folder")
    parser.add_argument("--out_dir", type=str, default="results", help="Base output directory")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")
    parser.add_argument("--verbose", action="store_true", help="Print generation progress")
    args = parser.parse_args()

    cfg = GPConfig.from_yaml(args.config) if Path(args.config).exists() else GPConfig()
    if args.dataset is not None:
        cfg.dataset = args.dataset
    if args.seed is not None:
        cfg.seed = args.seed
    if args.override:
        cfg.override(args.override)

    print(f"Running single execution for dataset='{cfg.dataset}', seed={cfg.seed}...")
    res = run_evolution(cfg, verbose=args.verbose)

    out_folder = Path(args.out_dir) / args.exp_name
    out_folder.mkdir(parents=True, exist_ok=True)

    # Save generation history
    history_file = out_folder / f"history_seed_{cfg.seed}.csv"
    save_generation_stats(res.history, history_file)

    # Save summary
    summary_data = {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "train_rmse": res.train_rmse,
        "test_rmse": res.test_rmse,
        "size": res.best_individual.size,
        "depth": res.best_individual.depth,
        "expression": res.best_individual.to_string(),
        "canonical": res.best_individual.canonical_repr(),
        "execution_time_sec": res.execution_time,
        "config": cfg.to_dict(),
    }
    summary_file = out_folder / f"summary_seed_{cfg.seed}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Execution complete in {res.execution_time:.2f}s!")
    print(f"Train RMSE: {res.train_rmse:.6f} | Test RMSE: {res.test_rmse:.6f}")
    print(f"Best Expression: {res.best_individual.to_string()}")
    print(f"Saved artifacts to: {out_folder}")


if __name__ == "__main__":
    main()
