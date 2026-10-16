from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np
import pandas as pd

from gp.tree import Individual


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    worst_fitness: float
    mean_fitness: float
    std_fitness: float
    n_unique: int
    n_repeated: int
    mean_size: float
    max_size_pop: int
    mean_depth: float
    size_best: int
    depth_best: int
    cx_better_than_mean_parents: int
    cx_worse_than_mean_parents: int
    cx_better_than_best_parent: int
    cx_worse_than_best_parent: int
    mut_better: int
    mut_worse: int


def collect_generation_stats(
    generation: int,
    population: Sequence[Individual],
    evaluated_offspring: Optional[Sequence[Individual]] = None,
) -> GenerationStats:
    """Compute and compile comprehensive metrics for the current generation."""
    fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
    sizes = [ind.size for ind in population]
    depths = [ind.depth for ind in population]
    canonicals = [ind.canonical_repr() for ind in population]

    best_idx = int(np.argmin(fitnesses))
    best_fit = float(fitnesses[best_idx])
    worst_fit = float(np.max(fitnesses))
    mean_fit = float(np.mean(fitnesses))
    std_fit = float(np.std(fitnesses))

    n_unique = len(set(canonicals))
    n_repeated = len(population) - n_unique

    mean_sz = float(np.mean(sizes))
    max_sz = int(np.max(sizes))
    mean_dp = float(np.mean(depths))
    sz_best = int(sizes[best_idx])
    dp_best = int(depths[best_idx])

    # Operator effect statistics
    cx_better_mean = 0
    cx_worse_mean = 0
    cx_better_best = 0
    cx_worse_best = 0
    mut_better = 0
    mut_worse = 0

    target_group = evaluated_offspring if evaluated_offspring is not None else []
    for ind in target_group:
        if ind.fitness is None or not np.isfinite(ind.fitness):
            continue

        if ind.origin_operator == "crossover" and len(ind.parent_fitness) >= 2:
            mean_p = float(np.mean(ind.parent_fitness))
            best_p = float(np.min(ind.parent_fitness))

            if ind.fitness < mean_p - 1e-9:
                cx_better_mean += 1
            elif ind.fitness > mean_p + 1e-9:
                cx_worse_mean += 1

            if ind.fitness < best_p - 1e-9:
                cx_better_best += 1
            elif ind.fitness > best_p + 1e-9:
                cx_worse_best += 1

        elif ind.origin_operator == "mutation" and len(ind.parent_fitness) >= 1:
            p_fit = float(ind.parent_fitness[0])
            if ind.fitness < p_fit - 1e-9:
                mut_better += 1
            elif ind.fitness > p_fit + 1e-9:
                mut_worse += 1

    return GenerationStats(
        generation=generation,
        best_fitness=best_fit,
        worst_fitness=worst_fit,
        mean_fitness=mean_fit,
        std_fitness=std_fit,
        n_unique=n_unique,
        n_repeated=n_repeated,
        mean_size=mean_sz,
        max_size_pop=max_sz,
        mean_depth=mean_dp,
        size_best=sz_best,
        depth_best=dp_best,
        cx_better_than_mean_parents=cx_better_mean,
        cx_worse_than_mean_parents=cx_worse_mean,
        cx_better_than_best_parent=cx_better_best,
        cx_worse_than_best_parent=cx_worse_best,
        mut_better=mut_better,
        mut_worse=mut_worse,
    )


def stats_to_dataframe(history: Sequence[GenerationStats]) -> pd.DataFrame:
    """Convert sequence of GenerationStats to a pandas DataFrame."""
    return pd.DataFrame([asdict(st) for st in history])


def save_generation_stats(history: Sequence[GenerationStats], path: Path) -> None:
    """Save generation history to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = stats_to_dataframe(history)
    df.to_csv(path, index=False)
