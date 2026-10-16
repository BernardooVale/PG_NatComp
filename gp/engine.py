from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from gp.config import GPConfig
from gp.data import load_dataset
from gp.fitness import evaluate_individual, evaluate_population, rmse
from gp.init import initialize_population
from gp.operators import mutate_individual, subtree_crossover
from gp.selection import select_parent
from gp.stats import GenerationStats, collect_generation_stats
from gp.tree import Individual


@dataclass
class EvolutionResult:
    config: GPConfig
    best_individual: Individual
    train_rmse: float
    test_rmse: float
    history: List[GenerationStats]
    execution_time: float


def run_evolution(
    config: GPConfig,
    data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    verbose: bool = False,
) -> EvolutionResult:
    """Run genetic programming symbolic regression according to config."""
    start_time = time.time()
    rng = np.random.default_rng(config.seed)

    if data is None:
        X_train, y_train, X_test, y_test = load_dataset(
            config.dataset,
            normalize=config.normalize_concrete,
        )
    else:
        X_train, y_train, X_test, y_test = data

    n_features = X_train.shape[1]

    # 1. Initialize population
    population = initialize_population(config, n_features, rng)

    # 2. Evaluate initial population
    _, error_matrix = evaluate_population(population, X_train, y_train)

    # 3. Collect generation 0 stats
    stats_0 = collect_generation_stats(0, population, evaluated_offspring=None)
    history: List[GenerationStats] = [stats_0]

    best_ind = min(population, key=lambda ind: ind.fitness if ind.fitness is not None else float("inf"))
    best_overall = best_ind.copy()

    if verbose:
        print(f"Gen 0: Best RMSE = {best_overall.fitness:.6f}, Mean = {stats_0.mean_fitness:.6f}")

    # 4. Generational evolutionary loop
    for gen in range(1, config.generations + 1):
        # Elitism
        elites: List[Individual] = []
        if config.elitism and config.elite_size > 0:
            sorted_pop = sorted(
                population,
                key=lambda ind: ind.fitness if ind.fitness is not None else float("inf"),
            )
            k_elites = min(config.elite_size, len(sorted_pop))
            elites = [ind.copy() for ind in sorted_pop[:k_elites]]

        # Generate offspring
        offspring: List[Individual] = []
        needed = config.pop_size - len(elites)

        while len(offspring) < needed:
            p1 = select_parent(population, config, rng, error_matrix)
            p2 = select_parent(population, config, rng, error_matrix)

            if rng.random() < config.crossover_prob:
                c1, c2 = subtree_crossover(p1, p2, config, rng)
            else:
                c1 = p1.copy()
                c1.origin_operator = "reproduction"
                c1.parent_fitness = [p1.fitness if p1.fitness is not None else float("inf")]
                c2 = p2.copy()
                c2.origin_operator = "reproduction"
                c2.parent_fitness = [p2.fitness if p2.fitness is not None else float("inf")]

            for c in (c1, c2):
                if rng.random() < config.mutation_prob:
                    # Capture current parent fitness before mutation
                    prior_fitness = c.fitness if c.fitness is not None else float("inf")
                    c = mutate_individual(c, n_features, config, rng)
                    c.parent_fitness = [prior_fitness]

                offspring.append(c)
                if len(offspring) == needed:
                    break

        # Evaluate offspring
        for ind in offspring:
            evaluate_individual(ind, X_train, y_train)

        # Form new population
        population = elites + offspring
        _, error_matrix = evaluate_population(population, X_train, y_train)

        # Collect stats
        gen_stats = collect_generation_stats(gen, population, evaluated_offspring=offspring)
        history.append(gen_stats)

        # Update best overall
        current_best = min(
            population,
            key=lambda ind: ind.fitness if ind.fitness is not None else float("inf"),
        )
        if (
            best_overall.fitness is None
            or (current_best.fitness is not None and current_best.fitness < best_overall.fitness)
        ):
            best_overall = current_best.copy()

        if verbose and (gen % 10 == 0 or gen == config.generations):
            print(f"Gen {gen}: Best RMSE = {best_overall.fitness:.6f}, Mean = {gen_stats.mean_fitness:.6f}")

    # Final evaluation of best individual on test set
    test_pred = best_overall.root.evaluate(X_test)
    test_err = rmse(test_pred, y_test)
    total_time = time.time() - start_time

    return EvolutionResult(
        config=config,
        best_individual=best_overall,
        train_rmse=float(best_overall.fitness) if best_overall.fitness is not None else float("inf"),
        test_rmse=test_err,
        history=history,
        execution_time=total_time,
    )
