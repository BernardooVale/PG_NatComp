from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
import numpy as np

from gp.config import GPConfig
from gp.data import load_dataset
from gp.fitness import evaluate_individual, evaluate_population, rmse
from gp.init import initialize_population
from gp.operators import mutate_individual, point_mutation, subtree_crossover
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


def _evolve_subpopulation(
    subpop: List[Individual],
    sub_error_matrix: np.ndarray,
    target_size: int,
    n_features: int,
    config: GPConfig,
    rng: np.random.Generator,
    seen_canonicals: Optional[Set[str]] = None,
) -> Tuple[List[Individual], List[Individual]]:
    """Perform one generation of evolution (elitism + breeding) on a (sub)population."""
    elites: List[Individual] = []
    if config.elitism and config.elite_size > 0:
        sorted_pop = sorted(
            subpop,
            key=lambda ind: (
                ind.fitness if ind.fitness is not None else float("inf"),
                ind.size if config.parsimony_tiebreak else 0,
            ),
        )
        k_elites = min(config.elite_size, len(sorted_pop))
        elites = [ind.copy() for ind in sorted_pop[:k_elites]]
        if seen_canonicals is not None:
            for el in elites:
                seen_canonicals.add(el.canonical_repr())

    offspring: List[Individual] = []
    needed = target_size - len(elites)

    while len(offspring) < needed:
        p1 = select_parent(subpop, config, rng, sub_error_matrix)
        p2 = select_parent(subpop, config, rng, sub_error_matrix)

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
                prior_fitness = c.fitness if c.fitness is not None else float("inf")
                c = mutate_individual(c, n_features, config, rng)
                c.parent_fitness = [prior_fitness]

            if seen_canonicals is not None:
                canon = c.canonical_repr()
                attempts = 0
                while canon in seen_canonicals and attempts < 3:
                    prior_fit = c.fitness if c.fitness is not None else (
                        c.parent_fitness[0] if c.parent_fitness else float("inf")
                    )
                    c = mutate_individual(c, n_features, config, rng)
                    c.parent_fitness = [prior_fit]
                    canon = c.canonical_repr()
                    attempts += 1
                seen_canonicals.add(canon)

            offspring.append(c)
            if len(offspring) == needed:
                break

    return elites, offspring


def _best_key(ind: Individual, parsimony: bool = False) -> Tuple[float, int]:
    fit = ind.fitness if ind.fitness is not None else float("inf")
    sz = ind.size if parsimony else 0
    return (fit, sz)


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
    _, error_matrix = evaluate_population(
        population, X_train, y_train, linear_scaling=config.linear_scaling
    )

    # 3. Collect generation 0 stats
    stats_0 = collect_generation_stats(0, population, evaluated_offspring=None)
    history: List[GenerationStats] = [stats_0]

    best_ind = min(population, key=lambda ind: _best_key(ind, config.parsimony_tiebreak))
    best_overall = best_ind.copy()

    if verbose:
        print(f"Gen 0: Best RMSE = {best_overall.fitness:.6f}, Mean = {stats_0.mean_fitness:.6f}")

    # Partition into islands if n_islands > 1
    n_islands = max(1, config.n_islands)
    if n_islands > 1:
        islands: List[List[Individual]] = []
        base_island_size = config.pop_size // n_islands
        rem = config.pop_size % n_islands
        start_idx = 0
        for i in range(n_islands):
            sz = base_island_size + (1 if i < rem else 0)
            islands.append(population[start_idx : start_idx + sz])
            start_idx += sz

    # 4. Generational evolutionary loop
    for gen in range(1, config.generations + 1):
        seen_canonicals: Optional[Set[str]] = set() if config.dedup_offspring else None

        if n_islands == 1:
            elites, offspring = _evolve_subpopulation(
                population,
                error_matrix,
                config.pop_size,
                n_features,
                config,
                rng,
                seen_canonicals=seen_canonicals,
            )
            for ind in offspring:
                evaluate_individual(ind, X_train, y_train, linear_scaling=config.linear_scaling)

            population = elites + offspring
            _, error_matrix = evaluate_population(
                population, X_train, y_train, linear_scaling=config.linear_scaling
            )
            all_offspring = offspring
        else:
            all_offspring = []
            new_islands: List[List[Individual]] = []
            for island in islands:
                _, isl_err_matrix = evaluate_population(
                    island, X_train, y_train, linear_scaling=config.linear_scaling
                )
                isl_seen: Optional[Set[str]] = set() if config.dedup_offspring else None
                elites, offspring = _evolve_subpopulation(
                    island,
                    isl_err_matrix,
                    len(island),
                    n_features,
                    config,
                    rng,
                    seen_canonicals=isl_seen,
                )
                for ind in offspring:
                    evaluate_individual(ind, X_train, y_train, linear_scaling=config.linear_scaling)
                new_islands.append(elites + offspring)
                all_offspring.extend(offspring)

            islands = new_islands

            # Ring migration every migration_interval generations
            if config.migration_size > 0 and gen % config.migration_interval == 0 and gen < config.generations:
                migrants_per_island: List[List[Individual]] = []
                for island in islands:
                    sorted_isl = sorted(island, key=lambda ind: _best_key(ind, config.parsimony_tiebreak))
                    k_mig = min(config.migration_size, max(1, len(island) // 4))
                    migrants_per_island.append([ind.copy() for ind in sorted_isl[:k_mig]])

                for i in range(n_islands):
                    target_isl = islands[(i + 1) % n_islands]
                    incoming = migrants_per_island[i]
                    k_mig = len(incoming)
                    # Sort target island best-to-worst and replace worst k_mig
                    target_sorted = sorted(target_isl, key=lambda ind: _best_key(ind, config.parsimony_tiebreak))
                    islands[(i + 1) % n_islands] = target_sorted[:-k_mig] + incoming

            population = [ind for isl in islands for ind in isl]
            _, error_matrix = evaluate_population(
                population, X_train, y_train, linear_scaling=config.linear_scaling
            )

        # Collect stats
        gen_stats = collect_generation_stats(gen, population, evaluated_offspring=all_offspring)
        history.append(gen_stats)

        # Update best overall
        current_best = min(
            population,
            key=lambda ind: _best_key(ind, config.parsimony_tiebreak),
        )
        if (
            best_overall.fitness is None
            or (
                current_best.fitness is not None
                and _best_key(current_best, config.parsimony_tiebreak)
                < _best_key(best_overall, config.parsimony_tiebreak)
            )
        ):
            best_overall = current_best.copy()

        if verbose and (gen % 10 == 0 or gen == config.generations):
            print(f"Gen {gen}: Best RMSE = {best_overall.fitness:.6f}, Mean = {gen_stats.mean_fitness:.6f}")

    # Final evaluation of best individual on test set
    test_pred = best_overall.predict(X_test)
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
