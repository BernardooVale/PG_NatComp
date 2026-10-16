from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np

from gp.config import GPConfig
from gp.tree import Individual


def tournament_selection(
    population: Sequence[Individual],
    tournament_size: int,
    rng: np.random.Generator,
) -> Individual:
    """Select the individual with minimum RMSE from a random tournament of size k."""
    pop_size = len(population)
    k = min(tournament_size, pop_size)
    sampled_indices = rng.choice(pop_size, size=k, replace=False)

    best_idx = sampled_indices[0]
    best_fitness = population[best_idx].fitness
    if best_fitness is None:
        best_fitness = float("inf")

    for idx in sampled_indices[1:]:
        fit = population[idx].fitness
        if fit is None:
            fit = float("inf")
        if fit < best_fitness:
            best_fitness = fit
            best_idx = idx

    return population[best_idx]


def roulette_selection(
    population: Sequence[Individual],
    rng: np.random.Generator,
) -> Individual:
    """Roulette wheel selection using fitness conversion 1 / (1 + RMSE).

    Handles penalized fitness with near-zero or zero probability.
    """
    pop_size = len(population)
    aptitudes = np.empty(pop_size, dtype=np.float64)

    for i, ind in enumerate(population):
        fit = ind.fitness
        if fit is None or not np.isfinite(fit) or fit >= 1e11:
            aptitudes[i] = 0.0
        else:
            aptitudes[i] = 1.0 / (1.0 + fit)

    total_apt = np.sum(aptitudes)
    if total_apt <= 0.0 or not np.isfinite(total_apt):
        probs = np.full(pop_size, 1.0 / pop_size, dtype=np.float64)
    else:
        probs = aptitudes / total_apt

    selected_idx = rng.choice(pop_size, p=probs)
    return population[selected_idx]


def epsilon_lexicase_selection(
    population: Sequence[Individual],
    error_matrix: np.ndarray,
    epsilon_mode: str,
    rng: np.random.Generator,
) -> Individual:
    """Epsilon-Lexicase Selection (La Cava et al. / Spector).

    1. Candidates = entire population.
    2. Shuffle test case indices.
    3. For each case t:
       - Find e*_t = min(errors of current candidates on case t).
       - In 'mad' mode: lambda = MAD = median(|e_t - median(e_t)|) across current candidates.
         Keep candidates with error <= e*_t + lambda.
       - In 'none' mode (standard Lexicase): Keep candidates with e_t == e*_t.
       - If only 1 candidate remains, return it.
    4. If cases exhaust with multiple candidates, return one uniformly at random.
    """
    pop_size, n_cases = error_matrix.shape
    candidates = np.arange(pop_size)
    shuffled_cases = rng.permutation(n_cases)

    for case_idx in shuffled_cases:
        cand_errors = error_matrix[candidates, case_idx]
        e_star = np.min(cand_errors)

        if epsilon_mode == "mad":
            median_err = np.median(cand_errors)
            mad = np.median(np.abs(cand_errors - median_err))
            threshold = e_star + mad
            survivor_mask = cand_errors <= threshold
        else:
            # Classic Lexicase: exact or numerical float equality
            survivor_mask = np.isclose(cand_errors, e_star, atol=1e-12, rtol=1e-9)

        # Safeguard: never eliminate all candidates
        if not np.any(survivor_mask):
            survivor_mask = cand_errors <= e_star

        candidates = candidates[survivor_mask]

        if len(candidates) == 1:
            return population[candidates[0]]

    # If multiple candidates remain after checking all cases, choose randomly
    chosen_idx = rng.choice(candidates)
    return population[chosen_idx]


def select_parent(
    population: Sequence[Individual],
    config: GPConfig,
    rng: np.random.Generator,
    error_matrix: Optional[np.ndarray] = None,
) -> Individual:
    """Select a single parent using the method configured in GPConfig."""
    if config.selection == "tournament":
        return tournament_selection(population, config.tournament_size, rng)
    elif config.selection == "roulette":
        return roulette_selection(population, rng)
    elif config.selection == "lexicase":
        if error_matrix is None:
            raise ValueError("error_matrix must be provided when using lexicase selection")
        return epsilon_lexicase_selection(population, error_matrix, config.lexicase_epsilon, rng)
    else:
        raise ValueError(f"Unknown selection method: {config.selection}")
