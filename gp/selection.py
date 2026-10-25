from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np

from gp.config import GPConfig
from gp.tree import Individual


def tournament_selection(
    population: Sequence[Individual],
    tournament_size: int,
    rng: np.random.Generator,
    parsimony_tiebreak: bool = False,
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
        if parsimony_tiebreak and np.isfinite(fit) and np.isfinite(best_fitness):
            tol = 0.01 * max(1e-6, abs(best_fitness))
            if abs(fit - best_fitness) <= tol:
                if population[idx].size < population[best_idx].size:
                    best_fitness = fit
                    best_idx = idx
                continue
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


def _pick_parsimonious_candidate(
    population: Sequence[Individual],
    candidates: np.ndarray,
    rng: np.random.Generator,
) -> Individual:
    """Among surviving candidates within 5% of the best candidate's fitness, pick the smallest tree."""
    fits = np.array([
        population[i].fitness if population[i].fitness is not None else float("inf")
        for i in candidates
    ])
    min_fit = np.min(fits)
    if np.isfinite(min_fit):
        near_best_mask = fits <= min_fit * 1.05 + 1e-6
        viable = candidates[near_best_mask]
    else:
        viable = candidates

    sizes = np.array([population[i].size for i in viable])
    min_size = np.min(sizes)
    smallest = viable[sizes == min_size]
    if len(smallest) == 1:
        return population[smallest[0]]
    return population[rng.choice(smallest)]


def epsilon_lexicase_selection(
    population: Sequence[Individual],
    error_matrix: np.ndarray,
    epsilon_mode: str,
    rng: np.random.Generator,
    parsimony_tiebreak: bool = False,
) -> Individual:
    """Epsilon-Lexicase Selection (La Cava et al. / Spector)."""
    pop_size, n_cases = error_matrix.shape
    candidates = np.arange(pop_size)
    shuffled_cases = rng.permutation(n_cases)
    min_cases_for_parsimony = min(10, max(3, n_cases // 4))

    for step_idx, case_idx in enumerate(shuffled_cases):
        cand_errors = error_matrix[candidates, case_idx]
        e_star = np.min(cand_errors)

        if epsilon_mode == "mad":
            median_err = np.median(cand_errors)
            mad = np.median(np.abs(cand_errors - median_err))
            threshold = e_star + mad
            survivor_mask = cand_errors <= threshold
        else:
            survivor_mask = np.isclose(cand_errors, e_star, atol=1e-12, rtol=1e-9)

        if not np.any(survivor_mask):
            survivor_mask = cand_errors <= e_star

        candidates = candidates[survivor_mask]

        if len(candidates) == 1:
            return population[candidates[0]]

        if parsimony_tiebreak and step_idx + 1 >= min_cases_for_parsimony and len(candidates) <= 4:
            return _pick_parsimonious_candidate(population, candidates, rng)

    if parsimony_tiebreak and len(candidates) > 1:
        return _pick_parsimonious_candidate(population, candidates, rng)

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
        return tournament_selection(
            population,
            config.tournament_size,
            rng,
            parsimony_tiebreak=config.parsimony_tiebreak,
        )
    elif config.selection == "roulette":
        return roulette_selection(population, rng)
    elif config.selection == "lexicase":
        if error_matrix is None:
            raise ValueError("error_matrix must be provided when using lexicase selection")
        return epsilon_lexicase_selection(
            population,
            error_matrix,
            config.lexicase_epsilon,
            rng,
            parsimony_tiebreak=config.parsimony_tiebreak,
        )
    else:
        raise ValueError(f"Unknown selection method: {config.selection}")
