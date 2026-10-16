from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np

from gp.tree import Individual

PENALTY_FITNESS: float = 1e12


def compute_case_errors(y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return vector of absolute errors per test case: |y_pred - y|."""
    if (
        not np.all(np.isfinite(y_pred))
        or np.any(np.abs(y_pred) > 1e100)
    ):
        return np.full(len(y), PENALTY_FITNESS, dtype=np.float64)

    errors = np.abs(y_pred - y)
    if not np.all(np.isfinite(errors)):
        return np.full(len(y), PENALTY_FITNESS, dtype=np.float64)
    return errors.astype(np.float64)


def rmse(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Compute Root Mean Squared Error with penalty on invalid values."""
    if (
        not np.all(np.isfinite(y_pred))
        or np.any(np.abs(y_pred) > 1e100)
    ):
        return PENALTY_FITNESS

    with np.errstate(over="ignore", invalid="ignore"):
        diff = y_pred - y
        mse = np.mean(diff ** 2)

    if not np.isfinite(mse) or mse > 1e100:
        return PENALTY_FITNESS

    val = float(np.sqrt(mse))
    return val if np.isfinite(val) else PENALTY_FITNESS


def evaluate_individual(
    ind: Individual,
    X: np.ndarray,
    y: np.ndarray,
    force_recompute: bool = False,
) -> Tuple[float, np.ndarray]:
    """Evaluate individual against X and y, utilizing caching if available."""
    if not force_recompute and ind.fitness is not None and ind.case_errors is not None:
        return ind.fitness, ind.case_errors

    try:
        y_pred = ind.root.evaluate(X)
        fit = rmse(y_pred, y)
        c_errs = compute_case_errors(y_pred, y)
    except Exception:
        fit = PENALTY_FITNESS
        c_errs = np.full(len(y), PENALTY_FITNESS, dtype=np.float64)

    ind.fitness = fit
    ind.case_errors = c_errs
    return fit, c_errs


def evaluate_population(
    population: Sequence[Individual],
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate entire population, returning (fitness_array, error_matrix).

    Returns:
        fitnesses: shape (pop_size,)
        error_matrix: shape (pop_size, n_cases)
    """
    n_pop = len(population)
    n_cases = len(y)
    error_matrix = np.empty((n_pop, n_cases), dtype=np.float64)
    fitnesses = np.empty(n_pop, dtype=np.float64)

    for i, ind in enumerate(population):
        fit, errs = evaluate_individual(ind, X, y)
        fitnesses[i] = fit
        error_matrix[i, :] = errs

    return fitnesses, error_matrix
