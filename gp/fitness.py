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


def fit_linear_scaling(y_raw: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Fit optimal OLS linear scaling y ~ alpha + beta * y_raw (Keijzer, 2003)."""
    if not np.all(np.isfinite(y_raw)) or np.any(np.abs(y_raw) > 1e50):
        return 0.0, 1.0, y_raw

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mean_raw = float(np.mean(y_raw))
        mean_y = float(np.mean(y))
        diff_raw = y_raw - mean_raw
        var_raw = float(np.mean(diff_raw * diff_raw))

        if not np.isfinite(var_raw) or var_raw < 1e-12:
            alpha = mean_y
            beta = 0.0
        else:
            cov = float(np.mean(diff_raw * (y - mean_y)))
            beta = float(np.clip(cov / var_raw, -1e6, 1e6))
            alpha = float(np.clip(mean_y - beta * mean_raw, -1e6, 1e6))

        y_scaled = alpha + beta * y_raw
        if not np.all(np.isfinite(y_scaled)):
            return 0.0, 1.0, y_raw
        return alpha, beta, y_scaled


def evaluate_individual(
    ind: Individual,
    X: np.ndarray,
    y: np.ndarray,
    force_recompute: bool = False,
    linear_scaling: bool = False,
) -> Tuple[float, np.ndarray]:
    """Evaluate individual against X and y, utilizing caching if available."""
    if not force_recompute and ind.fitness is not None and ind.case_errors is not None:
        return ind.fitness, ind.case_errors

    try:
        y_pred = ind.root.evaluate(X)
        if linear_scaling:
            alpha, beta, y_pred = fit_linear_scaling(y_pred, y)
            ind.scale_offset = alpha
            ind.scale_slope = beta
        else:
            ind.scale_offset = 0.0
            ind.scale_slope = 1.0

        fit = rmse(y_pred, y)
        c_errs = compute_case_errors(y_pred, y)
    except Exception:
        fit = PENALTY_FITNESS
        c_errs = np.full(len(y), PENALTY_FITNESS, dtype=np.float64)
        ind.scale_offset = 0.0
        ind.scale_slope = 1.0

    ind.fitness = fit
    ind.case_errors = c_errs
    return fit, c_errs


def evaluate_population(
    population: Sequence[Individual],
    X: np.ndarray,
    y: np.ndarray,
    linear_scaling: bool = False,
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
        fit, errs = evaluate_individual(ind, X, y, linear_scaling=linear_scaling)
        fitnesses[i] = fit
        error_matrix[i, :] = errs

    return fitnesses, error_matrix
