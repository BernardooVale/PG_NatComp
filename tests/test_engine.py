import numpy as np
from gp.config import GPConfig
from gp.engine import run_evolution


def test_evolution_sanity_and_reproducibility():
    # Synthetic target: y = x0 * x0 + x1
    rng_data = np.random.default_rng(100)
    X = rng_data.uniform(-2.0, 2.0, size=(50, 2))
    y = X[:, 0] * X[:, 0] + X[:, 1]
    data = (X, y, X, y)

    cfg1 = GPConfig(
        dataset="synthetic_test",
        pop_size=100,
        generations=20,
        max_size=6,
        size_mode="depth",
        tournament_size=3,
        seed=42,
    )

    res1 = run_evolution(cfg1, data=data)
    # Check that best RMSE drops over generations
    initial_best = res1.history[0].best_fitness
    final_best = res1.train_rmse
    assert final_best < initial_best

    # Check reproducibility with identical seed
    cfg2 = GPConfig(
        dataset="synthetic_test",
        pop_size=100,
        generations=20,
        max_size=6,
        size_mode="depth",
        tournament_size=3,
        seed=42,
    )
    res2 = run_evolution(cfg2, data=data)

    assert np.isclose(res1.train_rmse, res2.train_rmse)
    assert np.isclose(res1.test_rmse, res2.test_rmse)
    assert res1.best_individual.canonical_repr() == res2.best_individual.canonical_repr()
