from __future__ import annotations

import numpy as np
from gp.config import GPConfig
from gp.engine import run_evolution
from gp.primitives import analytic_quotient
from gp.tree import Individual, Node


def test_analytic_quotient():
    a = np.array([1.0, 10.0, -5.0])
    b = np.array([0.0, 1e-12, 3.0])
    out = analytic_quotient(a, b)
    assert np.all(np.isfinite(out))
    assert np.isclose(out[0], 1.0)
    assert np.isclose(out[2], -5.0 / np.sqrt(10.0))


def test_phase2_mechanisms_integration():
    X_train = np.linspace(-2, 2, 40).reshape(-1, 2)
    y_train = 50.0 + 10.0 * (X_train[:, 0] ** 2 - X_train[:, 1])
    X_test = np.linspace(-3, 3, 20).reshape(-1, 2)
    y_test = 50.0 + 10.0 * (X_test[:, 0] ** 2 - X_test[:, 1])

    cfg = GPConfig(
        dataset="synth1",
        pop_size=40,
        generations=10,
        max_size=6,
        selection="lexicase",
        crossover_prob=0.7,
        mutation_prob=0.15,
        mutation_type="mixed",
        function_set=["add", "sub", "mul", "aq", "log", "sqrt", "exp"],
        dedup_offspring=True,
        n_islands=4,
        migration_interval=5,
        migration_size=2,
        parsimony_tiebreak=True,
        linear_scaling=True,
        seed=42,
    )
    res = run_evolution(cfg, data=(X_train, y_train, X_test, y_test))
    assert np.isfinite(res.train_rmse)
    assert np.isfinite(res.test_rmse)
    assert len(res.history) == 11
