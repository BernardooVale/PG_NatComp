import numpy as np
from gp.config import GPConfig
from gp.init import initialize_population


def test_initialization_depth_mode():
    cfg = GPConfig(pop_size=50, max_size=5, size_mode="depth", init_method="ramped_half_and_half")
    rng = np.random.default_rng(42)
    pop = initialize_population(cfg, n_features=3, rng=rng)

    assert len(pop) == 50
    for ind in pop:
        assert ind.depth <= 5
        assert ind.depth >= 1


def test_initialization_nodes_mode():
    cfg = GPConfig(pop_size=50, max_size=7, size_mode="nodes", init_method="grow")
    rng = np.random.default_rng(42)
    pop = initialize_population(cfg, n_features=3, rng=rng)

    assert len(pop) == 50
    for ind in pop:
        assert ind.size <= 7
        assert ind.size >= 1


def test_full_method_distribution():
    cfg = GPConfig(pop_size=30, max_size=4, size_mode="depth", init_method="full")
    rng = np.random.default_rng(42)
    pop = initialize_population(cfg, n_features=2, rng=rng)

    for ind in pop:
        assert ind.depth <= 4
