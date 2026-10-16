import numpy as np
from gp.config import GPConfig
from gp.operators import mutate_individual, point_mutation, subtree_crossover, subtree_mutation
from gp.tree import Individual, Node


def make_sample_individual():
    # ((x0 + x1) * (x0 - 2.0))
    n1 = Node("add", [Node("x0"), Node("x1")])
    n2 = Node("sub", [Node("x0"), Node(2.0)])
    root = Node("mul", [n1, n2])
    return Individual(root, fitness=1.5)


def test_crossover_size_limits_and_parent_immutability():
    cfg = GPConfig(max_size=4, size_mode="depth")
    rng = np.random.default_rng(123)

    p1 = make_sample_individual()
    p2 = make_sample_individual()
    p1_repr_before = p1.canonical_repr()
    p2_repr_before = p2.canonical_repr()

    for _ in range(20):
        c1, c2 = subtree_crossover(p1, p2, cfg, rng)
        assert c1.depth <= cfg.max_size
        assert c2.depth <= cfg.max_size
        assert c1.origin_operator in ("crossover", "reproduction")
        assert len(c1.parent_fitness) >= 1

    # Parents must remain unchanged
    assert p1.canonical_repr() == p1_repr_before
    assert p2.canonical_repr() == p2_repr_before


def test_mutation_size_limits_and_immutability():
    cfg = GPConfig(max_size=5, size_mode="depth")
    rng = np.random.default_rng(456)

    parent = make_sample_individual()
    p_repr_before = parent.canonical_repr()

    for _ in range(20):
        mutant = subtree_mutation(parent, n_features=2, config=cfg, rng=rng)
        assert mutant.depth <= cfg.max_size
        assert mutant.origin_operator in ("mutation", "reproduction")

    assert parent.canonical_repr() == p_repr_before


def test_point_mutation():
    cfg = GPConfig(function_set=["add", "sub", "mul", "div"])
    rng = np.random.default_rng(789)
    parent = make_sample_individual()

    mutant = point_mutation(parent, n_features=2, config=cfg, rng=rng)
    assert mutant.depth == parent.depth
