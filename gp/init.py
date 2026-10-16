from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np

from gp.config import GPConfig
from gp.primitives import get_primitive
from gp.tree import Individual, Node


def random_terminal(
    n_features: int,
    const_range: Tuple[float, float],
    const_prob: float,
    rng: np.random.Generator,
) -> Node:
    """Generate a random terminal node (either variable or ERC constant)."""
    if rng.random() < const_prob or n_features == 0:
        val = rng.uniform(const_range[0], const_range[1])
        return Node(val)
    else:
        var_idx = rng.integers(0, n_features)
        return Node(f"x{var_idx}")


def random_function(
    function_set: Sequence[str],
    rng: np.random.Generator,
) -> str:
    """Choose a random function name from the configured function set."""
    idx = rng.integers(0, len(function_set))
    return function_set[idx]


def build_tree_grow(
    current_depth: int,
    max_depth: int,
    n_features: int,
    function_set: Sequence[str],
    const_range: Tuple[float, float],
    const_prob: float,
    rng: np.random.Generator,
) -> Node:
    """Recursively build a tree using the 'grow' method."""
    # If at max depth, must be a terminal
    if current_depth >= max_depth:
        return random_terminal(n_features, const_range, const_prob, rng)

    # At root (current_depth == 1), prefer a function if max_depth > 1
    if current_depth == 1 and max_depth > 1:
        choose_terminal = False
    else:
        # Uniformly choose between terminals and functions
        total_options = len(function_set) + 1  # 1 for terminal group
        choose_terminal = rng.random() < (1.0 / total_options)

    if choose_terminal:
        return random_terminal(n_features, const_range, const_prob, rng)

    fn_name = random_function(function_set, rng)
    prim = get_primitive(fn_name)
    children = [
        build_tree_grow(
            current_depth + 1,
            max_depth,
            n_features,
            function_set,
            const_range,
            const_prob,
            rng,
        )
        for _ in range(prim.arity)
    ]
    return Node(fn_name, children)


def build_tree_full(
    current_depth: int,
    max_depth: int,
    n_features: int,
    function_set: Sequence[str],
    const_range: Tuple[float, float],
    const_prob: float,
    rng: np.random.Generator,
) -> Node:
    """Recursively build a tree using the 'full' method."""
    if current_depth >= max_depth:
        return random_terminal(n_features, const_range, const_prob, rng)

    fn_name = random_function(function_set, rng)
    prim = get_primitive(fn_name)
    children = [
        build_tree_full(
            current_depth + 1,
            max_depth,
            n_features,
            function_set,
            const_range,
            const_prob,
            rng,
        )
        for _ in range(prim.arity)
    ]
    return Node(fn_name, children)


def is_valid_size(node: Node, max_size: int, size_mode: str) -> bool:
    """Check if tree satisfies the max_size constraint for the given mode."""
    if size_mode == "depth":
        return node.depth() <= max_size
    elif size_mode == "nodes":
        return node.size() <= max_size
    else:
        raise ValueError(f"Unknown size_mode: '{size_mode}'")


def generate_single_tree(
    method: str,
    target_depth: int,
    n_features: int,
    config: GPConfig,
    rng: np.random.Generator,
    max_attempts: int = 50,
) -> Node:
    """Generate a single tree satisfying size constraints."""
    for _ in range(max_attempts):
        if method == "full":
            tree = build_tree_full(
                1,
                target_depth,
                n_features,
                config.function_set,
                config.const_range,
                config.const_prob,
                rng,
            )
        else:
            tree = build_tree_grow(
                1,
                target_depth,
                n_features,
                config.function_set,
                config.const_range,
                config.const_prob,
                rng,
            )

        if is_valid_size(tree, config.max_size, config.size_mode):
            return tree

    # Fallback to single terminal if bounds could not be satisfied
    return random_terminal(n_features, config.const_range, config.const_prob, rng)


def initialize_population(
    config: GPConfig,
    n_features: int,
    rng: np.random.Generator,
) -> List[Individual]:
    """Generate initial population according to config (grow, full, ramped_half_and_half)."""
    pop: List[Individual] = []

    # Determine maximum depth allowed during initialization
    if config.size_mode == "depth":
        max_init_depth = config.max_size
    else:
        # In nodes mode, depth cannot exceed max_size nodes
        max_init_depth = min(config.max_size, 6)

    min_init_depth = max(1, config.init_min_depth)
    if min_init_depth > max_init_depth:
        min_init_depth = max_init_depth

    if config.init_method == "grow":
        for _ in range(config.pop_size):
            depth = rng.integers(min_init_depth, max_init_depth + 1)
            tree = generate_single_tree("grow", depth, n_features, config, rng)
            pop.append(Individual(tree, origin_operator="init"))

    elif config.init_method == "full":
        for _ in range(config.pop_size):
            depth = rng.integers(min_init_depth, max_init_depth + 1)
            tree = generate_single_tree("full", depth, n_features, config, rng)
            pop.append(Individual(tree, origin_operator="init"))

    elif config.init_method == "ramped_half_and_half":
        depth_range = list(range(min_init_depth, max_init_depth + 1))
        n_depths = len(depth_range)
        per_depth = config.pop_size // n_depths
        remainder = config.pop_size % n_depths

        for i, d in enumerate(depth_range):
            count_d = per_depth + (1 if i < remainder else 0)
            half = count_d // 2
            other_half = count_d - half

            for _ in range(half):
                tree = generate_single_tree("full", d, n_features, config, rng)
                pop.append(Individual(tree, origin_operator="init"))

            for _ in range(other_half):
                tree = generate_single_tree("grow", d, n_features, config, rng)
                pop.append(Individual(tree, origin_operator="init"))

    return pop
