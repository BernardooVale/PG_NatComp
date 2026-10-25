from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np

from gp.config import GPConfig
from gp.init import build_tree_grow, is_valid_size, random_terminal
from gp.primitives import get_primitive
from gp.tree import Individual, Node, get_all_nodes


def select_crossover_node(
    root: Node,
    rng: np.random.Generator,
    internal_prob: float = 0.9,
) -> Tuple[Node, Optional[Node], int]:
    """Select a node for crossover, optionally biasing toward internal nodes."""
    all_nodes = get_all_nodes(root)
    internals = [n for n in all_nodes if n[0].is_function]
    terminals = [n for n in all_nodes if n[0].is_terminal]

    if internals and rng.random() < internal_prob:
        idx = rng.integers(0, len(internals))
        return internals[idx]
    elif terminals:
        idx = rng.integers(0, len(terminals))
        return terminals[idx]
    else:
        idx = rng.integers(0, len(all_nodes))
        return all_nodes[idx]


def replace_node_in_tree(
    root: Node,
    target_node: Node,
    parent_node: Optional[Node],
    child_idx: int,
    new_sub: Node,
) -> Node:
    """Replace target_node with new_sub. Returns new tree root."""
    if parent_node is None:
        return new_sub
    parent_node.children[child_idx] = new_sub
    return root


def subtree_crossover(
    parent1: Individual,
    parent2: Individual,
    config: GPConfig,
    rng: np.random.Generator,
    max_attempts: int = 20,
) -> Tuple[Individual, Individual]:
    """Perform subtree crossover, returning two offspring with metadata."""
    p1_fit = parent1.fitness if parent1.fitness is not None else float("inf")
    p2_fit = parent2.fitness if parent2.fitness is not None else float("inf")
    parents_fitness = [p1_fit, p2_fit]

    for _ in range(max_attempts):
        t1 = parent1.root.copy()
        t2 = parent2.root.copy()

        node1, pnode1, idx1 = select_crossover_node(t1, rng)
        node2, pnode2, idx2 = select_crossover_node(t2, rng)

        sub1 = node1.copy()
        sub2 = node2.copy()

        child1_root = replace_node_in_tree(t1, node1, pnode1, idx1, sub2)
        child2_root = replace_node_in_tree(t2, node2, pnode2, idx2, sub1)

        v1 = is_valid_size(child1_root, config.max_size, config.size_mode)
        v2 = is_valid_size(child2_root, config.max_size, config.size_mode)

        if v1 and v2:
            c1 = Individual(child1_root, origin_operator="crossover", parent_fitness=parents_fitness)
            c2 = Individual(child2_root, origin_operator="crossover", parent_fitness=parents_fitness)
            return c1, c2

    # Fallback to copies of parents if constraints couldn't be satisfied
    c1 = parent1.copy()
    c1.origin_operator = "reproduction"
    c1.parent_fitness = [p1_fit]
    c2 = parent2.copy()
    c2.origin_operator = "reproduction"
    c2.parent_fitness = [p2_fit]
    return c1, c2


def subtree_mutation(
    parent: Individual,
    n_features: int,
    config: GPConfig,
    rng: np.random.Generator,
    max_attempts: int = 20,
) -> Individual:
    """Perform subtree mutation by replacing a random node with a newly generated subtree."""
    p_fit = parent.fitness if parent.fitness is not None else float("inf")

    for _ in range(max_attempts):
        t = parent.root.copy()
        all_nodes = get_all_nodes(t)
        node, pnode, idx = all_nodes[rng.integers(0, len(all_nodes))]

        # Choose mutation subtree depth
        if config.size_mode == "depth":
            mut_depth = rng.integers(1, max(2, config.max_size // 2 + 1))
        else:
            mut_depth = rng.integers(1, 4)

        new_sub = build_tree_grow(
            1,
            mut_depth,
            n_features,
            config.function_set,
            config.const_range,
            config.const_prob,
            rng,
        )

        child_root = replace_node_in_tree(t, node, pnode, idx, new_sub)

        if is_valid_size(child_root, config.max_size, config.size_mode):
            return Individual(child_root, origin_operator="mutation", parent_fitness=[p_fit])

    # Fallback: copy parent
    child = parent.copy()
    child.origin_operator = "reproduction"
    child.parent_fitness = [p_fit]
    return child


def point_mutation(
    parent: Individual,
    n_features: int,
    config: GPConfig,
    rng: np.random.Generator,
) -> Individual:
    """Perform point mutation: substitute a node's primitive or value without changing arity."""
    p_fit = parent.fitness if parent.fitness is not None else float("inf")
    t = parent.root.copy()
    all_nodes = get_all_nodes(t)
    node, _, _ = all_nodes[rng.integers(0, len(all_nodes))]

    if node.is_function:
        current_prim = get_primitive(str(node.value))
        same_arity = [
            f for f in config.function_set
            if get_primitive(f).arity == current_prim.arity and f != node.value
        ]
        if same_arity:
            node.value = same_arity[rng.integers(0, len(same_arity))]
    elif node.is_constant:
        # Perturb constant or replace with new ERC
        if rng.random() < 0.5:
            node.value = float(node.value) + float(rng.normal(0.0, 0.2))
        else:
            node.value = float(rng.uniform(config.const_range[0], config.const_range[1]))
    elif node.is_variable:
        # Switch to another variable or constant
        term = random_terminal(n_features, config.const_range, config.const_prob, rng)
        node.value = term.value

    return Individual(t, origin_operator="mutation", parent_fitness=[p_fit])


def mixed_mutation(
    parent: Individual,
    n_features: int,
    config: GPConfig,
    rng: np.random.Generator,
) -> Individual:
    """Mixed mutation (H2): combines targeted ERC Gaussian tuning, point mutation, and subtree mutation."""
    p_fit = parent.fitness if parent.fitness is not None else float("inf")
    roll = rng.random()

    if roll < 0.35:
        # Targeted ERC Gaussian perturbation if tree contains constants
        t = parent.root.copy()
        all_nodes = get_all_nodes(t)
        const_nodes = [n[0] for n in all_nodes if n[0].is_constant]
        if const_nodes:
            target_c = const_nodes[rng.integers(0, len(const_nodes))]
            sigma = 0.25 * max(1.0, abs(float(target_c.value)))
            target_c.value = float(target_c.value) + float(rng.normal(0.0, sigma))
            return Individual(t, origin_operator="mutation", parent_fitness=[p_fit])
        return point_mutation(parent, n_features, config, rng)
    elif roll < 0.70:
        return point_mutation(parent, n_features, config, rng)
    else:
        return subtree_mutation(parent, n_features, config, rng)


def mutate_individual(
    ind: Individual,
    n_features: int,
    config: GPConfig,
    rng: np.random.Generator,
) -> Individual:
    """Apply mutation based on configured mutation_type."""
    if config.mutation_type == "point":
        return point_mutation(ind, n_features, config, rng)
    elif config.mutation_type == "mixed":
        return mixed_mutation(ind, n_features, config, rng)
    return subtree_mutation(ind, n_features, config, rng)
