from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import numpy as np

from gp.primitives import PRIMITIVES, get_primitive


class Node:
    """Tree node representing either a function, a variable or a constant."""

    def __init__(
        self,
        value: Union[str, float, int],
        children: Optional[List[Node]] = None,
    ):
        self.value = value
        self.children: List[Node] = children if children is not None else []

    @property
    def is_terminal(self) -> bool:
        return len(self.children) == 0

    @property
    def is_constant(self) -> bool:
        return isinstance(self.value, (int, float)) and not isinstance(self.value, bool)

    @property
    def is_variable(self) -> bool:
        return isinstance(self.value, str) and self.value.startswith("x") and self.value[1:].isdigit()

    @property
    def is_function(self) -> bool:
        return len(self.children) > 0

    def size(self) -> int:
        """Total number of nodes in the subtree rooted at self."""
        return 1 + sum(child.size() for child in self.children)

    def depth(self) -> int:
        """Depth of subtree rooted at self (leaf node has depth 1)."""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def copy(self) -> Node:
        """Deep copy of the subtree."""
        return Node(self.value, [child.copy() for child in self.children])

    def to_string(self) -> str:
        """Human-readable infix string representation."""
        if self.is_constant:
            return f"{float(self.value):.4g}"
        if self.is_variable:
            return str(self.value)

        # Function node
        fn_str = str(self.value)
        if fn_str == "add":
            return f"({self.children[0].to_string()} + {self.children[1].to_string()})"
        elif fn_str == "sub":
            return f"({self.children[0].to_string()} - {self.children[1].to_string()})"
        elif fn_str == "mul":
            return f"({self.children[0].to_string()} * {self.children[1].to_string()})"
        elif fn_str == "div":
            return f"({self.children[0].to_string()} / {self.children[1].to_string()})"
        else:
            args = ", ".join(child.to_string() for child in self.children)
            return f"{fn_str}({args})"

    def canonical_repr(self) -> str:
        """Canonical string representation for hashing and counting duplicates."""
        if self.is_constant:
            return f"C({float(self.value):.5g})"
        if self.is_variable:
            return str(self.value)
        args = ",".join(child.canonical_repr() for child in self.children)
        return f"{self.value}({args})"

    def __repr__(self) -> str:
        return self.to_string()

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of tree over input matrix X (shape: [N, d]).

        Returns:
            np.ndarray of shape (N,)
        """
        N = X.shape[0]
        if self.is_constant:
            return np.full(N, float(self.value), dtype=np.float64)

        if self.is_variable:
            var_idx = int(str(self.value)[1:])
            return np.asarray(X[:, var_idx], dtype=np.float64)

        # Function node
        prim = get_primitive(str(self.value))
        child_evals = [child.evaluate(X) for child in self.children]
        out = prim.func(*child_evals)

        # Ensure 1D array matching length of X
        if np.isscalar(out):
            out = np.full(N, out, dtype=np.float64)
        else:
            out = np.asarray(out, dtype=np.float64)
            if out.shape != (N,):
                out = out.reshape(N)
        return out


def get_all_nodes(root: Node) -> List[Tuple[Node, Optional[Node], int]]:
    """Return all nodes as tuples of (node, parent, child_index_in_parent).

    For the root, parent is None and index is -1.
    """
    nodes: List[Tuple[Node, Optional[Node], int]] = []

    def _traverse(node: Node, parent: Optional[Node], idx: int):
        nodes.append((node, parent, idx))
        for i, child in enumerate(node.children):
            _traverse(child, node, i)

    _traverse(root, None, -1)
    return nodes


class Individual:
    """Individual in the GP population wrapping tree root, fitness cache, and origin metadata."""

    def __init__(
        self,
        root: Node,
        fitness: Optional[float] = None,
        case_errors: Optional[np.ndarray] = None,
        origin_operator: str = "init",
        parent_fitness: Optional[List[float]] = None,
        scale_offset: float = 0.0,
        scale_slope: float = 1.0,
    ):
        self.root: Node = root
        self.fitness: Optional[float] = fitness
        self.case_errors: Optional[np.ndarray] = case_errors
        self.origin_operator: str = origin_operator
        self.parent_fitness: List[float] = parent_fitness if parent_fitness is not None else []
        self.scale_offset: float = scale_offset
        self.scale_slope: float = scale_slope

    @property
    def size(self) -> int:
        return self.root.size()

    @property
    def depth(self) -> int:
        return self.root.depth()

    def canonical_repr(self) -> str:
        return self.root.canonical_repr()

    def to_string(self) -> str:
        base_str = self.root.to_string()
        if self.scale_offset != 0.0 or self.scale_slope != 1.0:
            return f"({self.scale_offset:.4g} + {self.scale_slope:.4g} * {base_str})"
        return base_str

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate tree and apply linear scaling if active."""
        raw = self.root.evaluate(X)
        if self.scale_offset != 0.0 or self.scale_slope != 1.0:
            with np.errstate(over="ignore", invalid="ignore"):
                return self.scale_offset + self.scale_slope * raw
        return raw

    def copy(self) -> Individual:
        return Individual(
            root=self.root.copy(),
            fitness=self.fitness,
            case_errors=self.case_errors.copy() if self.case_errors is not None else None,
            origin_operator=self.origin_operator,
            parent_fitness=list(self.parent_fitness),
            scale_offset=self.scale_offset,
            scale_slope=self.scale_slope,
        )

    def invalidate_cache(self) -> None:
        self.fitness = None
        self.case_errors = None
        self.scale_offset = 0.0
        self.scale_slope = 1.0
