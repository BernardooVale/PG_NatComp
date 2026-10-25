from __future__ import annotations

from typing import Callable, Dict, NamedTuple
import numpy as np


class Primitive(NamedTuple):
    name: str
    arity: int
    func: Callable[..., np.ndarray]


# Protected vectorized functions
def protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Protected division: returns 1.0 when |b| < 1e-6 (Koza standard)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < 1e-6, 1.0, a / b)


def analytic_quotient(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Analytic Quotient (Ni et al., 2013): a / sqrt(1 + b^2).

    Everywhere continuous and differentiable, eliminating vertical asymptotes (poles).
    """
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        b_safe = np.clip(b, -1e150, 1e150)
        return a / np.sqrt(1.0 + b_safe * b_safe)


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def sin(a: np.ndarray) -> np.ndarray:
    return np.sin(a)


def cos(a: np.ndarray) -> np.ndarray:
    return np.cos(a)


def exp(a: np.ndarray) -> np.ndarray:
    """Protected exp: clip input to [-50, 50] to prevent overflow."""
    clipped = np.clip(a, -50.0, 50.0)
    return np.exp(clipped)


def protected_log(a: np.ndarray) -> np.ndarray:
    """Protected log: log(|a| + 1e-6)."""
    return np.log(np.abs(a) + 1e-6)


def protected_sqrt(a: np.ndarray) -> np.ndarray:
    """Protected sqrt: sqrt(|a|)."""
    return np.sqrt(np.abs(a))


PRIMITIVES: Dict[str, Primitive] = {
    "add": Primitive("add", 2, add),
    "sub": Primitive("sub", 2, sub),
    "mul": Primitive("mul", 2, mul),
    "div": Primitive("div", 2, protected_div),
    "aq": Primitive("aq", 2, analytic_quotient),
    "sin": Primitive("sin", 1, sin),
    "cos": Primitive("cos", 1, cos),
    "exp": Primitive("exp", 1, exp),
    "log": Primitive("log", 1, protected_log),
    "sqrt": Primitive("sqrt", 1, protected_sqrt),
}


def get_primitive(name: str) -> Primitive:
    if name not in PRIMITIVES:
        raise ValueError(f"Unknown primitive '{name}'. Available: {list(PRIMITIVES.keys())}")
    return PRIMITIVES[name]
