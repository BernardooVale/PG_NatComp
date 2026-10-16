from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union
import yaml


@dataclass
class GPConfig:
    dataset: str = "synth1"
    pop_size: int = 100
    generations: int = 100
    max_size: int = 7
    size_mode: str = "depth"  # 'depth' or 'nodes'
    init_method: str = "ramped_half_and_half"  # 'grow', 'full', 'ramped_half_and_half'
    init_min_depth: int = 1
    selection: str = "tournament"  # 'roulette', 'tournament', 'lexicase'
    tournament_size: int = 2
    lexicase_epsilon: str = "mad"  # 'mad' or 'none'
    crossover_prob: float = 0.9
    mutation_prob: float = 0.05
    mutation_type: str = "subtree"  # 'subtree' or 'point'
    elitism: bool = True
    elite_size: int = 1
    function_set: List[str] = field(default_factory=lambda: ["add", "sub", "mul", "div"])
    const_range: Tuple[float, float] = (-1.0, 1.0)
    const_prob: float = 0.3
    normalize_concrete: bool = False
    seed: int = 0
    n_runs: int = 30
    n_jobs: int = -1

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.pop_size <= 0:
            raise ValueError(f"pop_size must be positive, got {self.pop_size}")
        if self.generations <= 0:
            raise ValueError(f"generations must be positive, got {self.generations}")
        if self.max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {self.max_size}")
        if self.size_mode not in ("depth", "nodes"):
            raise ValueError(f"size_mode must be 'depth' or 'nodes', got {self.size_mode}")
        if self.init_method not in ("grow", "full", "ramped_half_and_half"):
            raise ValueError(f"init_method must be 'grow', 'full' or 'ramped_half_and_half', got {self.init_method}")
        if self.init_min_depth < 1:
            raise ValueError(f"init_min_depth must be >= 1, got {self.init_min_depth}")
        if self.size_mode == "depth" and self.init_min_depth > self.max_size:
            raise ValueError(f"init_min_depth ({self.init_min_depth}) cannot be greater than max_size ({self.max_size})")
        if self.selection not in ("tournament", "roulette", "lexicase"):
            raise ValueError(f"selection must be 'tournament', 'roulette' or 'lexicase', got {self.selection}")
        if self.selection == "tournament" and self.tournament_size < 1:
            raise ValueError(f"tournament_size must be >= 1, got {self.tournament_size}")
        if self.lexicase_epsilon not in ("mad", "none"):
            raise ValueError(f"lexicase_epsilon must be 'mad' or 'none', got {self.lexicase_epsilon}")
        if not (0.0 <= self.crossover_prob <= 1.0):
            raise ValueError(f"crossover_prob must be in [0, 1], got {self.crossover_prob}")
        if not (0.0 <= self.mutation_prob <= 1.0):
            raise ValueError(f"mutation_prob must be in [0, 1], got {self.mutation_prob}")
        if self.mutation_type not in ("subtree", "point"):
            raise ValueError(f"mutation_type must be 'subtree' or 'point', got {self.mutation_type}")
        if self.elite_size < 0:
            raise ValueError(f"elite_size must be >= 0, got {self.elite_size}")
        if self.elitism and self.elite_size >= self.pop_size:
            raise ValueError(f"elite_size ({self.elite_size}) must be smaller than pop_size ({self.pop_size})")
        if not (0.0 <= self.const_prob <= 1.0):
            raise ValueError(f"const_prob must be in [0, 1], got {self.const_prob}")
        if len(self.const_range) != 2 or self.const_range[0] > self.const_range[1]:
            raise ValueError(f"const_range must be a pair (min, max) with min <= max, got {self.const_range}")
        if not self.function_set:
            raise ValueError("function_set cannot be empty")
        valid_fns = {"add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt"}
        for fn in self.function_set:
            if fn not in valid_fns:
                raise ValueError(f"Unknown function '{fn}' in function_set. Valid functions are: {sorted(valid_fns)}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> GPConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Convert const_range to tuple if list
        if "const_range" in data and isinstance(data["const_range"], list):
            data["const_range"] = tuple(data["const_range"])
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        data = asdict(self)
        if isinstance(data.get("const_range"), tuple):
            data["const_range"] = list(data["const_range"])
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def override(self, overrides: Sequence[str]) -> GPConfig:
        """Override configuration fields using list of 'key=value' strings."""
        type_hints = {
            "dataset": str,
            "pop_size": int,
            "generations": int,
            "max_size": int,
            "size_mode": str,
            "init_method": str,
            "init_min_depth": int,
            "selection": str,
            "tournament_size": int,
            "lexicase_epsilon": str,
            "crossover_prob": float,
            "mutation_prob": float,
            "mutation_type": str,
            "elitism": lambda v: str(v).lower() in ("true", "1", "yes"),
            "elite_size": int,
            "function_set": lambda v: [x.strip() for x in v.split(",") if x.strip()],
            "const_range": lambda v: tuple(float(x.strip()) for x in v.split(",")),
            "const_prob": float,
            "normalize_concrete": lambda v: str(v).lower() in ("true", "1", "yes"),
            "seed": int,
            "n_runs": int,
            "n_jobs": int,
        }
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be in the format key=value, got '{item}'")
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not hasattr(self, k):
                raise ValueError(f"Unknown config parameter: '{k}'")
            conv = type_hints.get(k, str)
            setattr(self, k, conv(v))
        self.validate()
        return self


def parse_cli_config(argv: Sequence[str] | None = None) -> Tuple[GPConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="GP Symbolic Regression Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to base YAML config")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name override")
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    parser.add_argument("--override", nargs="*", default=[], help="Overrides in key=value format")
    args = parser.parse_args(argv)

    if Path(args.config).exists():
        cfg = GPConfig.from_yaml(args.config)
    else:
        cfg = GPConfig()

    if args.dataset is not None:
        cfg.dataset = args.dataset
    if args.seed is not None:
        cfg.seed = args.seed
    if args.override:
        cfg.override(args.override)

    return cfg, args
