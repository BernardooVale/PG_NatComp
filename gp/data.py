from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd


DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def find_dataset_files(name: str, base_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Locate train and test CSV files for a given dataset name."""
    if base_dir is None:
        base_dir = DATASETS_DIR

    clean_name = name.lower().strip()
    candidate_names = [clean_name]
    if clean_name == "concrete1":
        candidate_names.append("concrete")
    elif clean_name == "concrete":
        candidate_names.append("concrete1")

    patterns = []
    for cand in candidate_names:
        patterns.extend([
            (base_dir / cand / f"{cand}-train.csv", base_dir / cand / f"{cand}-test.csv"),
            (base_dir / f"{cand}-train.csv", base_dir / f"{cand}-test.csv"),
        ])

    for train_p, test_p in patterns:
        if train_p.exists() and test_p.exists():
            return train_p, test_p

    raise FileNotFoundError(
        f"Could not find dataset files for '{name}' in '{base_dir}'. "
        f"Searched candidates: {candidate_names}"
    )


def load_dataset(
    name: str,
    normalize: bool = False,
    base_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset, splitting features X and target y.

    Parameters:
        name: dataset name ('synth1', 'synth2', 'concrete', 'concrete1')
        normalize: if True, applies z-score normalization to X using train statistics only.
        base_dir: optional base datasets directory

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    train_file, test_file = find_dataset_files(name, base_dir=base_dir)

    train_df = pd.read_csv(train_file, header=None)
    test_df = pd.read_csv(test_file, header=None)

    X_train = train_df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y_train = train_df.iloc[:, -1].to_numpy(dtype=np.float64)

    X_test = test_df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y_test = test_df.iloc[:, -1].to_numpy(dtype=np.float64)

    if normalize:
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std = np.where(std < 1e-9, 1.0, std)

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test
