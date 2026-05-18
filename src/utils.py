"""Shared utilities."""

import json
import random
import sys
from pathlib import Path
from typing import Any, List

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(*paths) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def chunk(lst: List, size: int) -> List[List]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def print_summary(metrics: dict, method: str) -> None:
    print(f"\n{'─'*50}")
    print(f"Method : {method}")
    print(f"  Accuracy       : {metrics.get('accuracy')}")
    print(f"  Macro F1       : {metrics.get('macro_f1')}")
    print(f"  Weighted F1    : {metrics.get('weighted_f1')}")
    print(f"  Worst-class F1 : {metrics.get('worst_class_f1')}")
    if "per_class_f1" in metrics:
        print("  Per-class F1:")
        for label, score in metrics["per_class_f1"].items():
            print(f"    {label:<12}: {score}")
    print(f"{'─'*50}")
