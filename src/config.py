"""Configuration loading with YAML and environment overrides."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Load config from YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
