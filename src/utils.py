"""Small helpers shared by the scripts: seeds, device, config loading, model construction."""

import random

import numpy as np
import torch
import yaml


def set_seed(seed):
    """Fix every random number generator we use (python, numpy, torch CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    """One flag for CPU / GPU: 'auto' (GPU if available), 'cpu' or 'cuda'."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    """Build the TinyVLM from the `model:` section of a config dict."""
    from src.model.vlm import TinyVLM  # local import keeps `utils` importable without the model

    return TinyVLM(**cfg.get("model", {}))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
