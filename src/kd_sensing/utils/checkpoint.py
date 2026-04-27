from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], save_path: str | Path, filename: str = "checkpoint.pth") -> Path:
    directory = Path(save_path)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    torch.save(state, filepath)
    return filepath


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None):
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint

