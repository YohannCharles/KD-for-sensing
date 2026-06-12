from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from kd_sensing.diagnostics.viewer_manifest_sampling import SampleCandidate

def _sample_id(dataset: Any, split: str, candidate: SampleCandidate) -> str:
    scene_slug = str(getattr(dataset, "scene_slug", "scene"))
    seq = str(candidate.seq_index).replace("/", "_").replace("\\", "_")
    return f"{scene_slug}_{split}_idx{candidate.dataset_index:06d}_seq{seq}"

def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().numpy())
    return value

__all__ = ["_json_ready", "_sample_id"]
