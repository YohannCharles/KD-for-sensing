from .paths import (
    config_dir,
    data_dir,
    output_dir,
    project_root,
    resolve_path,
)

__all__ = [
    "project_root",
    "resolve_path",
    "data_dir",
    "config_dir",
    "output_dir",
    "set_seed",
    "CheckpointResolution",
    "archive_best_checkpoint",
    "find_registry_checkpoint",
    "resolve_evaluation_checkpoint",
]


def __getattr__(name: str):
    if name == "set_seed":
        from .seed import set_seed

        globals()[name] = set_seed
        return set_seed
    if name in {
        "CheckpointResolution",
        "archive_best_checkpoint",
        "find_registry_checkpoint",
        "resolve_evaluation_checkpoint",
    }:
        from . import artifact_registry

        return getattr(artifact_registry, name)
    raise AttributeError(f"module 'kd_sensing.utils' has no attribute {name!r}")
