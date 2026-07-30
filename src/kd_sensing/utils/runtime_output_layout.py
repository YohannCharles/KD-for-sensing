from pathlib import Path
import re
from typing import Any, Mapping


OUTPUT_ROOT = "outputs"
PARTITION_CACHE = "cache"
PARTITION_CLEANUP_MANIFESTS = "cleanup_manifests"
PARTITION_ANALYSIS = "analysis"
PARTITION_EVALUATIONS = "evaluations"
PARTITION_ARCHIVE = "archive"
PARTITION_FEATURES = "features"
PARTITION_TRAINING = "training"
DEFAULT_NON_RUN_PARTITIONS = (PARTITION_CACHE, PARTITION_ARCHIVE, PARTITION_CLEANUP_MANIFESTS)
PROTECTED_MAINLINE_PARTITIONS = (
    PARTITION_ANALYSIS,
    PARTITION_CACHE,
    PARTITION_CLEANUP_MANIFESTS,
    PARTITION_EVALUATIONS,
    PARTITION_ARCHIVE,
    PARTITION_FEATURES,
    PARTITION_TRAINING,
)


def canonical_runtime_partitions(outputs_root: str | Path = OUTPUT_ROOT) -> dict[str, str]:
    root = Path(outputs_root)
    return {name: str(root / name) for name in PROTECTED_MAINLINE_PARTITIONS}


def runtime_scope_metadata_from_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    dataset = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), Mapping) else {}
    if not isinstance(dataset, Mapping) or str(dataset.get("type", "mmw")) != "mmw":
        return {}
    return {"scope_kind": "mmw", "scope_slug": "mmw", "source": "data.dataset.type"}


def evaluation_study_id_from_config(cfg: Mapping[str, Any]) -> str:
    output = cfg.get("output", {}) if isinstance(cfg.get("output"), Mapping) else {}
    evaluation = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), Mapping) else {}
    experiment = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), Mapping) else {}
    return _safe_slug(
        str(
            output.get("evaluation_study_id")
            or evaluation.get("study_id")
            or experiment.get("name")
            or output.get("run_name")
            or "evaluation"
        )
    )


def evaluation_output_base(base: str | Path, cfg: Mapping[str, Any]) -> Path:
    root = Path(base)
    study = evaluation_study_id_from_config(cfg)
    if root.name == study and root.parent.name == PARTITION_EVALUATIONS:
        return root
    return root / PARTITION_EVALUATIONS / study


def output_layout_summary(path: str | Path, *, outputs_root: str | Path | None = None) -> dict[str, Any]:
    target = Path(path)
    root = Path(outputs_root) if outputs_root is not None else _nearest_outputs_root(target)
    if root is None:
        return {"outputs_root": None, "canonical_partition": "outside_outputs", "scope_kind": None}
    try:
        relative = target.resolve().relative_to(root.resolve()).parts
    except (OSError, ValueError):
        relative = ()
    partition = relative[0] if relative else ""
    return {
        "outputs_root": str(root),
        "canonical_partition": partition or "root",
        "scope_kind": "mmw" if partition == "mmw" else None,
        "scope_slug": partition or None,
        "archive": partition == PARTITION_ARCHIVE,
        "explicit_non_run_partition": partition in DEFAULT_NON_RUN_PARTITIONS and len(relative) == 1,
    }


def is_default_outputs_root(path: str | Path) -> bool:
    return Path(path).name == OUTPUT_ROOT


def is_default_skipped_partition(path: str | Path) -> bool:
    target = Path(path)
    return target.name in DEFAULT_NON_RUN_PARTITIONS and target.parent.name == OUTPUT_ROOT


def _nearest_outputs_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if candidate.name == OUTPUT_ROOT:
            return candidate
    return None


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "evaluation"


__all__ = [
    "DEFAULT_NON_RUN_PARTITIONS",
    "OUTPUT_ROOT",
    "PARTITION_ANALYSIS",
    "PARTITION_ARCHIVE",
    "PARTITION_CACHE",
    "PARTITION_CLEANUP_MANIFESTS",
    "PARTITION_EVALUATIONS",
    "PROTECTED_MAINLINE_PARTITIONS",
    "canonical_runtime_partitions",
    "evaluation_output_base",
    "evaluation_study_id_from_config",
    "is_default_outputs_root",
    "is_default_skipped_partition",
    "output_layout_summary",
    "runtime_scope_metadata_from_config",
]
