import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckpointResolution:
    path: Path | None = None
    epoch: int | None = None
    policy: str = ""
    metric_value: float | None = None
    source: str = "missing"
    warnings: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path is not None else "",
            "epoch": self.epoch if self.epoch is not None else "",
            "policy": self.policy,
            "metric_value": self.metric_value if self.metric_value is not None else "",
            "source": self.source,
            "warnings": list(self.warnings),
            "candidates": list(self.candidates),
            "metadata": self.metadata,
        }


def resolve_checkpoint(
    root: str | Path,
    run_name: str,
    policy: str = "best_val_top1",
    manual_path: str | Path | None = None,
    metric_name: str = "primary_acc",
) -> CheckpointResolution:
    root_path = Path(root)
    run_name = str(run_name)
    if manual_path:
        return _manual_resolution(Path(manual_path), run_name, policy=policy)
    if policy == "manual_path":
        return CheckpointResolution(policy=policy, warnings=["manual_path policy requires manual_path"])

    candidates = _checkpoint_candidates(root_path, run_name, metric_name=metric_name)
    if not candidates:
        return CheckpointResolution(
            policy=policy,
            warnings=[f"no checkpoint candidates found for run {run_name} under {root_path}"],
        )

    if policy == "latest":
        selected = max(candidates, key=lambda item: item["mtime"])
        return _resolution_from_candidate(selected, policy=policy, candidates=candidates)
    if policy == "best_avg_missing_top1":
        avg_candidates = [
            item
            for item in candidates
            if "avg" in item["path"].name.lower() and "missing" in item["path"].name.lower()
        ]
        if avg_candidates:
            selected = max(avg_candidates, key=lambda item: (_metric_or_neg_inf(item.get("metric_value")), item["mtime"]))
            return _resolution_from_candidate(selected, policy=policy, candidates=candidates)
        fallback = resolve_checkpoint(root_path, run_name, "best_val_top1", metric_name=metric_name)
        fallback.policy = policy
        fallback.warnings.insert(0, "best_avg_missing_top1 checkpoint not found; fell back to best_val_top1")
        return fallback

    best_epoch, best_value, metric_warnings = _best_epoch_from_metrics(root_path / run_name / "metrics.csv")
    if policy == "best_epoch_from_metrics":
        exact = _candidate_for_epoch(candidates, best_epoch)
        if exact is None:
            return CheckpointResolution(
                policy=policy,
                warnings=[
                    *metric_warnings,
                    f"metrics best epoch {best_epoch or 'unavailable'} has no matching checkpoint for {run_name}",
                ],
                candidates=[str(item["path"]) for item in candidates],
            )
        return _resolution_from_candidate(exact, policy=policy, candidates=candidates, metric_value=best_value)

    if policy != "best_val_top1":
        return CheckpointResolution(
            policy=policy,
            warnings=[f"unsupported checkpoint policy: {policy}"],
            candidates=[str(item["path"]) for item in candidates],
        )

    exact = _candidate_for_epoch(candidates, best_epoch)
    if exact is not None:
        return _resolution_from_candidate(
            exact,
            policy=policy,
            candidates=candidates,
            metric_value=best_value if best_value is not None else exact.get("metric_value"),
            warnings=metric_warnings,
        )

    top1_candidates = [item for item in candidates if item["kind"] in {"local_best_top1", "registry_top1"}]
    metric_candidates = [item for item in candidates if item.get("metric_value") is not None]
    pool = top1_candidates or metric_candidates or candidates
    selected = max(pool, key=lambda item: (_metric_or_neg_inf(item.get("metric_value")), item["mtime"]))
    warnings = list(metric_warnings)
    if best_epoch is not None:
        warnings.append(f"best epoch {best_epoch} from metrics.csv did not match a checkpoint; used best candidate")
    return _resolution_from_candidate(
        selected,
        policy=policy,
        candidates=candidates,
        metric_value=selected.get("metric_value"),
        warnings=warnings,
    )


def _manual_resolution(path: Path, run_name: str, *, policy: str) -> CheckpointResolution:
    if not path.exists():
        return CheckpointResolution(policy=policy, warnings=[f"manual checkpoint not found: {path}"])
    metadata = _metadata(path)
    return CheckpointResolution(
        path=path,
        epoch=_epoch_from_metadata_or_name(metadata, path),
        policy=policy,
        metric_value=_metric_from_metadata_or_name(metadata, path),
        source="manual_path",
        warnings=[] if _belongs_to_run(path, metadata, run_name, metric_name="primary_acc") else [f"manual checkpoint is not clearly owned by run {run_name}"],
        candidates=[str(path)],
        metadata=metadata,
    )


def _checkpoint_candidates(root: Path, run_name: str, *, metric_name: str) -> list[dict[str, Any]]:
    paths: list[Path] = []
    run_ckpt_dir = root / run_name / "checkpoints"
    for name in ("best_top1.pth", "best.pth", "last.pth"):
        paths.append(run_ckpt_dir / name)
    if run_ckpt_dir.exists():
        paths.extend(sorted(run_ckpt_dir.glob("*.pth")))
    registry = root / "best_checkpoints"
    if registry.exists():
        paths.extend(sorted(registry.glob("*.pth")))

    seen: set[Path] = set()
    candidates: list[dict[str, Any]] = []
    for path in paths:
        path = path.resolve() if path.exists() else path
        if path in seen or not path.exists():
            continue
        seen.add(path)
        metadata = _metadata(path)
        if not _belongs_to_run(path, metadata, run_name, metric_name=metric_name):
            continue
        candidates.append(
            {
                "path": path,
                "metadata": metadata,
                "epoch": _epoch_from_metadata_or_name(metadata, path),
                "metric_value": _metric_from_metadata_or_name(metadata, path),
                "mtime": path.stat().st_mtime,
                "kind": _candidate_kind(path, metadata),
            }
        )
    return candidates


def _belongs_to_run(path: Path, metadata: dict[str, Any], run_name: str, *, metric_name: str) -> bool:
    run_dir = metadata.get("run_dir")
    if run_dir and Path(str(run_dir)).name == run_name:
        return True
    if metadata.get("config_slug") == run_name:
        return True
    if metadata.get("experiment_name") == run_name:
        return True
    if run_name in path.parts and "checkpoints" in path.parts:
        return True
    stem = path.stem
    return stem.startswith(f"{run_name}_{metric_name}_") or stem.startswith(f"{run_name}_acc_")


def _candidate_kind(path: Path, metadata: dict[str, Any]) -> str:
    source = str(metadata.get("checkpoint_source") or metadata.get("selection_mode") or "").lower()
    if path.name == "best_top1.pth" or "top1" in source:
        return "local_best_top1" if path.name == "best_top1.pth" else "registry_top1"
    if path.name == "last.pth":
        return "local_latest"
    if path.name == "best.pth":
        return "local_best_objective"
    return "registry"


def _candidate_for_epoch(candidates: list[dict[str, Any]], epoch: int | None) -> dict[str, Any] | None:
    if epoch is None:
        return None
    exact = [item for item in candidates if item.get("epoch") == epoch]
    if not exact:
        return None
    return max(exact, key=lambda item: (item["kind"] in {"local_best_top1", "registry_top1"}, item["mtime"]))


def _best_epoch_from_metrics(path: Path) -> tuple[int | None, float | None, list[str]]:
    if not path.exists():
        return None, None, [f"metrics.csv not found: {path}"]
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        return None, None, [f"failed to read metrics.csv: {exc}"]
    best_epoch: int | None = None
    best_value: float | None = None
    for row in rows:
        value = _first_float(row, "val_acc", "val_beam_top1", "top1_acc", "accuracy/top1", "primary_acc")
        epoch = _int_value(row.get("epoch"))
        if value is None or epoch is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_epoch = epoch
    if best_epoch is None:
        return None, None, [f"metrics.csv has no usable val_acc/top1 rows: {path}"]
    return best_epoch, best_value, []


def _resolution_from_candidate(
    candidate: dict[str, Any],
    *,
    policy: str,
    candidates: list[dict[str, Any]],
    metric_value: float | None = None,
    warnings: list[str] | None = None,
) -> CheckpointResolution:
    return CheckpointResolution(
        path=candidate["path"],
        epoch=candidate.get("epoch"),
        policy=policy,
        metric_value=metric_value if metric_value is not None else candidate.get("metric_value"),
        source=candidate.get("kind", "checkpoint"),
        warnings=list(warnings or []),
        candidates=[str(item["path"]) for item in candidates],
        metadata=candidate.get("metadata") or {},
    )


def _metadata(path: Path) -> dict[str, Any]:
    metadata = _sidecar_json(path)
    return metadata if isinstance(metadata, dict) else {}


def _sidecar_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _epoch_from_metadata_or_name(metadata: dict[str, Any], path: Path) -> int | None:
    for key in ("selected_epoch", "epoch", "best_top1_epoch", "best_early_stopping_epoch"):
        value = _int_value(metadata.get(key))
        if value is not None:
            return value
    for match in re.findall(r"(?:epoch|ep)[_-]?(\d+)", path.name.lower()):
        return int(match)
    return None


def _metric_from_metadata_or_name(metadata: dict[str, Any], path: Path) -> float | None:
    for key in ("metric_value", "primary_acc", "val_acc", "best_val_top1"):
        value = _float_value(metadata.get(key))
        if value is not None:
            return value
    task_metrics = metadata.get("task_metrics")
    if isinstance(task_metrics, dict):
        value = _first_float(task_metrics, "val_acc", "primary_acc", "val_beam_top1")
        if value is not None:
            return value
    match = re.search(r"(?:primary_acc|acc|top1)_([0-9]+(?:\.[0-9]+)?)", path.stem)
    return float(match.group(1)) if match else None


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float_value(row.get(key))
        if value is not None:
            return value
    return None


def _float_value(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _int_value(value: Any) -> int | None:
    numeric = _float_value(value)
    return int(numeric) if numeric is not None else None


def _metric_or_neg_inf(value: Any) -> float:
    numeric = _float_value(value)
    return numeric if numeric is not None else float("-inf")
