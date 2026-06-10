from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SUPERVISION_FIELDS = {
    "beam",
    "target_beam",
    "beam_abs",
    "beam_geo",
    "beam_residual",
    "residual_class",
    "beam_power",
    "beamspace_power",
    "beamspace_power_label",
    "csi",
    "channel",
    "channel_path",
    "path",
    "path_params",
    "path_descriptor",
    "path_semantic_label",
    "radio_semantic_label",
}
TARGET_UNLABELED = "target_unlabeled"
TARGET_TEST = "target_test"
TARGET_LABELED = "target_labeled"


class TargetShotSupervisionGuardError(RuntimeError):
    pass


def assert_target_supervision_allowed(
    metadata: Mapping[str, Any] | None,
    field_name: str,
    *,
    scope: str = "training",
) -> None:
    meta = dict(metadata or {})
    subset = str(meta.get("target_subset", meta.get("subset", meta.get("split", "")))).strip()
    field = str(field_name)
    normalized = field.strip().lower()
    if normalized not in SUPERVISION_FIELDS:
        return
    if subset == TARGET_UNLABELED and scope not in {"offline_diagnostics"}:
        raise TargetShotSupervisionGuardError(
            f"Target-shot supervision guard blocked split=subset={subset}, field={field}, "
            f"label_fraction={meta.get('target_label_fraction')}, "
            f"artifact={meta.get('split_artifact_path', meta.get('target_shot_split_artifact'))}. "
            "Use target_labeled for supervised target loss or mark this read as offline diagnostics."
        )
    if subset == TARGET_TEST and scope not in {"evaluation", "offline_diagnostics"}:
        raise TargetShotSupervisionGuardError(
            f"Target-shot supervision guard blocked split=subset={subset}, field={field}, "
            f"artifact={meta.get('split_artifact_path', meta.get('target_shot_split_artifact'))}. "
            "target_test labels are available only in evaluation scope."
        )


def filter_target_shot_training_payload(
    sample: Mapping[str, Any],
    *,
    scope: str = "training",
) -> dict[str, Any]:
    payload = dict(sample)
    metadata = _metadata_from_sample(sample)
    for field in list(payload):
        assert_target_supervision_allowed(metadata, field, scope=scope)
    return payload


def target_shot_runtime_metadata(artifact: Mapping[str, Any], *, artifact_path: str | Path | None = None) -> dict[str, Any]:
    cfg = artifact.get("config_summary", {}) if isinstance(artifact.get("config_summary"), Mapping) else {}
    splits = artifact.get("splits", {}) if isinstance(artifact.get("splits"), Mapping) else {}
    strict = artifact.get("strict_eligibility", {}) if isinstance(artifact.get("strict_eligibility"), Mapping) else {}
    return {
        "target_shot_split_artifact": str(artifact_path or artifact.get("artifact_path", "")),
        "source_domains": list(cfg.get("source_domains", [])),
        "target_domains": list(cfg.get("target_domains", [])),
        "target_label_fraction": cfg.get("target_label_fraction"),
        "seed": cfg.get("seed"),
        "target_labeled_count": _split_count(splits, TARGET_LABELED),
        "target_unlabeled_count": _split_count(splits, TARGET_UNLABELED),
        "target_test_count": _split_count(splits, TARGET_TEST),
        "source_count": _split_count(splits, "source"),
        "strict_eligibility": strict,
        "leakage_diagnostics": artifact.get("leakage_diagnostics", {}),
    }


def load_target_shot_runtime_metadata(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    return target_shot_runtime_metadata(artifact, artifact_path=path)


def _metadata_from_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    meta = sample.get("metadata", {})
    if isinstance(meta, Mapping):
        return dict(meta)
    return {}


def _split_count(splits: Mapping[str, Any], name: str) -> int:
    payload = splits.get(name, {})
    if isinstance(payload, Mapping):
        return int(payload.get("count", len(payload.get("sample_ids", []))))
    return 0


__all__ = [
    "SUPERVISION_FIELDS",
    "TARGET_LABELED",
    "TARGET_TEST",
    "TARGET_UNLABELED",
    "TargetShotSupervisionGuardError",
    "assert_target_supervision_allowed",
    "filter_target_shot_training_payload",
    "load_target_shot_runtime_metadata",
    "target_shot_runtime_metadata",
]
