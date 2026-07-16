from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from kd_sensing.engine.modality_resolution import config_uses_gps
from kd_sensing.utils.checkpoint import checkpoint_sidecar_path, load_torch_payload
from kd_sensing.utils.paths import resolve_path


GPS_CHECKPOINT_PROVENANCE_KEYS = ("gps_feature_mode",)
TRAINING_PROFILE_CHECKPOINT_PROVENANCE_KEYS = ("training_profile", "t2_design_screening")


@dataclass(frozen=True)
class CheckpointResolution:
    path: Path | None
    source: str
    metadata: dict[str, Any] | None = None
    requested: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "source": self.source,
            "metadata": self.metadata,
            "requested": self.requested,
        }


def sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "run"


def load_checkpoint_metadata(checkpoint_path: str | Path | None) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    sidecar = checkpoint_sidecar_path(path)
    if sidecar.is_file():
        try:
            with sidecar.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, dict):
            return metadata
    if not path.is_file():
        return None
    try:
        payload = load_torch_payload(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    metadata: dict[str, Any] = {"path": str(path)}
    normalization = payload.get("normalization_artifacts")
    if isinstance(normalization, dict):
        metadata["normalization_artifacts"] = normalization
    metadata.update({key: payload[key] for key in GPS_CHECKPOINT_PROVENANCE_KEYS if key in payload})
    metadata.update({key: payload[key] for key in TRAINING_PROFILE_CHECKPOINT_PROVENANCE_KEYS if key in payload})
    role = payload.get("checkpoint_role")
    if role is not None:
        metadata["checkpoint_role"] = role
    return metadata


def gps_checkpoint_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    if not config_uses_gps(cfg):
        return {}
    dataset = cfg.get("data", {}).get("dataset", {})
    mode = str(dataset.get("gps_feature_mode", "relative_polar"))
    protocol = cfg.get("mmw_all_weather_protocol")
    if isinstance(protocol, dict) and protocol.get("gps_feature_mode", mode) != mode:
        raise ValueError("mmw_all_weather_protocol.gps_feature_mode must match data.dataset.gps_feature_mode.")
    return {"gps_feature_mode": mode}


def validate_evaluation_gps_checkpoint_provenance(cfg: dict[str, Any], metadata: dict[str, Any] | None) -> None:
    expected = gps_checkpoint_provenance(cfg)
    recorded = metadata or {}
    if expected and recorded.get("gps_feature_mode") not in {None, expected["gps_feature_mode"]}:
        raise ValueError("Checkpoint gps_feature_mode does not match the evaluation recipe.")


def canonical_payload_sha256(payload: Any) -> str:
    """Hash a JSON-compatible payload with a stable encoding."""
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_profile_sha256(profile_id: str, canonical_values: dict[str, Any]) -> str:
    return canonical_payload_sha256({"id": str(profile_id), "canonical_values": canonical_values})


def canonical_t2_design_config_sha256(cfg: dict[str, Any]) -> str:
    """Fingerprint a generated H4 config without trusting its self-reported digest."""
    payload = deepcopy(cfg)
    # The CLI records invocation details under runtime after loading generated YAML.
    # They locate a run but do not define its comparable recipe.
    payload.pop("runtime", None)
    screen = payload.get("mmw_t2_design_screening")
    if not isinstance(screen, dict):
        raise ValueError("T2 design-screening config is missing mmw_t2_design_screening provenance.")
    screen.pop("config_sha256", None)
    screen.pop("candidate_recipe_sha256", None)
    return canonical_payload_sha256(payload)


def t2_design_candidate_recipe_sha256(cfg: dict[str, Any]) -> str:
    """Fingerprint candidate-defining fields shared by a full run and a shape probe."""
    payload = deepcopy(cfg)
    payload.pop("runtime", None)
    payload.pop("output", None)
    dataset = payload.get("data", {}).get("dataset")
    if isinstance(dataset, dict):
        dataset.pop("domains", None)
    screen = payload.get("mmw_t2_design_screening")
    if not isinstance(screen, dict):
        raise ValueError("T2 design-screening config is missing mmw_t2_design_screening provenance.")
    for key in (
        "config_sha256",
        "candidate_recipe_sha256",
        "inner_split_fingerprint",
        "probe_scope",
        "probe_scope_reason",
    ):
        screen.pop(key, None)
    return canonical_payload_sha256(payload)


def _validated_training_profile(cfg: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    profile_id = profile.get("id")
    canonical_values = profile.get("canonical_values")
    fingerprint = profile.get("sha256")
    if not isinstance(profile_id, str) or not profile_id or not isinstance(canonical_values, dict) or not fingerprint:
        raise ValueError("MMW training_profile provenance requires non-empty id, canonical_values and sha256.")
    expected_fingerprint = training_profile_sha256(profile_id, canonical_values)
    if str(fingerprint) != expected_fingerprint:
        raise ValueError("MMW training_profile sha256 does not match its canonical values.")
    training_values = canonical_values.get("training")
    scheduler_values = canonical_values.get("scheduler")
    if not isinstance(training_values, dict) or not isinstance(scheduler_values, dict):
        raise ValueError("MMW training_profile canonical values must contain training and scheduler mappings.")
    actual_training = cfg.get("training")
    if not isinstance(actual_training, dict) or any(actual_training.get(key) != value for key, value in training_values.items()):
        raise ValueError("MMW training_profile canonical training values do not match the resolved config.")
    if cfg.get("scheduler") != scheduler_values:
        raise ValueError("MMW training_profile canonical scheduler does not match the resolved config.")
    return deepcopy(profile)


def training_profile_checkpoint_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = cfg.get("mmw_all_weather_protocol")
    if not isinstance(protocol, dict):
        return {}
    profile = protocol.get("training_profile")
    if not isinstance(profile, dict):
        return {}
    result: dict[str, Any] = {"training_profile": _validated_training_profile(cfg, profile)}
    design = cfg.get("mmw_t2_design_screening")
    if isinstance(design, dict):
        candidate = design.get("candidate_id")
        fingerprint = design.get("config_sha256")
        recipe_fingerprint = design.get("candidate_recipe_sha256")
        inner_split_fingerprint = design.get("inner_split_fingerprint")
        if not candidate or not fingerprint or not recipe_fingerprint or not inner_split_fingerprint or inner_split_fingerprint == "unbound":
            raise ValueError(
                "T2 design-screening provenance requires candidate_id, config_sha256, candidate_recipe_sha256 and inner_split_fingerprint."
            )
        actual_fingerprint = canonical_t2_design_config_sha256(cfg)
        if str(fingerprint) != actual_fingerprint:
            raise ValueError("T2 design-screening config_sha256 does not match the resolved config.")
        if str(recipe_fingerprint) != t2_design_candidate_recipe_sha256(cfg):
            raise ValueError("T2 design-screening candidate_recipe_sha256 does not match the resolved config.")
        result["t2_design_screening"] = {
            "protocol": design.get("protocol"),
            "candidate_id": candidate,
            "wave": design.get("wave"),
            "matched_control": design.get("matched_control"),
            "config_sha256": fingerprint,
            "candidate_recipe_sha256": recipe_fingerprint,
            "inner_split_fingerprint": inner_split_fingerprint,
        }
    return result


def validate_evaluation_training_profile_provenance(cfg: dict[str, Any], metadata: dict[str, Any] | None) -> None:
    expected = training_profile_checkpoint_provenance(cfg)
    if not expected:
        return
    recorded = metadata or {}
    if recorded.get("training_profile") != expected["training_profile"]:
        raise ValueError("Checkpoint training_profile provenance does not match the evaluation recipe.")
    if "t2_design_screening" in expected and recorded.get("t2_design_screening") != expected["t2_design_screening"]:
        raise ValueError("Checkpoint T2 design-screening provenance does not match the evaluation recipe.")


def resolve_evaluation_checkpoint(cfg: dict[str, Any], weights: str | None = None) -> CheckpointResolution:
    requested = weights or cfg.get("evaluation", {}).get("weights")
    if not requested:
        return CheckpointResolution(path=None, source="none")
    path = resolve_path(str(requested))
    return CheckpointResolution(
        path=path,
        source="explicit",
        metadata=load_checkpoint_metadata(path),
        requested=str(requested),
    )
