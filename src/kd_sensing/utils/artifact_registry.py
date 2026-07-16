from copy import deepcopy
from dataclasses import dataclass
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


def training_profile_checkpoint_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = cfg.get("mmw_all_weather_protocol")
    if not isinstance(protocol, dict):
        return {}
    profile = protocol.get("training_profile")
    if not isinstance(profile, dict):
        return {}
    if not profile.get("id") or not profile.get("sha256"):
        raise ValueError("MMW training_profile provenance requires non-empty id and sha256.")
    result: dict[str, Any] = {"training_profile": deepcopy(profile)}
    design = cfg.get("mmw_t2_design_screening")
    if isinstance(design, dict):
        candidate = design.get("candidate_id")
        fingerprint = design.get("config_sha256")
        if not candidate or not fingerprint:
            raise ValueError("T2 design-screening provenance requires candidate_id and config_sha256.")
        result["t2_design_screening"] = {
            "protocol": design.get("protocol"),
            "candidate_id": candidate,
            "wave": design.get("wave"),
            "matched_control": design.get("matched_control"),
            "config_sha256": fingerprint,
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
