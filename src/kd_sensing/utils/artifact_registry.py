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
TRAINING_PROFILE_CHECKPOINT_PROVENANCE_KEYS = (
    "training_profile",
    "router_architecture_profile",
    "t2_design_screening",
    "mmw_twc_evidence",
)


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


def router_architecture_profile_sha256(profile_id: str, canonical_values: dict[str, Any]) -> str:
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


def canonical_mmw_twc_evidence_config_sha256(cfg: dict[str, Any]) -> str:
    """Fingerprint a strict TWC recipe while excluding runtime-only controls."""
    payload = deepcopy(cfg)
    payload.pop("runtime", None)
    payload.pop("output", None)
    training = payload.get("training")
    if isinstance(training, dict) and "resume" in training:
        # Existing frozen recipes include resume=false in their digest.  Normalize
        # the CLI-only resume=true override back to that frozen value instead of
        # changing the identity of an interrupted run.
        training["resume"] = False
    evidence = payload.get("mmw_twc_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("MMW TWC config is missing mmw_twc_evidence provenance.")
    evidence.pop("config_recipe_sha256", None)
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


def _validated_router_architecture_profile(cfg: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    profile_id = profile.get("id")
    canonical_values = profile.get("canonical_values")
    fingerprint = profile.get("sha256")
    if not isinstance(profile_id, str) or not profile_id or not isinstance(canonical_values, dict) or not fingerprint:
        raise ValueError("MMW router_architecture_profile provenance requires non-empty id, canonical_values and sha256.")
    expected_fingerprint = router_architecture_profile_sha256(profile_id, canonical_values)
    if str(fingerprint) != expected_fingerprint:
        raise ValueError("MMW router_architecture_profile sha256 does not match its canonical values.")
    model_values = canonical_values.get("model")
    if not isinstance(model_values, dict):
        raise ValueError("MMW router_architecture_profile canonical values must contain a model mapping.")
    primary_values = model_values.get("primary")
    if not isinstance(primary_values, dict) or set(primary_values) != {"router_use_pattern_features"}:
        raise ValueError(
            "MMW router_architecture_profile canonical values must declare only model.primary.router_use_pattern_features."
        )
    actual_primary = cfg.get("model", {}).get("primary", {})
    if not isinstance(actual_primary, dict) or any(actual_primary.get(key) != value for key, value in primary_values.items()):
        raise ValueError("MMW router_architecture_profile canonical values do not match the resolved config.")
    return deepcopy(profile)


def training_profile_checkpoint_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = cfg.get("mmw_all_weather_protocol")
    result: dict[str, Any] = {}
    if isinstance(protocol, dict):
        profile = protocol.get("training_profile")
        if isinstance(profile, dict):
            result["training_profile"] = _validated_training_profile(cfg, profile)
        router_profile = protocol.get("router_architecture_profile")
        if isinstance(router_profile, dict):
            result["router_architecture_profile"] = _validated_router_architecture_profile(cfg, router_profile)
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
    evidence = cfg.get("mmw_twc_evidence")
    if isinstance(evidence, dict):
        result["mmw_twc_evidence"] = _validated_mmw_twc_evidence(cfg, evidence)
    return result


def validate_evaluation_training_profile_provenance(cfg: dict[str, Any], metadata: dict[str, Any] | None) -> None:
    expected = training_profile_checkpoint_provenance(cfg)
    if not expected:
        return
    recorded = metadata or {}
    if "training_profile" in expected and recorded.get("training_profile") != expected["training_profile"]:
        raise ValueError("Checkpoint training_profile provenance does not match the evaluation recipe.")
    if "router_architecture_profile" in expected and recorded.get("router_architecture_profile") != expected[
        "router_architecture_profile"
    ]:
        raise ValueError("Checkpoint router_architecture_profile provenance does not match the evaluation recipe.")
    if "t2_design_screening" in expected and recorded.get("t2_design_screening") != expected["t2_design_screening"]:
        raise ValueError("Checkpoint T2 design-screening provenance does not match the evaluation recipe.")
    if "mmw_twc_evidence" in expected and recorded.get("mmw_twc_evidence") != expected["mmw_twc_evidence"]:
        raise ValueError("Checkpoint MMW TWC evidence provenance does not match the evaluation recipe.")


def _validated_mmw_twc_evidence(cfg: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    required = (
        "protocol_id",
        "protocol_manifest_sha256",
        "confirmation_split_manifest_sha256",
        "training_role",
        "smoke_preflight",
        "training_mask_seed",
        "training_mask_seed_algorithm",
        "domain_sampling_seed",
        "evaluation_mask_cache_sha256",
        "evaluation_mask_cache_checksum",
        "topology_id",
        "topology_descriptor_sha256",
        "topology_mapping_sha256",
        "evaluation_topology_id",
        "evaluation_topology_descriptor_sha256",
        "config_recipe_sha256",
    )
    missing = [key for key in required if not str(evidence.get(key, "")).strip()]
    if missing:
        raise ValueError(f"MMW TWC evidence provenance is missing {missing}.")
    if evidence.get("protocol_id") != "mmw_twc_outer_v1":
        raise ValueError("Unsupported MMW TWC evidence protocol id.")
    if evidence.get("config_recipe_sha256") != canonical_mmw_twc_evidence_config_sha256(cfg):
        raise ValueError("MMW TWC config_recipe_sha256 does not match the resolved config.")
    if evidence.get("training_role") != "confirmation_train":
        raise ValueError("MMW TWC checkpoint provenance requires training_role=confirmation_train.")
    experiment_seed = int(cfg.get("experiment", {}).get("seed", -1))
    if int(evidence["training_mask_seed"]) != experiment_seed:
        raise ValueError("MMW TWC training_mask_seed must match experiment.seed.")
    temporal_missing = cfg.get("temporal_missing")
    if not isinstance(temporal_missing, dict) or int(temporal_missing.get("seed", -1)) != experiment_seed:
        raise ValueError("MMW TWC temporal_missing.seed must match experiment.seed.")
    schedules = {
        "mmw_fair_pattern_v1": {
            "panel_size": 600,
            "condition_counts": {
                "clean": 120,
                "drop1": 60,
                "drop2": 60,
                "drop3": 60,
                "token20": 60,
                "token40": 60,
                "token60": 60,
                "token80": 60,
                "token90": 60,
            },
            "seed_algorithm": "sha256(base_seed,balanced_pattern_schedule,epoch); sample=(step*train_batch_size+row)%600",
        },
        "mmw_fair_whole_modality_v1": {
            "panel_size": 480,
            "condition_counts": {
                "clean": 120,
                "drop1": 120,
                "drop2": 120,
                "drop3": 120,
                "token20": 0,
                "token40": 0,
                "token60": 0,
                "token80": 0,
                "token90": 0,
            },
            "seed_algorithm": "sha256(base_seed,balanced_whole_pattern_schedule,epoch); sample=(step*train_batch_size+row)%480",
        },
    }
    schedule_id = str(temporal_missing.get("schedule_id", ""))
    schedule = schedules.get(schedule_id)
    configured_counts = temporal_missing.get("condition_counts")
    if (
        temporal_missing.get("mode") != "balanced_pattern_schedule"
        or schedule is None
        or int(temporal_missing.get("panel_size", -1)) != int(schedule["panel_size"])
        or configured_counts != schedule["condition_counts"]
        or evidence.get("training_mask_seed_algorithm") != schedule["seed_algorithm"]
    ):
        raise ValueError(
            "MMW TWC temporal-missing schedule provenance or training_mask_seed_algorithm is unsupported or inconsistent."
        )
    domain_sampling = cfg.get("data", {}).get("domain_balanced_sampling", {})
    if not isinstance(domain_sampling, dict) or int(domain_sampling.get("seed", -1)) != experiment_seed:
        raise ValueError("MMW TWC domain_balanced_sampling.seed must match experiment.seed.")
    if int(evidence["domain_sampling_seed"]) != experiment_seed:
        raise ValueError("MMW TWC domain_sampling_seed must match experiment.seed.")
    final_test = cfg.get("training", {}).get("final_test")
    if not isinstance(final_test, dict) or bool(final_test.get("enabled", True)):
        raise ValueError("MMW TWC confirmation training must explicitly disable final test.")
    expected_topology = _resolved_t2_topology(cfg)
    if (
        evidence.get("topology_id") != expected_topology["id"]
        or evidence.get("topology_descriptor_sha256") != expected_topology["descriptor_sha256"]
        or evidence.get("topology_mapping_sha256") != expected_topology["mapping_sha256"]
    ):
        raise ValueError("MMW TWC evidence topology provenance does not match the resolved config.")
    if evidence.get("evaluation_topology_id") != "ula_dft_phase_cycle_v1":
        raise ValueError("MMW TWC evaluation topology must be the audited ULA-DFT phase cycle.")
    evaluation_descriptor = str(evidence.get("evaluation_topology_descriptor_sha256", ""))
    if len(evaluation_descriptor) != 64 or any(char not in "0123456789abcdef" for char in evaluation_descriptor.lower()):
        raise ValueError("MMW TWC evaluation topology requires a 64-character descriptor sha256.")
    return {
        key: evidence[key]
        for key in required
    }


def _resolved_t2_topology(cfg: dict[str, Any]) -> dict[str, str]:
    loss_cfg = cfg.get("loss", {}).get("u_mask_beam_jepa", {})
    if not isinstance(loss_cfg, dict) or not bool(loss_cfg.get("use_beam_prototype_alignment", False)):
        return {
            "id": "not_applicable",
            "descriptor_sha256": "not_applicable",
            "mapping_sha256": "not_applicable",
        }
    topology = loss_cfg.get("prototype_topology")
    if topology is None:
        return {
            "id": "cyclic_index_v1" if bool(loss_cfg.get("prototype_target_circular", True)) else "linear_index_v1",
            "descriptor_sha256": "legacy_unbound",
            "mapping_sha256": "legacy_unbound",
        }
    if not isinstance(topology, dict):
        raise ValueError("MMW TWC prototype_topology must be a mapping.")
    topology_id = str(topology.get("id", "")).strip()
    descriptor = str(topology.get("descriptor_sha256", ""))
    if topology_id == "ula_dft_phase_cycle_v1" and not descriptor:
        raise ValueError("MMW TWC physical BPA topology requires descriptor_sha256.")
    permutation = topology.get("permutation", [])
    if topology_id == "permuted_index_v1":
        if not isinstance(permutation, list) or len(permutation) != 64 or set(permutation) != set(range(64)):
            raise ValueError("MMW TWC permuted topology must carry a 64-label bijection.")
    elif permutation not in (None, [], ()):
        raise ValueError("Only a permuted MMW TWC topology may carry a permutation.")
    return {
        "id": topology_id,
        "descriptor_sha256": descriptor or "not_applicable",
        "mapping_sha256": canonical_payload_sha256(
            {"id": topology_id, "permutation": list(permutation) if isinstance(permutation, tuple) else permutation or []}
        ),
    }


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
