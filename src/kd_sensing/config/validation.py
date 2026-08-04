from typing import Any

from kd_sensing.modalities import normalize_modalities, resolve_image_profile, validate_image_profile_size


RETAINED_MODALITIES = ("image", "radar", "gps", "lidar")
RETAINED_MODELS = {"u_mask_beam_jepa", "modular_sequence", "pcpf_temporal_risk_fusion"}
RETAINED_DATASETS = {"mmw", "deepsense6g"}
DEEPSENSE6G_SCENES = {31, 32, 33, 34}


def validate_loaded_config(cfg: dict[str, Any]) -> None:
    data_section = cfg.get("data", {})
    data = data_section.get("dataset", {})
    model = cfg.get("model", {}).get("primary", {})
    if cfg.get("preprocessing"):
        return
    if str(cfg.get("experiment", {}).get("objective", "beam")).strip().lower() != "beam":
        raise ValueError("Only experiment.objective='beam' is retained.")
    dataset_type = str(data.get("type", "")).strip().lower()
    if dataset_type not in RETAINED_DATASETS:
        raise ValueError(f"Supported data.dataset.type values are {sorted(RETAINED_DATASETS)}.")
    retired = sorted(set(cfg) & {"bcacl", "cmsbl", "standalone_capacity_reference"})
    if retired:
        raise ValueError(f"Retired training sections are not supported: {retired}.")
    modalities = normalize_modalities(model.get("modalities", ()), context="model.primary.modalities")
    if modalities != RETAINED_MODALITIES:
        raise ValueError(f"Retained workflows require modalities {list(RETAINED_MODALITIES)}.")
    if str(model.get("type", "")) not in RETAINED_MODELS:
        raise ValueError(f"Only retained model types are supported: {sorted(RETAINED_MODELS)}.")
    if int(data.get("seq_len", 0)) <= 0 or int(data.get("num_pred", 0)) <= 0:
        raise ValueError("data.dataset.seq_len and num_pred must be positive.")
    image_profile = resolve_image_profile(data.get("image_profile"))
    validate_image_profile_size(image_profile, tuple(data.get("image_size", (224, 224))))
    fft = tuple(data.get("fft_tuple", ()))
    if len(fft) < 3 or int(fft[0]) != 64 or int(fft[2]) != 128 or int(data.get("clipped_range", 0)) != 128:
        raise ValueError("Retained radar inputs require fft_tuple [64, *, 128] and clipped_range=128.")
    if str(model.get("type", "")) == "pcpf_temporal_risk_fusion":
        _validate_pcpf_config(cfg, model, data)
    if dataset_type == "mmw":
        if data_section.get("split_protocol") != "mmw_id_stratified_block_v1":
            raise ValueError("MMW data.split_protocol must be 'mmw_id_stratified_block_v1'.")
        if int(data_section.get("split_seed", -1)) < 0:
            raise ValueError("MMW data.split_seed must be a non-negative integer.")
        if int(data_section.get("block_size", 0)) != 32:
            raise ValueError("MMW data.block_size must be 32 for the canonical protocol.")
        if data_section.get("split_ratios") != {"train": 0.70, "validation": 0.15, "test": 0.15}:
            raise ValueError("MMW data.split_ratios must be the canonical 70/15/15 mapping.")
        retired_names = {"split_mode", "split_strategy", "train_ratio", "val_ratio", "test_ratio"}
        retired_split_fields = sorted((set(data_section) | set(data)) & retired_names)
        if retired_split_fields:
            raise ValueError(f"Retired MMW split fields are not supported: {retired_split_fields}.")
        experiment = cfg.get("experiment", {})
        train_seed = int(experiment.get("train_seed", experiment.get("seed", 0)))
        if train_seed != int(experiment.get("seed", train_seed)):
            raise ValueError("experiment.train_seed and experiment.seed must match; split_seed is configured separately.")
    if dataset_type == "deepsense6g":
        domains = data.get("domains")
        if domains:
            if not isinstance(domains, list) or {
                item.get("scene") for item in domains if isinstance(item, dict)
            } != DEEPSENSE6G_SCENES or len(domains) != len(DEEPSENSE6G_SCENES):
                raise ValueError(f"Pooled DeepSense6G domains must contain exactly scenes {sorted(DEEPSENSE6G_SCENES)}.")
        else:
            scene = data.get("scene")
            if type(scene) is not int or scene not in DEEPSENSE6G_SCENES:
                raise ValueError(f"DeepSense6G scene must be one of {sorted(DEEPSENSE6G_SCENES)}, got {scene!r}.")
        if data.get("gps_normalize") is not True:
            raise ValueError("DeepSense6G requires train-fitted GPS normalization.")
        if int(data.get("num_pred", 0)) > 3:
            raise ValueError("DeepSense6G supports at most three future_beam horizons.")
        if int(model.get("num_classes", 0)) != 64:
            raise ValueError("DeepSense6G future_beam labels require model.primary.num_classes=64.")


def _validate_pcpf_config(cfg: dict[str, Any], model: dict[str, Any], data: dict[str, Any]) -> None:
    from kd_sensing.losses.pcpf_temporal_risk_config import pcpf_temporal_risk_config
    from kd_sensing.models.pcpf_temporal_risk import validate_pcpf_model_config

    if str(data.get("type", "")).strip().lower() != "mmw":
        raise ValueError("PCPF-T is scoped to the MMW dataset.")
    validate_pcpf_model_config(model, data)
    loss = pcpf_temporal_risk_config(cfg)
    experiment = cfg.get("experiment", {})
    if experiment.get("claim_ineligible") is not True:
        raise ValueError("PCPF-T requires experiment.claim_ineligible=true.")
    training = cfg.get("training", {})
    final_test = training.get("final_test")
    final_test_enabled = final_test if isinstance(final_test, bool) else (final_test or {}).get("enabled", True)
    if bool(final_test_enabled):
        raise ValueError("PCPF-T requires training.final_test.enabled=false.")
    stage = loss["training_stage"]
    initialization = training.get("initialization_checkpoint")
    expected_source = {
        "stage2_risk": "stage1_expert",
        "stage3_fusion": "stage2_risk",
    }.get(stage)
    if expected_source is not None:
        if not isinstance(initialization, dict):
            raise ValueError(f"{stage} requires training.initialization_checkpoint.")
        if initialization.get("role") != "validation_best":
            raise ValueError(f"{stage} requires a validation_best initialization checkpoint.")
        if initialization.get("expected_source_training_stage") != expected_source:
            raise ValueError(f"{stage} requires expected_source_training_stage={expected_source!r}.")
    if stage == "stage1_expert" and initialization not in (None, False):
        raise ValueError("stage1_expert must start fresh.")
    if stage in {"stage2_risk", "stage3_fusion"} and not loss["stage_preparation"]["enabled"]:
        raise ValueError(f"{stage} requires train-only stage_preparation.enabled=true.")
    if stage == "stage3_fusion":
        gate = training.get("pcpf_stage2_gate")
        if not isinstance(gate, dict):
            raise ValueError(f"{stage} requires training.pcpf_stage2_gate.")
        unknown = sorted(set(gate) - {"report_path", "sha256", "stage2_gate_passed"})
        if unknown:
            raise ValueError(f"training.pcpf_stage2_gate contains unsupported fields: {unknown}.")
        digest = str(gate.get("sha256", "")).strip().lower()
        if not gate.get("report_path") or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("training.pcpf_stage2_gate requires report_path and a SHA256 digest.")
        if gate.get("stage2_gate_passed") is not True:
            raise ValueError(f"{stage} refuses a Stage 2 gate that did not pass.")
