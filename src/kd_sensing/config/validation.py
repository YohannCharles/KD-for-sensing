from typing import Any

from kd_sensing.modalities import normalize_modalities, resolve_image_profile, validate_image_profile_size


RETAINED_MODALITIES = ("image", "radar", "gps", "lidar")
RETAINED_MODELS = {"u_mask_beam_jepa", "modular_sequence", "four_modal_topology_predictor"}
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
    if str(model.get("type", "")) == "four_modal_topology_predictor":
        _validate_topology_predictor_config(cfg, model, data)
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
        evidence = cfg.get("deepsense6g_secondary_evidence")
        if evidence is not None:
            if not isinstance(evidence, dict) or evidence.get("protocol_id") != "deepsense6g_twc_secondary_v1":
                raise ValueError("DeepSense6G secondary evidence requires the fixed filtered protocol.")
            pooled = evidence.get("pooled_dataset", {})
            if pooled.get("train_row_count") != 13240 or pooled.get("test_row_count") != 4090:
                raise ValueError("DeepSense6G secondary evidence counts do not match the fixed protocol.")
            if evidence.get("test_policy") != "one_shot_after_fixed_40_epochs":
                raise ValueError("DeepSense6G secondary evidence requires the frozen one-shot test policy.")


def _validate_topology_predictor_config(cfg: dict[str, Any], model: dict[str, Any], data: dict[str, Any]) -> None:
    from kd_sensing.losses.four_modal_topology import four_modal_topology_config
    from kd_sensing.models.four_modal_topology_predictor import validate_topology_predictor_model_config

    dataset_type = str(data.get("type", "")).strip().lower()
    if dataset_type not in {"mmw", "deepsense6g"}:
        raise ValueError("The four-modal topology predictor supports only MMW or DeepSense6G.")
    validate_topology_predictor_model_config(model, data)
    topology_loss = four_modal_topology_config(cfg)
    experiment = cfg.get("experiment", {})
    if experiment.get("claim_ineligible") is not True:
        raise ValueError("The topology predictor requires experiment.claim_ineligible=true.")
    training = cfg.get("training", {})
    final_test = training.get("final_test")
    final_test_enabled = final_test if isinstance(final_test, bool) else (final_test or {}).get("enabled", True)
    if dataset_type == "mmw" and bool(final_test_enabled):
        raise ValueError("The topology predictor requires training.final_test.enabled=false.")
    if dataset_type == "deepsense6g":
        if str(model.get("prototype_topology_id", "")).strip().lower() != "linear_index_v1":
            raise ValueError("DeepSense6G topology transfer requires model topology linear_index_v1.")
        if topology_loss["prototype_topology"]["id"] != "linear_index_v1":
            raise ValueError("DeepSense6G topology transfer requires loss topology linear_index_v1.")
        if str(model.get("fusion_mode", "")) != "masked_feature_mlp":
            raise ValueError("DeepSense6G topology transfer requires masked_feature_mlp fusion.")
        if (
            topology_loss["unimodal_soft_weight"] != 0.0
            or not topology_loss["use_beam_prototype_alignment"]
            or topology_loss["lambda_proto"] != 0.1
            or topology_loss["lambda_modality_proto"] != 0.0
        ):
            raise ValueError("DeepSense6G topology transfer requires the frozen Prototype-only loss.")
        if str(training.get("checkpoint_selection", "last")).strip().lower() != "last":
            raise ValueError("DeepSense6G without a compatible validation split requires last checkpoint selection.")
        if training.get("final_test_missing_matrix") is not True:
            raise ValueError("DeepSense6G topology transfer requires the frozen final-test missing matrix.")
        if int(training.get("epochs", 0)) != 40 or int(training.get("max_epochs", 0)) != 40:
            raise ValueError("DeepSense6G topology transfer requires the frozen 40-epoch budget.")
        if "data_protocol" in cfg:
            raise ValueError("DeepSense6G topology transfer cannot carry the MMW data protocol.")
    retired = sorted(set(training) & {"initialization_checkpoint"})
    if retired:
        raise ValueError(f"Single-stage topology training rejects retired fields: {retired}.")
