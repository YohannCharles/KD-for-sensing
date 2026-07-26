"""Protocol-bound Stage A workflow for frozen-U0 decision adapters."""

from __future__ import annotations

import hashlib
import json
import math
import random
import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kd_sensing.baselines import full_pool_common as _common
from kd_sensing.config.defaults import DEFAULT_CONFIG
from kd_sensing.config.io import deep_merge, dump_config, load_config_source
from kd_sensing.config.normalization import normalize_loaded_config
from kd_sensing.config.validation import validate_loaded_config
from kd_sensing.data.mmw.clean_protocol import (
    audit_clean_inner_protocol,
    load_clean_inner_protocol,
    validate_clean_inner_protocol,
    write_clean_inner_protocol,
)
from kd_sensing.data.mmw.full_pool_protocol import (
    FULL_POOL_PROTOCOL_MODE,
    load_full_pool_protocol,
)
from kd_sensing.data.mmw.protocol import validate_mmw_config_protocol
from kd_sensing.engine.batch import prepare_fusion_inputs, prepare_labels
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts
from kd_sensing.engine.optim import build_model
from kd_sensing.engine.trainer_runtime_helpers import training_loss_early_stop_state
from kd_sensing.losses.beam_prototype_alignment import make_soft_beam_labels
from kd_sensing.models.missing_decision_adapter import FrozenU0DecisionAdapter, MissingDecisionAdapter
from kd_sensing.modalities import MODALITY_ORDER
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata, validate_evaluation_checkpoint_route
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_torch_payload,
    publish_checkpoint,
    validate_checkpoint_publication,
)
from kd_sensing.utils.seed import set_seed


EXPECTED_U0_SHA256 = "21fa02f473a5acee6edb3a4ea42b1b9786c29cb3e5177f2b9ea28bc4c127a9f6"
EXPERIMENT_SEED = 1
PROTOTYPE_CACHE_BATCH_SIZE = 128
PROTOTYPE_CACHE_NUM_WORKERS = 8
ADAPTER_LOSS_PROFILES = {
    "cross_entropy": {
        "hard_ce_weight": 1.0,
        "soft_ce_weight": 0.0,
        "soft_label_sigma": None,
        "circular": True,
        "delta_logit_weight": 1e-4,
    },
    "adba_surrogate": {
        "hard_ce_weight": 0.5,
        "soft_ce_weight": 0.5,
        "soft_label_sigma": 2.0,
        "circular": True,
        "delta_logit_weight": 1e-4,
    },
}
NON_FULL_MASKS = tuple(tuple(int(bit) for bit in f"{value:04b}") for value in range(1, 15))
MASKS = (
    ("full", "Full", (1, 1, 1, 1)),
    ("image_only", "Image Only", (1, 0, 0, 0)),
    ("radar_only", "Radar Only", (0, 1, 0, 0)),
    ("gps_only", "GPS Only", (0, 0, 1, 0)),
    ("lidar_only", "LiDAR Only", (0, 0, 0, 1)),
    ("image_radar", "Image+Radar", (1, 1, 0, 0)),
    ("image_gps", "Image+GPS", (1, 0, 1, 0)),
    ("image_lidar", "Image+LiDAR", (1, 0, 0, 1)),
    ("radar_gps", "Radar+GPS", (0, 1, 1, 0)),
    ("radar_lidar", "Radar+LiDAR", (0, 1, 0, 1)),
    ("gps_lidar", "GPS+LiDAR", (0, 0, 1, 1)),
    ("no_image", "No Image", (0, 1, 1, 1)),
    ("no_radar", "No Radar", (1, 0, 1, 1)),
    ("no_gps", "No GPS", (1, 1, 0, 1)),
    ("no_lidar", "No LiDAR", (1, 1, 1, 0)),
)


@dataclass(frozen=True)
class Experiment:
    key: str
    gpu: int
    run_name: str
    variant: str | None
    rank: int = 8
    shuffled: bool = False


EXPERIMENTS = {
    item.key: item
    for item in (
        Experiment("a0", 0, "gpu0_a0_frozen_u0_seed1", None),
        Experiment("a1", 1, "gpu1_a1_mask_bias_seed1", "mask_bias"),
        Experiment("a2", 2, "gpu2_a2_mask_lora_r4_seed1", "mask_lora", 4),
        Experiment("a3", 3, "gpu3_a3_mask_lora_r8_seed1", "mask_lora", 8),
        Experiment("a4", 4, "gpu4_a4_mask_lora_r16_seed1", "mask_lora", 16),
        Experiment("a5", 5, "gpu5_a5_proto_lora_r8_seed1", "proto_lora", 8),
        Experiment("a6", 6, "gpu6_a6_proto_uncertainty_lora_r8_seed1", "proto_uncertainty_lora", 8),
        Experiment("a7", 7, "gpu7_a7_shuffled_proto_control_seed1", "proto_uncertainty_lora", 8, True),
        Experiment("global_bias", 0, "gpu0_global_bias_seed1", "global_bias"),
        Experiment("mask_lookup", 4, "gpu4_mask_lookup_seed1", "mask_lookup"),
        Experiment("factorized_bias", 7, "gpu7_factorized_bias_seed1", "factorized_bias"),
        Experiment("factorized_all_seen", 4, "gpu4_factorized_all_seen_seed1", "factorized_bias"),
        Experiment("circular_transport", 0, "gpu0_circular_transport_seed1", "circular_transport"),
    )
}


sha256_json = _common.sha256_json


def write_json(path: Path, payload: Any) -> None:
    """Write a Stage A artifact with sorted keys, as every existing run did."""
    _common.write_json(path, payload, sort_keys=True)


def preflight(
    config_path: Path,
    checkpoint_path: Path,
    expected_sha256: str = EXPECTED_U0_SHA256,
    *,
    migration_root: Path | None = None,
) -> tuple[dict, dict]:
    cfg = load_u0_artifact_config(config_path, migration_root=migration_root)
    config_audit = validate_mmw_config_protocol(cfg)
    audit = config_audit
    if audit.get("status") != "passed" or audit.get("outer_test_accessed") is not False:
        raise ValueError("MMW protocol audit must pass with outer_test_accessed=false.")
    if cfg.get("training", {}).get("final_test", {}).get("enabled", True):
        raise ValueError("Adapter workflow requires training.final_test.enabled=false.")
    actual, size = checkpoint_file_digest(checkpoint_path)
    if actual != expected_sha256:
        raise ValueError(f"U0 checkpoint SHA256 mismatch: expected={expected_sha256}, actual={actual}.")
    payload = load_torch_payload(checkpoint_path, map_location="cpu")
    validate_checkpoint_publication(checkpoint_path, payload=payload)
    validate_evaluation_checkpoint_route(load_checkpoint_metadata(checkpoint_path))
    model, structure_audit = load_frozen_u0(cfg, checkpoint_path, torch.device("cpu"))
    del model
    return cfg, {
        "status": "passed",
        "outer_test_accessed": False,
        "protocol": audit,
        "config_protocol": config_audit,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": actual,
        "checkpoint_size_bytes": size,
        "strict_model_load": True,
        "u0_structure_audit": structure_audit,
    }


def load_u0_artifact_config(config_path: Path, *, migration_root: Path | None = None) -> dict[str, Any]:
    """Load the named U0 artifact while refusing to revive its disabled retired sections."""
    raw = copy.deepcopy(load_config_source(config_path).data)
    stripped = []
    for name in ("bcacl", "cmsbl"):
        section = raw.get(name)
        if section is not None:
            if not isinstance(section, Mapping) or section.get("enabled") is not False:
                raise ValueError(f"Retired section {name!r} is only accepted when explicitly disabled.")
            raw.pop(name)
            stripped.append(name)
    for name in ("clean_recovery", "runtime"):
        raw.pop(name, None)
    primary = raw.get("model", {}).get("primary", {})
    capacity_reference_mode = primary.pop("capacity_reference_mode", None)
    if capacity_reference_mode not in (None, False):
        raise ValueError("Legacy capacity_reference_mode is only accepted when explicitly false.")
    stripped_model_fields = []
    if capacity_reference_mode is False:
        stripped_model_fields.append("model.primary.capacity_reference_mode")
    cfg = deep_merge(copy.deepcopy(DEFAULT_CONFIG), raw)
    protocol_cfg = cfg.get("data_protocol")
    if not isinstance(protocol_cfg, dict) or not protocol_cfg.get("path"):
        raise ValueError("Named U0 artifact lacks its clean data protocol path.")
    protocol_path = Path(protocol_cfg["path"])
    if protocol_cfg.get("mode") == FULL_POOL_PROTOCOL_MODE:
        protocol = load_full_pool_protocol(protocol_path)
    else:
        try:
            protocol = load_clean_inner_protocol(protocol_path)
        except ValueError as exc:
            if migration_root is None or "inner_train and inner_validation roles" not in str(exc):
                raise
            legacy = copy.deepcopy(load_config_source(protocol_path).data)
            legacy.pop("protocol_fingerprint", None)
            legacy.update(train_role="inner_train", validation_role="inner_validation")
            protocol = validate_clean_inner_protocol(legacy)
            migrated_path = migration_root / "clean_inner_development_protocol_current.yaml"
            write_clean_inner_protocol(protocol, migrated_path)
            migrated_audit = audit_clean_inner_protocol(migrated_path, fail_closed=True)
            audit_json = migration_root / "clean_split_audit_current.json"
            write_json(audit_json, migrated_audit)
            (migration_root / "clean_split_audit_current.md").write_text(
                "# Clean Split Isolation Audit\n\n"
                f"- Status: `{migrated_audit['status']}`\n"
                f"- Train samples: {migrated_audit['train_sample_count']}\n"
                f"- Validation samples: {migrated_audit['validation_sample_count']}\n"
                "- Outer test accessed: `false`\n",
                encoding="utf-8",
            )
            write_json(
                migration_root / "protocol_migration.json",
                {
                    "source_path": str(protocol_path.resolve()),
                    "source_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                    "change": "added explicit inner_train/inner_validation role fields",
                    "migrated_path": str(migrated_path.resolve()),
                    "migrated_fingerprint": protocol["protocol_fingerprint"],
                    "outer_test_accessed": False,
                },
            )
            protocol_cfg["path"] = str(migrated_path.resolve())
            protocol_cfg["audit_report"] = str(audit_json.resolve())
    protocol_cfg.update(
        mode=protocol.get("mode", "clean_inner_development"),
        protocol_id=protocol["protocol_id"],
        protocol_fingerprint=protocol["protocol_fingerprint"],
        train_role=protocol.get("train_role", "inner_train"),
        validation_role=protocol.get("validation_role", "inner_validation"),
    )
    normalize_loaded_config(cfg)
    validate_loaded_config(cfg)
    input_audit = cfg.setdefault("prototype_adapter_input_audit", {})
    input_audit["stripped_disabled_retired_sections"] = stripped
    input_audit["stripped_disabled_legacy_model_fields"] = stripped_model_fields
    return cfg


def dataset_sample_ids(dataset: Any) -> list[str]:
    result = []
    for leaf, indices in leaf_datasets_with_indices(dataset):
        rows = leaf.samples.rows or []
        for index in indices:
            row = rows[index]
            source = str(row.get("sample_id") or row.get("target_sample_id") or f"{leaf.scene_slug}:{index}")
            result.append(f"mmw:{leaf.condition}:{leaf.scene_slug}:{leaf.split}:{source}")
    if len(result) != len(set(result)):
        raise ValueError("Clean split stable_sample_id values must be unique.")
    return result


def checkpoint_normalization_overrides(checkpoint_path: Path) -> dict[str, Any]:
    """Reuse the immutable train-fitted U0 scaler instead of refitting per Adapter."""
    overrides = load_normalization_artifacts(load_checkpoint_metadata(checkpoint_path))
    if "gps_scaler" not in overrides:
        raise ValueError("Published Full-data U0 checkpoint lacks its train-fitted GPS scaler artifact.")
    return overrides


def generate_mask_schedule(
    sample_ids: list[str],
    *,
    epochs: int = 20,
    seed: int = 1,
    allowed_masks: tuple[tuple[int, ...], ...] | None = None,
) -> dict[str, Any]:
    masks = NON_FULL_MASKS if allowed_masks is None else tuple(tuple(int(bit) for bit in mask) for mask in allowed_masks)
    if not masks or len(masks) != len(set(masks)) or any(mask not in NON_FULL_MASKS for mask in masks):
        raise ValueError("allowed_masks must be a non-empty unique subset of the 14 non-Full masks.")
    ordered = sorted(sample_ids)
    schedules: list[list[int]] = []
    for epoch in range(int(epochs)):
        positions = list(range(len(ordered)))
        random.Random((int(seed) << 16) + epoch).shuffle(positions)
        values = [0] * len(ordered)
        offset = epoch % len(masks)
        for rank, position in enumerate(positions):
            values[position] = (rank + offset) % len(masks)
        schedules.append(values)
    payload = {
        "schema_version": 1,
        "seed": int(seed),
        "epochs": int(epochs),
        "modality_order": list(MODALITY_ORDER),
        "masks": [list(mask) for mask in masks],
        "sample_ids": ordered,
        "mask_indices_by_epoch": schedules,
    }
    payload["schedule_sha256"] = sha256_json(payload)
    return payload


def stratified_mask_folds(*, seed: int = 1, fold_count: int = 4) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if int(fold_count) != 4:
        raise ValueError("The preregistered mask split requires exactly four folds.")
    folds: list[list[tuple[int, ...]]] = [[] for _ in range(int(fold_count))]
    for cardinality in (1, 2, 3):
        group = [mask for mask in NON_FULL_MASKS if sum(mask) == cardinality]
        random.Random((int(seed) << 16) + cardinality).shuffle(group)
        for index, mask in enumerate(group):
            folds[index % int(fold_count)].append(mask)
    return tuple(tuple(fold) for fold in folds)


def schedule_masks(schedule: Mapping[str, Any], epoch: int, sample_ids: list[str], device: torch.device) -> torch.Tensor:
    positions = {value: index for index, value in enumerate(schedule["sample_ids"])}
    try:
        indices = [schedule["mask_indices_by_epoch"][epoch][positions[value]] for value in sample_ids]
    except (KeyError, IndexError) as exc:
        raise ValueError("Batch sample ids are not covered by the fixed mask schedule.") from exc
    masks = [schedule["masks"][index] for index in indices]
    return torch.tensor(masks, dtype=torch.bool, device=device)


def split_permutation(sample_ids: list[str], *, seed: int, split: str) -> dict[str, str]:
    ordered = sorted(sample_ids)
    shuffled = ordered.copy()
    derived = int.from_bytes(hashlib.sha256(f"{seed}:{split}".encode()).digest()[:8], "big")
    random.Random(derived).shuffle(shuffled)
    if len(ordered) > 1 and shuffled == ordered:
        shuffled = shuffled[1:] + shuffled[:1]
    return dict(zip(ordered, shuffled))


def prepare_stage(config_path: Path, checkpoint_path: Path, protocol_dir: Path) -> dict[str, Any]:
    cfg, audit = preflight(config_path, checkpoint_path, migration_root=protocol_dir)
    resolved_config_path = protocol_dir / "resolved_u0_adapter_config.yaml"
    dump_config(cfg, resolved_config_path)
    loaders = build_dataloaders(
        cfg,
        normalization_overrides=checkpoint_normalization_overrides(checkpoint_path),
    )
    ids = dataset_sample_ids(loaders["train"].dataset)
    schedule = generate_mask_schedule(ids)
    schedule_path = protocol_dir / "mask_schedule_seed1.json"
    write_json(schedule_path, schedule)
    write_json(protocol_dir / "stage_a_configurations.json", [asdict(value) for value in EXPERIMENTS.values()])
    write_json(protocol_dir / "preflight.json", audit | {"mask_schedule_sha256": schedule["schedule_sha256"]})
    return {
        "schedule_path": str(schedule_path),
        "schedule_sha256": schedule["schedule_sha256"],
        "resolved_config_path": str(resolved_config_path),
        **audit,
    }


def _batch_ids(batch: Mapping[str, Any], field: str = "stable_sample_id") -> list[str]:
    metadata = batch.get("metadata")
    if not isinstance(metadata, Mapping) or field not in metadata:
        if field == "stable_sample_id" and "sample_id" in batch:
            return [str(value) for value in batch["sample_id"]]
        return ["" for _ in range(len(batch["target_beam"]))]
    values = metadata[field]
    return [str(value) for value in values]


def _inputs(batch: Mapping[str, Any], cfg: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return prepare_fusion_inputs(
        dict(batch), seq_length=int(cfg["model"]["seq_length"]), device=device, modalities=MODALITY_ORDER
    )


def _amp(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _base_forward(model, batch, mask, cfg, device) -> dict[str, Any]:
    with torch.no_grad(), _amp(device):
        return model(**_inputs(batch, cfg, device), missing_mask=mask)


def _state_to_cpu(state: Mapping[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return {key: value[row].detach().cpu() for key, value in state.items() if torch.is_tensor(value)}


def _stack_state(cache: Mapping[str, Mapping[str, torch.Tensor]], ids: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    keys = next(iter(cache.values())).keys()
    return {key: torch.stack([cache[value][key] for value in ids]).to(device) for key in keys}


def _sequential(loader) -> DataLoader:
    kwargs = {
        "batch_size": min(PROTOTYPE_CACHE_BATCH_SIZE, len(loader.dataset)),
        "shuffle": False,
        "num_workers": PROTOTYPE_CACHE_NUM_WORKERS,
        "pin_memory": loader.pin_memory,
        "collate_fn": loader.collate_fn,
        "drop_last": False,
        "worker_init_fn": loader.worker_init_fn,
        "persistent_workers": False,
        "prefetch_factor": loader.prefetch_factor or 2,
    }
    return DataLoader(loader.dataset, **kwargs)


def state_cache(model, loader, cfg, device, mask_for_ids) -> dict[str, dict[str, torch.Tensor]]:
    cache: dict[str, dict[str, torch.Tensor]] = {}
    for batch in _sequential(loader):
        ids = _batch_ids(batch)
        mask = mask_for_ids(ids)
        output = _base_forward(model, batch, mask, cfg, device)
        state = output.get("prototype_state")
        if not isinstance(state, Mapping):
            raise ValueError("U0 prototype state is unavailable.")
        for row, sample_id in enumerate(ids):
            cache[sample_id] = _state_to_cpu(state, row)
    return cache


def condition_statistics(cache: Mapping[str, Mapping[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor(
        [
            [
                float(state["entropy"]),
                float(state["nearest_distance"]),
                float(state["distance_margin"]),
                float(state["restoration_residual_norm"]),
            ]
            for state in cache.values()
        ],
        dtype=torch.float32,
    )
    return values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-6)


def state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        cpu_value = value.detach().cpu().contiguous()
        digest.update(str(cpu_value.dtype).encode())
        digest.update(str(tuple(cpu_value.shape)).encode())
        digest.update(cpu_value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_frozen_u0(cfg: dict, checkpoint_path: Path, device: torch.device):
    model = build_model(cfg["model"]["primary"])
    payload = load_torch_payload(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("U0 checkpoint did not load strictly.")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bank = getattr(model, "prototype_bank", None)
    if getattr(model, "head_type", None) != "prototype" or bank is None:
        raise ValueError("Selected U0 does not use its Beam prototype bank at inference.")
    audit = {
        "prototype_restoration_enabled": False,
        "prototype_type": type(bank).__name__,
        "prototype_count": int(bank.num_beams),
        "prototype_dimension": int(bank.d_model),
        "prototype_update_mode": "frozen_for_adapter_stage",
        "prototype_used_at_inference": True,
        "note": "Current U0 uses a prototype classification bank; it has no separate restoration module.",
    }
    return model, audit


def _adapter(experiment: Experiment, model, device: torch.device):
    if experiment.variant is None:
        return None
    return MissingDecisionAdapter(
        model.d_model,
        num_classes=model.num_classes,
        rank=experiment.rank,
        variant=experiment.variant,
        prototype_dim=model.num_classes,
    ).to(device)


def adapter_training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    delta_logits: torch.Tensor,
    *,
    loss_profile: str = "cross_entropy",
) -> torch.Tensor:
    if loss_profile not in ADAPTER_LOSS_PROFILES:
        raise ValueError(f"Unknown Adapter loss profile: {loss_profile}")
    profile = ADAPTER_LOSS_PROFILES[loss_profile]
    scores = logits.float()
    hard_ce = F.cross_entropy(scores, labels)
    task_loss = float(profile["hard_ce_weight"]) * hard_ce
    if float(profile["soft_ce_weight"]):
        soft_target = make_soft_beam_labels(
            labels,
            scores.shape[-1],
            float(profile["soft_label_sigma"]),
            circular=bool(profile["circular"]),
        ).to(scores)
        soft_ce = -(soft_target * F.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
        task_loss = task_loss + float(profile["soft_ce_weight"]) * soft_ce
    regularization = float(profile["delta_logit_weight"]) * delta_logits.float().pow(2).sum(dim=1).mean()
    return task_loss + regularization


def train_adapter(
    wrapper,
    experiment,
    train_loader,
    cfg,
    schedule,
    device,
    epochs=20,
    early_stopping: dict[str, Any] | None = None,
    loss_profile: str = "cross_entropy",
) -> list[dict[str, Any]]:
    adapter = wrapper.adapter
    if adapter is None:
        return []
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, len(train_loader))

    def factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    permutation = split_permutation(schedule["sample_ids"], seed=1, split="train")
    history = []
    for epoch in range(epochs):
        shuffled_cache = None
        if experiment.shuffled:
            shuffled_cache = state_cache(
                wrapper.base_model,
                train_loader,
                cfg,
                device,
                lambda ids: schedule_masks(schedule, epoch, ids, device),
            )
        wrapper.train(True)
        loss_sum = count = 0
        for batch in train_loader:
            ids = _batch_ids(batch)
            mask = schedule_masks(schedule, epoch, ids, device)
            override = None
            if shuffled_cache is not None:
                override = _stack_state(shuffled_cache, [permutation[value] for value in ids], device)
            labels = prepare_labels(dict(batch), num_pred=1, device=device)[:, 0]
            optimizer.zero_grad(set_to_none=True)
            with _amp(device):
                output = wrapper(**_inputs(batch, cfg, device), missing_mask=mask, adapter_proto_state=override)
                logits = output["logits"][:, 0, :]
                delta = output["delta_logits"]
                loss = adapter_training_loss(logits, labels, delta, loss_profile=loss_profile)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * labels.numel()
            count += labels.numel()
        epoch_row = {"epoch": epoch + 1, "loss": loss_sum / count, "lr": optimizer.param_groups[0]["lr"]}
        history.append(epoch_row)
        stop_state = training_loss_early_stop_state(
            [float(row["loss"]) for row in history],
            early_stopping,
        )
        stop_state["monitor"] = "loss"
        epoch_row["early_stopping"] = stop_state
        print(json.dumps({"event": "adapter_epoch", "experiment": experiment.key, **epoch_row}), flush=True)
        if stop_state["should_stop"]:
            break
    return history


def numpy_metrics(logits: np.ndarray, target: np.ndarray, *, delta: float = 5.0) -> dict[str, float]:
    order = np.argsort(-logits, axis=1, kind="stable")
    prediction = order[:, 0]
    distance = np.abs(prediction - target)
    distance = np.minimum(distance, logits.shape[1] - distance)
    shifted = logits - logits.max(axis=1, keepdims=True)
    loss = -shifted[np.arange(len(target)), target] + np.log(np.exp(shifted).sum(axis=1))
    top3_distance = np.abs(order[:, :3] - target[:, None])
    top3_distance = np.minimum(top3_distance, logits.shape[1] - top3_distance)
    progressive = 1.0 - np.minimum.accumulate(np.minimum(top3_distance / delta, 1.0), axis=1)
    return {
        "top1": float((prediction == target).mean()),
        "top3": float((order[:, :3] == target[:, None]).any(axis=1).mean()),
        "top5": float((order[:, :5] == target[:, None]).any(axis=1).mean()),
        "within3": float((distance <= 3).mean()),
        "mae": float(distance.mean()),
        "adba": float(progressive.mean()),
        "loss": float(loss.mean()),
    }


def evaluate(wrapper, experiment, validation_loader, cfg, device, output_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    permutation = split_permutation(dataset_sample_ids(validation_loader.dataset), seed=1, split="validation")
    before = state_digest(wrapper)
    wrapper.eval()
    for key, label, raw_mask in MASKS:
        fixed = lambda ids, raw=raw_mask: torch.tensor([raw] * len(ids), dtype=torch.bool, device=device)
        shuffled_cache = state_cache(wrapper.base_model, validation_loader, cfg, device, fixed) if experiment.shuffled and key != "full" else None
        collected: dict[str, list] = {name: [] for name in (
            "sample_id", "target_sample_id", "domain", "weather", "scenario", "ground_truth", "base_logits", "new_logits", "delta_logits",
            "base_prediction", "new_prediction", "correct_before", "correct_after", "nearest_id",
            "nearest_distance", "distance_margin", "prototype_entropy", "restoration_residual_norm", "alpha",
        )}
        with torch.no_grad():
            for batch in validation_loader:
                ids = _batch_ids(batch)
                mask = fixed(ids)
                override = None
                if shuffled_cache is not None:
                    override = _stack_state(shuffled_cache, [permutation[value] for value in ids], device)
                with _amp(device):
                    output = wrapper(**_inputs(batch, cfg, device), missing_mask=mask, adapter_proto_state=override)
                target = prepare_labels(dict(batch), num_pred=1, device=device)[:, 0]
                base = output["base_logits"][:, 0].float().cpu().numpy()
                new = output["logits"][:, 0].float().cpu().numpy()
                delta_logits = output["delta_logits"].float().cpu().numpy()
                base_pred, new_pred = base.argmax(1), new.argmax(1)
                state = output["prototype_state"]
                collected["sample_id"].extend(ids)
                collected["target_sample_id"].extend(_batch_ids(batch, "target_sample_id"))
                weather = _batch_ids(batch, "condition")
                scenario = _batch_ids(batch, "scenario")
                collected["weather"].extend(weather)
                collected["scenario"].extend(scenario)
                collected["domain"].extend(f"{item}/{scene}" for item, scene in zip(weather, scenario))
                collected["ground_truth"].extend(target.cpu().numpy())
                collected["base_logits"].extend(base)
                collected["new_logits"].extend(new)
                collected["delta_logits"].extend(delta_logits)
                collected["base_prediction"].extend(base_pred)
                collected["new_prediction"].extend(new_pred)
                collected["correct_before"].extend(base_pred == target.cpu().numpy())
                collected["correct_after"].extend(new_pred == target.cpu().numpy())
                collected["nearest_id"].extend(state["nearest_id"].cpu().numpy())
                collected["nearest_distance"].extend(state["nearest_distance"].float().cpu().numpy())
                collected["distance_margin"].extend(state["distance_margin"].float().cpu().numpy())
                collected["prototype_entropy"].extend(state["entropy"].float().cpu().numpy())
                collected["restoration_residual_norm"].extend(state["restoration_residual_norm"].float().cpu().numpy())
                collected["alpha"].extend(output["adapter_alpha"].float().cpu().numpy())
        arrays = {name: np.asarray(value) for name, value in collected.items()}
        arrays["mask"] = np.tile(np.asarray(raw_mask, dtype=np.int8), (len(arrays["ground_truth"]), 1))
        arrays["delta_logit_norm"] = np.linalg.norm(arrays["delta_logits"], axis=1)
        arrays["prediction"] = arrays["new_prediction"]
        arrays["logits"] = arrays["new_logits"]
        arrays["condition_summary"] = np.column_stack(
            [
                arrays["nearest_id"],
                arrays["prototype_entropy"],
                arrays["nearest_distance"],
                arrays["distance_margin"],
                arrays["restoration_residual_norm"],
                arrays["delta_logit_norm"],
            ]
        )
        np.savez_compressed(predictions_dir / f"{key}.npz", **arrays)
        new_metrics = numpy_metrics(arrays["new_logits"], arrays["ground_truth"])
        base_metrics = numpy_metrics(arrays["base_logits"], arrays["ground_truth"])
        norms = np.linalg.norm(arrays["delta_logits"], axis=1)
        rows.append({
            "key": key, "label": label, "mask": list(raw_mask), "sample_count": len(norms),
            "base": base_metrics, "new": new_metrics,
            "diagnostics": {
                "mean_delta_logit_norm": float(norms.mean()), "median_delta_logit_norm": float(np.median(norms)),
                "p95_delta_logit_norm": float(np.quantile(norms, 0.95)),
                "mean_alpha": float(arrays["alpha"].mean()) if arrays["alpha"].size else 0.0,
                "alpha_std": float(arrays["alpha"].std()) if arrays["alpha"].size else 0.0,
            },
        })
        print(json.dumps({"event": "adapter_evaluation_mask", "experiment": experiment.key, "mask": key}), flush=True)
    if state_digest(wrapper) != before:
        raise RuntimeError("Validation mutated U0, prototype, Adapter, or normalization state.")
    full = np.load(predictions_dir / "full.npz")
    difference = np.abs(full["base_logits"] - full["new_logits"])
    equivalence = {
        "sample_count": int(len(full["ground_truth"])),
        "max_abs_logit_diff": float(difference.max()), "mean_abs_logit_diff": float(difference.mean()),
        "argmax_mismatch_count": int((full["base_prediction"] != full["new_prediction"]).sum()),
        "top1_difference": float((full["correct_after"].mean() - full["correct_before"].mean())),
        "sha256_base_logits": hashlib.sha256(full["base_logits"].tobytes()).hexdigest(),
        "sha256_new_logits": hashlib.sha256(full["new_logits"].tobytes()).hexdigest(),
    }
    if equivalence["max_abs_logit_diff"] > 1e-7 or equivalence["argmax_mismatch_count"] or equivalence["top1_difference"]:
        raise RuntimeError("Full inference path equivalence failed.")
    write_json(output_dir / "full_equivalence.json", equivalence)
    return rows, equivalence


def aggregate(mask_rows: list[dict]) -> dict[str, Any]:
    by_key = {row["key"]: row for row in mask_rows}
    full = by_key["full"]["new"]["top1"]
    groups = {count: [row for row in mask_rows if sum(row["mask"]) == count] for count in (1, 2, 3)}
    summary: dict[str, Any] = {"full": by_key["full"]["new"]}
    names = {1: "single", 2: "double", 3: "triple"}
    for count, rows in groups.items():
        summary[f"{names[count]}_macro"] = {metric: float(np.mean([row["new"][metric] for row in rows])) for metric in rows[0]["new"]}
        summary[f"{names[count]}_worst_top1"] = min(row["new"]["top1"] for row in rows)
    missing = [row for row in mask_rows if row["key"] != "full"]
    summary["all14_macro"] = {metric: float(np.mean([row["new"][metric] for row in missing])) for metric in missing[0]["new"]}
    summary["all14_worst_top1"] = min(row["new"]["top1"] for row in missing)
    retention_keys = ("radar_gps", "radar_only", "gps_only", "no_image", "no_lidar")
    retention = {key: by_key[key]["new"]["top1"] / full for key in retention_keys}
    retention["all14_macro"] = summary["all14_macro"]["top1"] / full
    retention["single_worst"] = summary["single_worst_top1"] / full
    spa = float(np.mean([by_key[key]["new"]["top1"] for key in ("radar_only", "gps_only", "radar_gps")]))
    return {"aggregates": summary, "retention": retention, "spa_macro": spa}


def run_experiment(
    experiment_key: str,
    config_path: Path,
    checkpoint_path: Path,
    schedule_path: Path,
    output_dir: Path,
    *,
    epochs: int = 20,
    expected_u0_sha256: str = EXPECTED_U0_SHA256,
    early_stopping: dict[str, Any] | None = None,
    loss_profile: str = "cross_entropy",
) -> dict[str, Any]:
    experiment = EXPERIMENTS[experiment_key]
    if loss_profile not in ADAPTER_LOSS_PROFILES:
        raise ValueError(f"Unknown Adapter loss profile: {loss_profile}")
    set_seed(EXPERIMENT_SEED)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "status.json",
        {"status": "running", "experiment": asdict(experiment), "loss_profile": loss_profile},
    )
    cfg, protocol_audit = preflight(config_path, checkpoint_path, expected_sha256=expected_u0_sha256)
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    expected = schedule.pop("schedule_sha256")
    if sha256_json(schedule) != expected:
        raise ValueError("Mask schedule hash mismatch.")
    schedule["schedule_sha256"] = expected
    loaders = build_dataloaders(
        cfg,
        normalization_overrides=checkpoint_normalization_overrides(checkpoint_path),
    )
    if set(loaders) != {"train", "validation"}:
        raise ValueError("Adapter workflow may only construct train and validation loaders.")
    if sorted(dataset_sample_ids(loaders["train"].dataset)) != schedule["sample_ids"]:
        raise ValueError("Mask schedule sample domain does not match inner_train.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, structure_audit = load_frozen_u0(cfg, checkpoint_path, device)
    base_before = state_digest(base_model)
    adapter = _adapter(experiment, base_model, device)
    wrapper = FrozenU0DecisionAdapter(base_model, adapter).to(device)
    if adapter is not None and experiment.variant == "proto_uncertainty_lora":
        initial_cache = state_cache(
            base_model, loaders["train"], cfg, device,
            lambda ids: schedule_masks(schedule, 0, ids, device),
        )
        mean, scale = condition_statistics(initial_cache)
        adapter.set_condition_normalizer(mean.to(device), scale.to(device))
        write_json(output_dir / "condition_normalizer.json", {
            "fit_split": cfg["data_protocol"]["train_role"], "mean": mean.tolist(), "scale": scale.tolist(),
            "cache_batch_size": PROTOTYPE_CACHE_BATCH_SIZE,
            "cache_num_workers": PROTOTYPE_CACHE_NUM_WORKERS,
        })
    history = train_adapter(
        wrapper,
        experiment,
        loaders["train"],
        cfg,
        schedule,
        device,
        epochs=epochs,
        early_stopping=early_stopping,
        loss_profile=loss_profile,
    )
    actual_epochs = int(history[-1]["epoch"]) if history else 0
    stop_state = (
        dict(history[-1]["early_stopping"])
        if history
        else training_loss_early_stop_state([], early_stopping)
    )
    stop_state["max_epochs"] = int(epochs)
    if adapter is None:
        stop_state["stop_reason"] = "not_applicable"
    elif stop_state["stop_reason"] is None:
        stop_state["stop_reason"] = "max_epochs_reached"
    if adapter is not None:
        publish_checkpoint(
            {
                "checkpoint_schema_version": 1,
                "checkpoint_role": "last",
                "epoch": actual_epochs,
                "state_dict": adapter.state_dict(),
            },
            output_dir / "checkpoints", "last.pth",
            metadata={
                "source": "prototype_decision_adapter",
                "checkpoint_policy": "fixed_epoch_last_pth",
                "loss_profile": loss_profile,
            },
        )
    mask_rows, equivalence = evaluate(wrapper, experiment, loaders["validation"], cfg, device, output_dir)
    if state_digest(base_model) != base_before:
        raise RuntimeError("Frozen U0 parameter/state SHA256 changed during Adapter run.")
    summary = {
        "experiment": asdict(experiment), "outer_test_accessed": False, "data_leakage_detected": False,
        "protocol_audit": protocol_audit, "u0_structure_audit": structure_audit,
        "u0_state_sha256_before": base_before, "u0_state_sha256_after": state_digest(base_model),
        "adapter_parameter_count": adapter.parameter_count() if adapter else 0,
        "adapter_flops": adapter.flops_per_sample() if adapter else 0,
        "transport_kernel_audit": (
            adapter.transport_audit()
            if adapter is not None and getattr(adapter, "variant", None) == "circular_transport"
            else None
        ),
        "experiment_seed": EXPERIMENT_SEED,
        "loss_profile": {"name": loss_profile, **ADAPTER_LOSS_PROFILES[loss_profile]},
        "prototype_cache": {
            "batch_size": PROTOTYPE_CACHE_BATCH_SIZE,
            "num_workers": PROTOTYPE_CACHE_NUM_WORKERS,
            "changes_training_effective_batch_size": False,
        },
        "mask_schedule_sha256": expected,
        "training": {
            "max_epochs": epochs,
            "actual_epochs": actual_epochs,
            "actual_optimizer_steps": actual_epochs * len(loaders["train"]),
            "early_stopping": stop_state,
            "history": history,
        },
        "mask_metrics": mask_rows, "full_equivalence": equivalence,
        **aggregate(mask_rows),
    }
    write_json(output_dir / "metrics.json", summary)
    write_json(
        output_dir / "status.json",
        {"status": "completed", "experiment": asdict(experiment), "loss_profile": loss_profile, "return_code": 0},
    )
    for loader in loaders.values():
        shutdown_dataloader_workers(loader)
    return summary


__all__ = [
    "ADAPTER_LOSS_PROFILES", "EXPECTED_U0_SHA256", "EXPERIMENTS", "EXPERIMENT_SEED", "MASKS", "adapter_training_loss",
    "aggregate", "checkpoint_normalization_overrides", "generate_mask_schedule", "numpy_metrics",
    "preflight", "prepare_stage", "run_experiment", "schedule_masks", "split_permutation", "stratified_mask_folds",
]
