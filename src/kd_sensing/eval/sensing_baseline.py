"""Sensing-only adapters for the retained AMBER-Full/RMBP-MM baselines.

This module deliberately lives outside the native topology-predictor loader.  A
baseline checkpoint is evaluated with its own model/config, while the resolved
topology config is used only to bind the audited ULA topology, train-only
likelihood and private validation radio index.  Thus a baseline posterior can
be sent through the exact same finite-probing evaluator without weakening the
native checkpoint contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.evaluation_pass_runtime import metadata_rows_from_batch, prepare_evaluation_batch
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import prepare_task_labels, run_model_step
from kd_sensing.eval.beam_probe_diagnostic import (
    EXPECTED_MODALITIES,
    EXPECTED_PROTOCOL,
    EXPECTED_TOPOLOGY,
    ProbeEvidence,
    build_train_power_index,
    build_validation_power_index,
    run_tbcp_probe_diagnostic,
)
from kd_sensing.eval.beam_topology_likelihood import (
    load_topology_likelihood,
    train_power_content_sha256,
)
from kd_sensing.eval.topology_predictor import resolve_topology_missing_patterns
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_model_state,
    load_torch_payload,
    validate_checkpoint_publication,
)
from kd_sensing.utils.missing_patterns import make_fixed_missing_mask


NUM_BEAMS = 64
BASELINE_FAMILY_BY_CORE = {
    "amber_full_adaptive_mask_transformer": "AMBER-Full-local",
    "rmbp_channel_attention_fusion": "RMBP-MM-local",
}

# Keep this list in sync with the native topology evaluator without importing
# its CLI.  Only these fields are used for cross-config/checkpoint identity.
PROTOCOL_LINEAGE_KEYS = (
    "mode",
    "protocol_id",
    "protocol_version",
    "split_protocol_version",
    "manifest_version",
    "assignment_algorithm",
    "protocol_fingerprint",
    "audit_id",
    "audit_sha256",
    "split_seed",
    "block_size",
    "split_manifest_hash",
    "data_source_hash",
    "window_config_hash",
    "weather_binding",
    "split_manifest",
    "train_role",
    "validation_role",
    "test_role",
    "train_sample_count",
    "validation_sample_count",
    "test_sample_count",
    "train_sample_id_hash",
    "validation_sample_id_hash",
    "test_sample_id_hash",
    "test_evaluated",
)
TOPOLOGY_KEYS = ("id", "descriptor_sha256", "audit_sha256")


def validate_sensing_only_baseline_config(
    baseline_cfg: Mapping[str, Any],
    topology_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the strict baseline/topology binding and return baseline metadata."""

    baseline_model = _mapping(baseline_cfg.get("model", {}).get("primary"))
    topology_model = _mapping(topology_cfg.get("model", {}).get("primary"))
    if baseline_model.get("type") != "modular_sequence":
        raise ValueError("Sensing baseline adapter requires a modular_sequence baseline config.")
    if tuple(baseline_model.get("modalities", ())) != EXPECTED_MODALITIES:
        raise ValueError("Sensing baseline must use canonical image/radar/gps/lidar modalities.")
    if topology_model.get("type") != "four_modal_topology_predictor":
        raise ValueError("The binding config must be a resolved four_modal_topology_predictor config.")
    if tuple(topology_model.get("modalities", ())) != EXPECTED_MODALITIES:
        raise ValueError("The topology binding config must use canonical four-sensing modalities.")

    baseline_protocol = _require_protocol(baseline_cfg, "baseline")
    topology_protocol = _require_protocol(topology_cfg, "topology")
    if _protocol_lineage(baseline_protocol) != _protocol_lineage(topology_protocol):
        raise ValueError("Baseline and topology configs do not share one MMW protocol identity.")
    baseline_seed = _experiment_seed(baseline_cfg)
    topology_seed = _experiment_seed(topology_cfg)
    if baseline_seed != topology_seed:
        raise ValueError(
            f"Baseline seed {baseline_seed} does not match topology binding seed {topology_seed}."
        )
    if int(baseline_model.get("seq_length", 5)) != int(topology_model.get("seq_length", 5)):
        raise ValueError("Baseline and topology configs must use the same history window.")
    if int(baseline_model.get("num_pred", 1)) != int(topology_model.get("num_pred", 1)):
        raise ValueError("Baseline and topology configs must use the same prediction horizon.")
    if int(baseline_model.get("num_classes", NUM_BEAMS)) != NUM_BEAMS:
        raise ValueError("Sensing baseline must predict the audited 64-beam label space.")
    if int(baseline_model.get("seq_length", 5)) != 5 or int(baseline_model.get("num_pred", 1)) != 1:
        raise ValueError("Sensing baseline evidence requires the canonical five-frame/one-step setup.")

    # A baseline may only use the four sensing streams.  Be explicit about
    # common names used by historical local recipes so a stale history-beam
    # branch cannot silently enter the strict panel.
    forbidden_history_fields = {
        "history_beam",
        "history_beam_index",
        "current_beam",
        "previous_beam",
        "beam_history",
    }
    for field in forbidden_history_fields:
        if bool(baseline_model.get(field, False)):
            raise ValueError(f"Sensing-only baseline cannot consume history-beam model field {field!r}.")

    dataset_cfg = _mapping(_mapping(baseline_cfg.get("data")).get("dataset"))
    if dataset_cfg.get("type") != "mmw":
        raise ValueError("Sensing baseline evidence is restricted to the MMW validation protocol.")
    for field in ("include_channel_ref", "include_channel_history_refs", "use_channel", "use_csi"):
        if bool(dataset_cfg.get(field, False)):
            raise ValueError(f"Sensing-only baseline cannot consume {field}.")
    training = _mapping(baseline_cfg.get("training"))
    final_test = training.get("final_test", {})
    if final_test is True or (isinstance(final_test, Mapping) and final_test.get("enabled", True)):
        raise ValueError("Sensing baseline evaluation requires training.final_test.enabled=false.")
    runtime = _mapping(baseline_cfg.get("runtime"))
    if runtime.get("evaluate_test_requested", False) or baseline_protocol.get("test_evaluated") is not False:
        raise ValueError("Sensing baseline evaluation requires the outer test to remain sealed.")

    topology = _topology_binding(topology_cfg)
    if topology["id"] != EXPECTED_TOPOLOGY:
        raise ValueError(f"Expected the audited ULA topology {EXPECTED_TOPOLOGY!r}, got {topology['id']!r}.")
    core_type = str(_mapping(baseline_model.get("representation_core")).get("type", ""))
    if core_type not in BASELINE_FAMILY_BY_CORE:
        raise ValueError(
            "Sensing baseline must be AMBER-Full-local or RMBP-MM-local; "
            f"got representation_core={core_type!r}."
        )
    return {
        "family": BASELINE_FAMILY_BY_CORE[core_type],
        "representation_core_type": core_type,
        "experiment_seed": baseline_seed,
        "data_protocol": dict(baseline_protocol),
        "prototype_topology": topology,
    }


def validate_baseline_checkpoint(
    checkpoint: str | Path,
    baseline_cfg: Mapping[str, Any],
    topology_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate a published validation-best baseline checkpoint and its lineage."""

    path = Path(checkpoint).resolve()
    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("checkpoint_role") != "validation_best":
        raise ValueError("Sensing baseline evaluation requires a validation_best checkpoint.")
    validate_checkpoint_publication(path, payload=payload)
    binding = validate_sensing_only_baseline_config(baseline_cfg, topology_cfg)
    model_metadata = payload.get("model_metadata")
    if not isinstance(model_metadata, Mapping):
        resume = _mapping(payload.get("resume_contract"))
        recorded_cfg = _mapping(resume.get("config"))
        recorded_model = _mapping(_mapping(recorded_cfg.get("model")).get("primary"))
        model_metadata = {
            "type": recorded_model.get("type"),
            "modalities": recorded_model.get("modalities"),
            "representation_core_type": _mapping(recorded_model.get("representation_core")).get("type"),
        }
    if model_metadata.get("type") != "modular_sequence":
        raise ValueError("Checkpoint is not an AMBER-Full/RMBP-MM modular_sequence checkpoint.")
    if tuple(model_metadata.get("modalities", ())) != EXPECTED_MODALITIES:
        raise ValueError("Baseline checkpoint modalities are not the canonical four sensing modalities.")
    for field in ("history_beam_usage", "history_beam_index", "current_beam_usage"):
        value = model_metadata.get(field)
        if value not in (None, False, "disabled", "not_used", "not_applicable"):
            raise ValueError(f"Baseline checkpoint declares forbidden history-beam usage: {field}={value!r}.")

    checkpoint_protocol = payload.get("data_protocol")
    if not isinstance(checkpoint_protocol, Mapping):
        resume = payload.get("resume_contract")
        recorded = resume.get("config") if isinstance(resume, Mapping) else None
        checkpoint_protocol = recorded.get("data_protocol") if isinstance(recorded, Mapping) else None
    if not isinstance(checkpoint_protocol, Mapping):
        raise ValueError("Baseline checkpoint is missing data-protocol provenance.")
    if _protocol_lineage(checkpoint_protocol) != _protocol_lineage(binding["data_protocol"]):
        raise ValueError("Baseline checkpoint protocol does not match the resolved baseline config.")
    checkpoint_seed = payload.get("experiment_seed")
    if checkpoint_seed is None:
        resume = payload.get("resume_contract")
        recorded = resume.get("config") if isinstance(resume, Mapping) else None
        checkpoint_seed = _mapping(_mapping(recorded).get("experiment")).get("seed")
    if checkpoint_seed is None or int(checkpoint_seed) != int(binding["experiment_seed"]):
        raise ValueError("Baseline checkpoint seed does not match the baseline/topology config.")

    core_type = str(model_metadata.get("representation_core_type", ""))
    if core_type != binding["representation_core_type"]:
        raise ValueError("Baseline checkpoint representation core does not match the baseline config.")
    return dict(payload), checkpoint_file_digest(path)[0]


def collect_sensing_baseline_observations(
    model: Any,
    dataloader: Any,
    cfg: Mapping[str, Any],
    *,
    device: str | torch.device,
    patterns: Mapping[str, Sequence[int]] | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Collect complete 15-mask validation logits and softmax posteriors."""

    if tuple(getattr(model, "modalities", ())) != EXPECTED_MODALITIES:
        raise ValueError("Baseline observation collection requires canonical four-sensing modalities.")
    resolved_patterns = dict(
        patterns
        or resolve_topology_missing_patterns(
            _mapping(_mapping(cfg.get("evaluation")).get("missing_patterns")).get("patterns")
            or _default_pattern_names(),
            EXPECTED_MODALITIES,
        )
    )
    _validate_patterns(resolved_patterns)
    target_device = torch.device(device)
    model_cfg = _mapping(_mapping(cfg.get("model")).get("primary"))
    task = str(_mapping(cfg.get("experiment")).get("task", "fusion"))
    chunks: dict[str, list[Any]] = {key: [] for key in ("labels", "available", "logits", "probabilities")}
    strings: dict[str, list[str]] = {
        key: [] for key in ("weather", "domain", "pattern", "sample_id", "protocol_sample_id", "group_id")
    }
    model.to(target_device).eval()
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = prepare_evaluation_batch(raw_batch)
            labels = prepare_task_labels(
                batch,
                num_pred=int(model_cfg.get("num_pred", 1)),
                device=target_device,
            )[:, 0]
            metadata = metadata_rows_from_batch(batch.get("metadata"))
            if len(metadata) != int(labels.shape[0]):
                raise ValueError("Baseline evaluation requires metadata for every validation sample.")
            for pattern_name, raw_pattern in resolved_patterns.items():
                mask = tuple(int(value) for value in raw_pattern)
                forced = make_fixed_missing_mask(int(labels.shape[0]), mask, device=target_device)
                extra = {} if all(mask) else {"missing_mask": forced}
                step = run_model_step(
                    model,
                    task,
                    batch,
                    seq_length=int(model_cfg.get("seq_length", 5)),
                    num_pred=int(model_cfg.get("num_pred", 1)),
                    device=target_device,
                    extra_model_kwargs=extra,
                )
                logits = step.logits[:, 0].float()
                if logits.ndim != 2 or logits.shape[1] != NUM_BEAMS:
                    raise ValueError(f"Baseline logits must be [B,64], got {tuple(logits.shape)}.")
                probability = torch.softmax(logits, dim=-1)
                chunks["labels"].append(labels.detach().cpu())
                chunks["available"].append(
                    torch.as_tensor(mask, dtype=torch.bool).view(1, -1).expand(labels.shape[0], -1).clone()
                )
                chunks["logits"].append(logits.detach().cpu())
                chunks["probabilities"].append(probability.detach().cpu())
                for row in metadata:
                    weather = str(row.get("condition") or "unknown")
                    scenario = str(row.get("scenario") or row.get("sensor_scenario") or "unknown")
                    sample_id = str(row.get("stable_sample_id") or row.get("source_sample_id") or row.get("sample_id") or "")
                    protocol_id = str(row.get("source_sample_id") or row.get("sample_id") or "")
                    if not sample_id or not protocol_id:
                        raise ValueError("Baseline evaluation metadata is missing stable/protocol sample identity.")
                    strings["weather"].append(weather)
                    strings["domain"].append(f"{weather}/{scenario}")
                    strings["pattern"].append(str(pattern_name))
                    strings["sample_id"].append(sample_id)
                    strings["protocol_sample_id"].append(protocol_id)
                    strings["group_id"].append(
                        str(row.get("trajectory_group_id") or row.get("contiguous_segment_id") or sample_id)
                    )
    if not chunks["labels"]:
        raise ValueError("Baseline observation collection observed zero validation batches.")
    return {
        **{key: torch.cat(values, dim=0) for key, values in chunks.items()},
        **strings,
        "modalities": list(EXPECTED_MODALITIES),
        "model_type": "modular_sequence_sensing_only_baseline",
        "bounded_evaluation": max_batches is not None,
        "patterns": list(resolved_patterns),
    }


def build_baseline_probe_evidence(
    records: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    max_samples_per_pattern: int | None = None,
) -> ProbeEvidence:
    """Turn generic baseline matrix records into the native probing evidence dataclass."""

    if tuple(records.get("modalities", ())) != EXPECTED_MODALITIES:
        raise ValueError("Baseline records must contain exactly the canonical four modalities.")
    patterns = tuple(str(value) for value in records.get("pattern", ()))
    sample_ids = tuple(str(value) for value in records.get("sample_id", ()))
    labels = torch.as_tensor(records.get("labels"), dtype=torch.long)
    available = torch.as_tensor(records.get("available"), dtype=torch.bool)
    logits = torch.as_tensor(
        records.get("logits", records.get("pred_logits")),
        dtype=torch.float32,
    )
    probability = torch.as_tensor(
        records.get("probabilities", records.get("fused_probability", records.get("pred_prob"))),
        dtype=torch.float32,
    )
    row_count = len(patterns)
    if not (
        row_count == len(sample_ids) == labels.numel() == available.shape[0] == logits.shape[0] == probability.shape[0]
        and available.ndim == 2
        and available.shape[1] == len(EXPECTED_MODALITIES)
        and logits.ndim == 2
        and logits.shape[1] == NUM_BEAMS
        and probability.ndim == 2
        and probability.shape[1] == NUM_BEAMS
    ):
        raise ValueError("Baseline evidence tensors and identity rows have inconsistent shapes.")
    expected = {tuple(bool(bits & (1 << index)) for index in range(4)) for bits in range(1, 1 << 4)}
    masks: dict[str, tuple[bool, ...]] = {}
    for name in dict.fromkeys(patterns):
        rows = [index for index, value in enumerate(patterns) if value == name]
        current = {tuple(bool(value) for value in available[index].tolist()) for index in rows}
        if len(current) != 1:
            raise ValueError(f"Baseline pattern {name} has inconsistent availability masks.")
        masks[name] = next(iter(current))
    if set(masks.values()) != expected or len(masks) != 15:
        raise ValueError("Baseline evidence must contain exactly 15 non-empty four-sensing masks.")
    selected_rows: list[int] = []
    reference_ids: tuple[str, ...] | None = None
    expected_count = int(_mapping(source.get("data_protocol")).get("validation_sample_count", -1))
    if expected_count <= 0:
        raise ValueError("Baseline evidence source has no valid validation sample count.")
    selected_names = tuple(sorted(masks, key=lambda name: (-sum(masks[name]), name)))
    for name in selected_names:
        rows = [index for index, value in enumerate(patterns) if value == name]
        ids = tuple(sample_ids[index] for index in rows)
        if len(ids) != expected_count or len(set(ids)) != expected_count:
            raise ValueError(f"Baseline pattern {name} does not contain the complete validation identity set.")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError("Baseline patterns do not share identical validation identity/order.")
        if max_samples_per_pattern is not None:
            if int(max_samples_per_pattern) <= 0:
                raise ValueError("max_samples_per_pattern must be positive.")
            rows = rows[: int(max_samples_per_pattern)]
        selected_rows.extend(rows)
    row_index = torch.as_tensor(selected_rows, dtype=torch.long)
    selected_probability = probability[row_index].numpy()
    selected_logits = logits[row_index].numpy()
    if not np.isfinite(selected_logits).all() or not np.isfinite(selected_probability).all():
        raise ValueError("Baseline logits/posterior contains non-finite values.")
    if np.any(selected_probability < 0.0) or not np.allclose(
        selected_probability.sum(axis=-1), 1.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError("Baseline posterior rows must be non-negative and normalized.")
    # Evidence must be the actual softmax of the saved logits, not a detached
    # recalibration.  A small tolerance accommodates CPU/GPU softmax rounding.
    expected_probability = torch.softmax(logits[row_index], dim=-1).numpy()
    if not np.allclose(selected_probability, expected_probability, atol=2e-5, rtol=2e-5):
        raise ValueError("Baseline posterior does not match the saved validation logits softmax.")
    selected_source = dict(source)
    selected_source.update(
        {
            "bounded_evaluation": bool(source.get("bounded_evaluation", False)) or max_samples_per_pattern is not None,
            "patterns": list(selected_names),
            "pattern_available_sensing_count": {
                name: int(sum(masks[name])) for name in selected_names
            },
            "validation_stable_sample_id_hash": _sha256_lines(reference_ids or ()),
            "validation_sample_order_sha256": _sha256_ordered_lines(reference_ids or ()),
        }
    )
    return ProbeEvidence(
        sample_id=tuple(sample_ids[index] for index in selected_rows),
        pattern=tuple(patterns[index] for index in selected_rows),
        gt_beam=labels[row_index].numpy(),
        pred_beam=np.argmax(selected_probability, axis=-1).astype(np.int64),
        pred_prob=selected_probability,
        pred_logits=selected_logits,
        source=selected_source,
    )


def write_baseline_observation_cache(records: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(records), target)


def run_sensing_baseline_evaluation(
    *,
    baseline_cfg: Mapping[str, Any],
    checkpoint: str | Path,
    topology_cfg: Mapping[str, Any],
    topology_likelihood: str | Path,
    output_dir: str | Path,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    max_samples_per_pattern: int | None = None,
    batch_size: int = 256,
    include_diagonal_covariance_ablation: bool = False,
    include_defense_experiments: bool = False,
    include_batch_feedback_experiments: bool = False,
) -> dict[str, Any]:
    """Evaluate one baseline and reuse the native TBCP-3 diagnostic."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Refusing to overwrite a non-empty baseline output directory: {root}")
    binding = validate_sensing_only_baseline_config(baseline_cfg, topology_cfg)
    checkpoint_payload, checkpoint_sha256 = validate_baseline_checkpoint(checkpoint, baseline_cfg, topology_cfg)
    checkpoint_path = Path(checkpoint).resolve()

    train_paths, _train_labels, _train_ids, expected_likelihood = build_train_power_index(dict(topology_cfg))
    likelihood_path = Path(topology_likelihood).resolve()
    likelihood = load_topology_likelihood(
        likelihood_path,
        expected_provenance=expected_likelihood,
        expected_train_power_content_sha256=train_power_content_sha256(train_paths),
    )
    likelihood_sha256, likelihood_size = checkpoint_file_digest(likelihood_path)
    loaders = None
    try:
        metadata = _normalization_metadata(baseline_cfg, checkpoint_payload)
        validate_normalization_artifact_fingerprint(dict(baseline_cfg), metadata)
        loaders = build_dataloaders(
            dict(baseline_cfg),
            normalization_overrides=load_normalization_artifacts(metadata) or None,
        )
        if set(loaders) != {"train", "validation"}:
            raise ValueError("Sensing baseline evaluation must construct train/validation only; test is sealed.")
        model = build_model(dict(_mapping(baseline_cfg["model"])["primary"]))
        device_value = torch.device(device) if device is not None else build_device(dict(baseline_cfg))
        load_model_state(checkpoint_path, model, role="sensing_baseline_evaluation", map_location=device_value, strict=True)
        patterns = resolve_topology_missing_patterns(
            _mapping(_mapping(baseline_cfg.get("evaluation")).get("missing_patterns")).get("patterns")
            or _default_pattern_names(),
            EXPECTED_MODALITIES,
        )
        records = collect_sensing_baseline_observations(
            model,
            loaders["validation"],
            baseline_cfg,
            device=device_value,
            patterns=patterns,
            max_batches=max_batches,
        )
        root.mkdir(parents=True, exist_ok=True)
        evidence_path = root / "baseline_sample_records.pt"
        write_baseline_observation_cache(records, evidence_path)
        evidence_sha256, evidence_size = checkpoint_file_digest(evidence_path)
        source = {
            "baseline_family": binding["family"],
            "baseline_model_type": "modular_sequence",
            "baseline_representation_core": binding["representation_core_type"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": checkpoint_file_digest(checkpoint_path)[1],
            "evidence_path": str(evidence_path),
            "evidence_sha256": evidence_sha256,
            "data_protocol": dict(binding["data_protocol"]),
            "prototype_topology": dict(binding["prototype_topology"]),
            "experiment_seed": binding["experiment_seed"],
            "bounded_evaluation": max_batches is not None,
        }
        evidence = build_baseline_probe_evidence(
            records,
            source=source,
            max_samples_per_pattern=max_samples_per_pattern,
        )
        matrix_report = root / "baseline_matrix_report.json"
        provenance = {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "size_bytes": checkpoint_file_digest(checkpoint_path)[1],
                "role": "validation_best",
            },
            "sample_evidence": {"path": str(evidence_path), "sha256": evidence_sha256, "size_bytes": evidence_size},
            "baseline_config": _config_provenance(baseline_cfg),
            "topology_config": _config_provenance(topology_cfg),
            "data_protocol": dict(binding["data_protocol"]),
            "prototype_topology": dict(binding["prototype_topology"]),
            "baseline_family": binding["family"],
            "experiment_seed": binding["experiment_seed"],
        }
        matrix_report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "report_type": "sensing_only_baseline_validation_matrix",
                    "model_type": "modular_sequence_sensing_only_baseline",
                    "baseline_family": binding["family"],
                    "patterns": list(dict.fromkeys(evidence.pattern)),
                    "sample_rows": len(evidence.sample_id),
                    "claim_ineligible": True,
                    "outer_test_accessed": False,
                    "model_trained_or_updated": False,
                    "provenance": provenance,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "provenance.json").write_text(
            json.dumps(
                {
                    **provenance,
                    "claim_ineligible": True,
                    "outer_test_accessed": False,
                    "model_trained_or_updated": False,
                    "topology_likelihood": {
                        "path": str(likelihood_path),
                        "sha256": likelihood_sha256,
                        "size_bytes": likelihood_size,
                        "artifact_fingerprint": likelihood.metadata["artifact_fingerprint"],
                        "fit_split": "train",
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        validation_paths, validation_labels = build_validation_power_index(dict(topology_cfg))
        likelihood_source = {
            "path": str(likelihood_path),
            "sha256": likelihood_sha256,
            "size_bytes": likelihood_size,
            "artifact_fingerprint": likelihood.metadata["artifact_fingerprint"],
            "fit_split": "train",
        }
        return run_tbcp_probe_diagnostic(
            evidence,
            power_paths=validation_paths,
            indexed_labels=validation_labels,
            likelihood=likelihood,
            likelihood_source=likelihood_source,
            output_dir=root / "probe_diagnostic",
            batch_size=batch_size,
            include_diagonal_covariance_ablation=include_diagonal_covariance_ablation,
            include_defense_experiments=include_defense_experiments,
            include_batch_feedback_experiments=include_batch_feedback_experiments,
        ) | {
            "output_dir": str(root),
            "baseline_matrix_report": str(matrix_report),
            "baseline_sample_evidence": str(evidence_path),
            "provenance": str(root / "provenance.json"),
            "baseline_family": binding["family"],
            "claim_ineligible": True,
            "outer_test_accessed": False,
        }
    finally:
        if loaders is not None:
            for loader in loaders.values():
                shutdown_dataloader_workers(loader)


def _normalization_metadata(cfg: Mapping[str, Any], checkpoint_payload: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_artifacts = checkpoint_payload.get("normalization_artifacts", {})
    configured = _mapping(_mapping(cfg.get("data")).get("normalization_artifacts"))
    if checkpoint_artifacts and configured and checkpoint_artifacts != configured:
        raise ValueError("Checkpoint and baseline config normalization artifacts do not match.")
    return {"normalization_artifacts": checkpoint_artifacts or configured}


def _require_protocol(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    protocol = cfg.get("data_protocol")
    if not isinstance(protocol, Mapping) or protocol.get("protocol_id") != EXPECTED_PROTOCOL:
        raise ValueError(f"{name.capitalize()} config must bind {EXPECTED_PROTOCOL}.")
    if protocol.get("test_evaluated") is not False or bool(protocol.get("outer_test_accessed", False)):
        raise ValueError(f"{name.capitalize()} config has test access enabled.")
    return protocol


def _protocol_lineage(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.get(key) for key in PROTOCOL_LINEAGE_KEYS}


def _topology_binding(cfg: Mapping[str, Any]) -> dict[str, str]:
    model = _mapping(_mapping(cfg.get("model")).get("primary"))
    runtime = _mapping(_mapping(cfg.get("runtime")).get("topology_predictor_resolver"))
    loss = _mapping(_mapping(cfg.get("loss")).get("four_modal_topology"))
    candidates = (
        _mapping(model.get("prototype_topology")),
        runtime.get("prototype_topology"),
        loss.get("prototype_topology"),
    )
    topology: dict[str, Any] = {}
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            topology.update(candidate)
    # Resolved configs also flatten these values under model.primary.
    topology.setdefault("id", model.get("prototype_topology_id"))
    topology.setdefault("descriptor_sha256", model.get("prototype_topology_descriptor_sha256"))
    topology.setdefault("audit_sha256", model.get("prototype_topology_audit_sha256"))
    result = {key: str(topology.get(key, "")) for key in TOPOLOGY_KEYS}
    if any(not value for value in result.values()):
        raise ValueError("Resolved topology config is missing formal ULA descriptor/audit provenance.")
    return result


def _experiment_seed(cfg: Mapping[str, Any]) -> int:
    experiment = _mapping(cfg.get("experiment"))
    value = experiment.get("seed", experiment.get("train_seed"))
    if value is None:
        raise ValueError("Config is missing experiment.seed provenance.")
    if experiment.get("train_seed", value) != value:
        raise ValueError("experiment.seed and experiment.train_seed must match.")
    return int(value)


def _config_provenance(cfg: Mapping[str, Any]) -> dict[str, Any]:
    path = cfg.get("_source_path")
    if path:
        file_path = Path(str(path)).resolve()
        return {"path": str(file_path), "sha256": _sha256_file(file_path) if file_path.is_file() else ""}
    return {"path": "", "sha256": ""}


def _validate_patterns(patterns: Mapping[str, Sequence[int]]) -> None:
    if len(patterns) != 15:
        raise ValueError("Baseline matrix requires exactly 15 missing-modality patterns.")
    masks = {tuple(int(value) for value in pattern) for pattern in patterns.values()}
    expected = {tuple(int(bool(bits & (1 << index))) for index in range(4)) for bits in range(1, 1 << 4)}
    if masks != expected:
        raise ValueError("Baseline matrix patterns must enumerate all non-empty four-sensing masks.")


def _default_pattern_names() -> list[str]:
    return [
        "full",
        "missing_image",
        "missing_radar",
        "missing_gps",
        "missing_lidar",
        "image_only",
        "radar_only",
        "gps_only",
        "lidar_only",
        "missing_image_radar",
        "missing_image_gps",
        "missing_image_lidar",
        "missing_radar_gps",
        "missing_radar_lidar",
        "missing_gps_lidar",
    ]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(value) for value in values)).encode("utf-8")).hexdigest()


def _sha256_ordered_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


__all__ = [
    "BASELINE_FAMILY_BY_CORE",
    "build_baseline_probe_evidence",
    "collect_baseline_observations",
    "collect_sensing_baseline_observations",
    "run_sensing_baseline_evaluation",
    "run_baseline_probe_adapter",
    "validate_baseline_checkpoint",
    "validate_sensing_only_baseline_config",
    "write_baseline_observation_cache",
]

# Descriptive alias retained for small local scripts; it does not create a
# second evaluator implementation.
collect_baseline_observations = collect_sensing_baseline_observations
run_baseline_probe_adapter = run_sensing_baseline_evaluation
