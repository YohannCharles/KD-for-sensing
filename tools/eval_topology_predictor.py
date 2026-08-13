#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from kd_sensing.config import load_config
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts, validate_normalization_artifact_fingerprint
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import configure_torch_runtime_threads
from kd_sensing.eval.beam_probe_diagnostic import (
    build_train_power_index,
    build_validation_power_index,
    load_probe_evidence,
    run_tbcp_probe_diagnostic,
    run_tbcp_robustness_sensitivity,
    summarize_tbcp_replays,
    summarize_tbcp_robustness_replays,
)
from kd_sensing.eval.beam_topology_likelihood import (
    fit_topology_likelihood,
    load_topology_likelihood,
    save_topology_likelihood,
    train_power_content_sha256,
)
from kd_sensing.eval.topology_predictor import (
    collect_topology_observations,
    resolve_topology_missing_patterns,
    summarize_topology_matrix,
    write_observation_cache,
)
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_model_state,
    load_torch_payload,
    validate_checkpoint_publication,
)
from kd_sensing.utils.seed import set_seed


PROTOCOL_LINEAGE_KEYS = (
    "mode", "protocol_id", "protocol_version", "split_protocol_version", "manifest_version",
    "assignment_algorithm", "protocol_fingerprint", "audit_id", "audit_sha256", "split_seed",
    "block_size", "split_manifest_hash", "data_source_hash", "window_config_hash", "weather_binding",
    "split_manifest", "train_role", "validation_role", "test_role", "train_sample_count",
    "validation_sample_count", "test_sample_count", "train_sample_id_hash", "validation_sample_id_hash",
    "test_sample_id_hash", "test_evaluated",
)
TOPOLOGY_LINEAGE_KEYS = ("id", "descriptor_sha256", "audit_sha256")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the native four-modal topology predictor and TBCP probing.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    matrix = subparsers.add_parser("matrix", help="Collect the sealed 15-mask validation matrix.")
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--checkpoint", required=True)
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--device")
    matrix.add_argument("--max-batches", type=int)
    fit_probe = subparsers.add_parser("fit-probe-likelihood", help="Fit the train-only topology likelihood.")
    fit_probe.add_argument("--config", required=True)
    fit_probe.add_argument("--output", required=True)
    for action in ("probe-diagnostic", "probe-robustness"):
        probe = subparsers.add_parser(action)
        probe.add_argument("--config", required=True)
        probe.add_argument("--checkpoint", required=True)
        probe.add_argument("--matrix-report", required=True)
        probe.add_argument("--topology-likelihood", required=True)
        probe.add_argument("--output-dir", required=True)
        if action == "probe-robustness":
            probe.add_argument("--samples-per-pattern", type=int, default=512)
        if action == "probe-diagnostic":
            probe.add_argument("--max-samples-per-pattern", type=int)
            probe.add_argument("--include-diagonal-covariance-ablation", action="store_true")
            probe.add_argument("--include-defense-experiments", action="store_true")
            probe.add_argument("--include-batch-feedback-experiments", action="store_true")
    for action in ("probe-summary", "probe-robustness-summary"):
        summary = subparsers.add_parser(action)
        summary.add_argument("--run", action="append", nargs=2, required=True, metavar=("SEED", "RESULT"))
        summary.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action in {"probe-summary", "probe-robustness-summary"}:
        runs = _three_seed_runs(args.run)
        result = (
            summarize_tbcp_replays(runs, output_dir=args.output_dir)
            if args.action == "probe-summary"
            else summarize_tbcp_robustness_replays(runs, output_dir=args.output_dir)
        )
        print(json.dumps(result, indent=2))
        return 0
    cfg = load_config(args.config)
    _require_topology_predictor(cfg)
    if args.action == "fit-probe-likelihood":
        paths, labels, sample_ids, provenance = build_train_power_index(cfg)
        artifact = fit_topology_likelihood(paths, labels, sample_ids, provenance=provenance)
        record = save_topology_likelihood(artifact, args.output)
        print(json.dumps({**record, "fit_split": "train", "outer_test_accessed": False}, indent=2))
        return 0
    if args.action in {"probe-diagnostic", "probe-robustness"}:
        evidence, power_paths, labels, likelihood, source = _load_probe_inputs(args, cfg)
        if args.action == "probe-diagnostic":
            result = run_tbcp_probe_diagnostic(
                evidence,
                power_paths=power_paths,
                indexed_labels=labels,
                likelihood=likelihood,
                likelihood_source=source,
                output_dir=args.output_dir,
                include_diagonal_covariance_ablation=args.include_diagonal_covariance_ablation,
                include_defense_experiments=args.include_defense_experiments,
                include_batch_feedback_experiments=args.include_batch_feedback_experiments,
            )
        else:
            result = run_tbcp_robustness_sensitivity(
                evidence,
                power_paths=power_paths,
                indexed_labels=labels,
                likelihood=likelihood,
                likelihood_source=source,
                output_dir=args.output_dir,
                samples_per_pattern=args.samples_per_pattern,
            )
        print(json.dumps({"report": result["report"], "output_dir": result["output_dir"]}, indent=2))
        return 0
    return _run_matrix(args, cfg)


def _run_matrix(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    configure_torch_runtime_threads(cfg)
    set_seed(int(cfg.get("experiment", {}).get("seed", 0)))
    device = torch.device(args.device) if args.device else build_device(cfg)
    checkpoint = Path(args.checkpoint).resolve()
    metadata = _normalization_metadata(cfg, checkpoint)
    validate_normalization_artifact_fingerprint(cfg, metadata)
    loaders = build_dataloaders(cfg, normalization_overrides=load_normalization_artifacts(metadata) or None)
    model = build_model(cfg["model"]["primary"]).to(device)
    checkpoint_sha256 = _load_checkpoint(model, checkpoint, cfg, device)
    patterns = resolve_topology_missing_patterns(
        cfg.get("evaluation", {}).get("missing_patterns", {}).get("patterns"),
        model.modalities,
    )
    output = Path(args.output).resolve()
    evidence_path = output.with_name(f"{output.stem}_sample_records.pt")
    try:
        records = collect_topology_observations(
            model,
            loaders["validation"],
            cfg,
            device=device,
            patterns=patterns,
            max_batches=args.max_batches,
        )
        binding = _evidence_binding(model, cfg, checkpoint_sha256)
        records["evidence_binding"] = binding
        write_observation_cache(records, evidence_path)
        evidence_sha256, evidence_size = checkpoint_file_digest(evidence_path)
        provenance = {
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256, "role": "validation_best"},
            "sample_evidence": {"path": str(evidence_path), "sha256": evidence_sha256, "size_bytes": evidence_size},
            "data_protocol": dict(_data_protocol(cfg)),
            "prototype_topology": model.prototype_topology_metadata(),
            "experiment_seed": int(cfg.get("experiment", {}).get("seed", -1)),
        }
        report = summarize_topology_matrix(
            records,
            provenance=provenance,
            dba_delta=float(cfg.get("evaluation", {}).get("dba_delta", 5.0)),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(output), "sample_evidence": str(evidence_path), "pattern_count": 15}, indent=2))
        return 0
    finally:
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)


def _load_probe_inputs(args: argparse.Namespace, cfg: Mapping[str, Any]):
    train_paths, _, _, expected = build_train_power_index(cfg)
    likelihood_path = Path(args.topology_likelihood).resolve()
    likelihood = load_topology_likelihood(
        likelihood_path,
        expected_provenance=expected,
        expected_train_power_content_sha256=train_power_content_sha256(train_paths),
    )
    digest, size = checkpoint_file_digest(likelihood_path)
    evidence = load_probe_evidence(
        matrix_report=args.matrix_report,
        checkpoint=args.checkpoint,
        cfg=cfg,
        max_samples_per_pattern=getattr(args, "max_samples_per_pattern", None),
    )
    power_paths, labels = build_validation_power_index(cfg)
    source = {
        "path": str(likelihood_path), "sha256": digest, "size_bytes": size,
        "artifact_fingerprint": likelihood.metadata["artifact_fingerprint"], "fit_split": "train",
    }
    return evidence, power_paths, labels, likelihood, source


def _load_checkpoint(model: Any, path: Path, cfg: Mapping[str, Any], device: torch.device) -> str:
    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("checkpoint_role") != "validation_best":
        raise ValueError("Topology evaluation requires a validation_best checkpoint.")
    validate_checkpoint_publication(path, payload=payload)
    metadata = payload.get("model_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("type") != "four_modal_topology_predictor":
        raise ValueError("Checkpoint is not a native four-modal topology predictor.")
    if _topology_lineage(metadata.get("prototype_topology", {})) != _topology_lineage(model.prototype_topology_metadata()):
        raise ValueError("Checkpoint topology provenance does not match config.")
    checkpoint_protocol, checkpoint_seed = _checkpoint_lineage(payload)
    if _protocol_lineage(checkpoint_protocol) != _protocol_lineage(_data_protocol(cfg)):
        raise ValueError("Checkpoint data protocol does not match config.")
    if checkpoint_seed != int(cfg.get("experiment", {}).get("seed", -1)):
        raise ValueError("Checkpoint seed does not match config.")
    load_model_state(path, model, role="topology_evaluation", map_location=device, strict=True)
    return checkpoint_file_digest(path)[0]


def _checkpoint_lineage(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], int]:
    resume = payload.get("resume_contract")
    recorded = resume.get("config") if isinstance(resume, Mapping) else None
    protocol = payload.get("data_protocol")
    if not isinstance(protocol, Mapping) and isinstance(recorded, Mapping):
        protocol = recorded.get("data_protocol")
    seed = payload.get("experiment_seed")
    if seed is None and isinstance(recorded, Mapping):
        seed = recorded.get("experiment", {}).get("seed")
    if not isinstance(protocol, Mapping) or seed is None:
        raise ValueError("Checkpoint is missing protocol or seed provenance.")
    return protocol, int(seed)


def _normalization_metadata(cfg: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    payload = load_torch_payload(checkpoint, map_location="cpu")
    checkpoint_artifacts = payload.get("normalization_artifacts", {}) if isinstance(payload, Mapping) else {}
    configured = cfg.get("data", {}).get("normalization_artifacts", {})
    if checkpoint_artifacts and configured and checkpoint_artifacts != configured:
        raise ValueError("Checkpoint and config normalization artifacts do not match.")
    return {"normalization_artifacts": checkpoint_artifacts or configured}


def _evidence_binding(model: Any, cfg: Mapping[str, Any], checkpoint_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "checkpoint_sha256": checkpoint_sha256,
        "prototype_topology": _topology_lineage(model.prototype_topology_metadata()),
        "data_protocol": _protocol_lineage(_data_protocol(cfg)),
        "experiment_seed": int(cfg.get("experiment", {}).get("seed", -1)),
    }


def _data_protocol(cfg: Mapping[str, Any]) -> dict[str, Any]:
    value = cfg.get("data_protocol")
    if not isinstance(value, Mapping) or value.get("protocol_id") != "mmw_id_stratified_block_v1":
        raise ValueError("Topology evaluation requires the audited MMW id-block protocol.")
    if value.get("test_evaluated") is not False:
        raise ValueError("Topology evaluation requires the outer test to remain sealed.")
    return dict(value)


def _protocol_lineage(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.get(key) for key in PROTOCOL_LINEAGE_KEYS}


def _topology_lineage(value: Mapping[str, Any]) -> dict[str, str]:
    return {key: str(value.get(key, "")) for key in TOPOLOGY_LINEAGE_KEYS}


def _require_topology_predictor(cfg: Mapping[str, Any]) -> None:
    if cfg.get("model", {}).get("primary", {}).get("type") != "four_modal_topology_predictor":
        raise ValueError("This tool accepts only four_modal_topology_predictor configs.")
    final_test = cfg.get("training", {}).get("final_test", {})
    if final_test is True or (isinstance(final_test, Mapping) and final_test.get("enabled", True)):
        raise ValueError("Topology evaluation requires training.final_test.enabled=false.")


def _three_seed_runs(values: list[list[str]]) -> dict[int, str]:
    runs = {int(seed): path for seed, path in values}
    if set(runs) != {1, 2, 3} or len(values) != 3:
        raise ValueError("Summary requires exactly seeds 1, 2, and 3.")
    return runs


if __name__ == "__main__":
    raise SystemExit(main())
