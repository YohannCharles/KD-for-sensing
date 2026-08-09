#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from kd_sensing.config import dump_config, load_config
from kd_sensing.config.validation import validate_loaded_config
from kd_sensing.data.mmw.trajectory_protocol import bind_trajectory_config
from kd_sensing.engine.optim import build_model
from kd_sensing.utils.checkpoint import checkpoint_file_digest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "tools/configs/topology_predictor/topology_on.yaml"
DEFAULT_MANIFEST = ROOT / "outputs/splits/mmw_id_stratified_block_v1/seed_0.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve, preflight, or train the single-stage four-modal topology predictor.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    resolve.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    resolve.add_argument("--topology-audit", required=True)
    resolve.add_argument("--output", required=True)
    resolve.add_argument("--train-seed", type=int, required=True)
    resolve.add_argument("--run-name", required=True)
    for action in ("preflight", "train"):
        command = subparsers.add_parser(action)
        command.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "resolve":
        cfg = resolve_config(
            template=Path(args.template),
            manifest=Path(args.manifest),
            topology_audit=Path(args.topology_audit),
            train_seed=int(args.train_seed),
            run_name=str(args.run_name),
        )
        output = Path(args.output).resolve()
        dump_config(cfg, output)
        load_config(output)
        print(json.dumps({"resolved_config": str(output), "train_seed": args.train_seed}, indent=2))
        return 0
    cfg = load_config(args.config)
    _require_resolved(cfg)
    if args.action == "preflight":
        model = build_model(cfg["model"]["primary"])
        print(json.dumps({"status": "passed", "model_metadata": model.checkpoint_metadata()}, indent=2))
        return 0
    from kd_sensing.engine.trainer import train

    result = train(cfg)
    print(json.dumps(result, indent=2))
    return 0


def resolve_config(
    *,
    template: Path,
    manifest: Path,
    topology_audit: Path,
    train_seed: int,
    run_name: str,
) -> dict[str, Any]:
    if train_seed < 0 or not run_name.strip():
        raise ValueError("train_seed must be non-negative and run_name must be non-empty.")
    cfg = load_config(template.resolve())
    cfg.setdefault("runtime", {})["evaluate_test_requested"] = False
    cfg.setdefault("experiment", {}).update(seed=train_seed, train_seed=train_seed)
    cfg.setdefault("output", {})["run_name"] = run_name.strip()
    cfg.setdefault("training", {}).update(resume=False, final_test={"enabled": False})
    bind_trajectory_config(cfg, manifest.resolve())
    topology = _bind_topology_audit(cfg, topology_audit.resolve())
    cfg["runtime"]["topology_predictor_resolver"] = {
        "schema_version": 1,
        "single_stage": True,
        "modalities": ["image", "radar", "gps", "lidar"],
        "mask_count": 15,
        "prototype_topology": topology,
        "outer_test_accessed": False,
    }
    validate_loaded_config(cfg)
    return cfg


def _bind_topology_audit(cfg: dict[str, Any], path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Topology audit is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    descriptor = payload.get("descriptor") if isinstance(payload, Mapping) else None
    if not isinstance(descriptor, Mapping):
        raise ValueError("Topology audit is missing its descriptor.")
    descriptor_sha256 = hashlib.sha256(
        json.dumps(dict(descriptor), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        payload.get("descriptor_sha256") != descriptor_sha256
        or descriptor.get("topology_id") != "ula_dft_phase_cycle_v1"
        or descriptor.get("codebook_type") != "ula_dft"
        or int(descriptor.get("num_beams", -1)) != 64
        or descriptor.get("claim_boundary") != "local_ula_dft_phase_codebook_not_world_azimuth_ring"
        or payload.get("metadata_consistent") is not True
        or payload.get("errors") != []
    ):
        raise ValueError("Topology audit does not match the formal ULA-DFT/protocol contract.")
    audited_domains = {
        str(row.get("id"))
        for row in payload.get("domains", [])
        if isinstance(row, Mapping) and row.get("metadata_status") == "verified"
    }
    protocol_domains = {
        str(row.get("id"))
        for row in cfg.get("data", {}).get("dataset", {}).get("domains", [])
        if isinstance(row, Mapping)
    }
    if len(audited_domains) != 15 or audited_domains != protocol_domains:
        raise ValueError("Topology audit domains do not match the 15-domain protocol inventory.")
    audit_sha256, _ = checkpoint_file_digest(path)
    topology = {
        "id": "ula_dft_phase_cycle_v1",
        "descriptor_sha256": descriptor_sha256,
        "audit_path": str(path),
        "audit_sha256": audit_sha256,
    }
    cfg["loss"]["four_modal_topology"]["prototype_topology"] = dict(topology)
    cfg["model"]["primary"].update(
        prototype_topology_id=topology["id"],
        prototype_topology_descriptor_sha256=descriptor_sha256,
        prototype_topology_audit_path=str(path),
        prototype_topology_audit_sha256=audit_sha256,
    )
    return topology


def _require_resolved(cfg: Mapping[str, Any]) -> None:
    if cfg.get("model", {}).get("primary", {}).get("type") != "four_modal_topology_predictor":
        raise ValueError("Resolved config must use four_modal_topology_predictor.")
    resolver = cfg.get("runtime", {}).get("topology_predictor_resolver")
    if not isinstance(resolver, Mapping) or resolver.get("single_stage") is not True:
        raise ValueError("Config was not produced by the topology-predictor resolver.")
    if cfg.get("data_protocol", {}).get("test_evaluated") is not False:
        raise ValueError("Resolved topology training must keep the outer test sealed.")


if __name__ == "__main__":
    raise SystemExit(main())
