#!/usr/bin/env python3

import argparse
import csv
import inspect
import sys
from pathlib import Path
from typing import Any

import torch

from kd_sensing.config.io import load_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import run_model_step
from kd_sensing.eval.missing_patterns import get_missing_pattern_mask
from kd_sensing.utils.checkpoint import load_model_state

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scene31_eval_resolution import complete_run_names, resolve_run_dir_and_config, run_name_sort_key


DEFAULT_RUN_PREFIXES = (
    "amr_lite_natural_es40",
    "amber_lite_natural_es40",
    "amr_lite_randomdrop_subset_es40",
    "amber_lite_randomdrop_subset_es40",
    "amr_lite_uniform_es40",
    "amber_lite_uniform_es40",
)
FIELDNAMES = (
    "model_name",
    "run_name",
    "run_dir",
    "config_path",
    "checkpoint_path",
    "forward_signature",
    "accepts_missing_mask",
    "accepts_missing_modality_metadata",
    "mask_passed_by_eval",
    "mask_filtered_by_batch",
    "mask_used_inside_forward",
    "full_output_equal_missing_output",
    "diagnosis",
    "warning",
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root)
    runs = _expand_runs(root, _requested_runs(args.runs))
    rows = [_diagnose_run(root, run_name, split=args.split, device_override=args.device) for run_name in runs]
    out_path = Path(args.out or root / "modular_missing_mask_diagnostics.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row.get("warning"):
            print(f"[WARN] {row['run_name']}: {row['warning']}")
    print(f"Wrote modular missing-mask diagnostics to {out_path}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose whether modular AMR/AMBER-lite fresh eval consumes missing masks.")
    parser.add_argument("--root", default="outputs/scene31_baseline_pack_lmdb")
    parser.add_argument("--runs", default=",".join(DEFAULT_RUN_PREFIXES), help="Comma/space-separated run names or method prefixes.")
    parser.add_argument("--out", default="")
    parser.add_argument("--split", default="test", choices=("test", "val", "validation"))
    parser.add_argument("--device", default=None)
    return parser


def _requested_runs(raw: str) -> list[str]:
    values = [item for chunk in str(raw or "").split(",") for item in chunk.split() if item]
    return values or list(DEFAULT_RUN_PREFIXES)


def _expand_runs(root: Path, requested: list[str]) -> list[str]:
    known = set(complete_run_names(root))
    manifest = root_manifest_names(root)
    known.update(manifest)
    expanded: list[str] = []
    for item in requested:
        matches = [
            name
            for name in known
            if name == item or name.startswith(f"{item}_seed") or name.startswith(item)
        ]
        expanded.extend(sorted(matches, key=run_name_sort_key) if matches else [item])
    return list(dict.fromkeys(expanded))


def root_manifest_names(root: Path) -> set[str]:
    candidates = [
        Path("configs/scene31/baseline_pack/experiment_manifest.csv"),
        root / "experiment_manifest.csv",
    ]
    names: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    run_name = str(row.get("run_name") or "").strip()
                    if run_name:
                        names.add(run_name)
        except OSError:
            continue
    return names


def _diagnose_run(root: Path, run_name: str, *, split: str, device_override: str | None) -> dict[str, Any]:
    resolution = resolve_run_dir_and_config(root, run_name)
    row: dict[str, Any] = {
        "model_name": "",
        "run_name": run_name,
        "run_dir": str(resolution.run_dir or ""),
        "config_path": str(resolution.config_path or ""),
        "checkpoint_path": str(resolution.checkpoint.path or ""),
        "forward_signature": "",
        "accepts_missing_mask": False,
        "accepts_missing_modality_metadata": False,
        "mask_passed_by_eval": True,
        "mask_filtered_by_batch": True,
        "mask_used_inside_forward": False,
        "full_output_equal_missing_output": "",
        "diagnosis": resolution.diagnosis,
        "warning": "",
    }
    cfg = _load_config(resolution.config_path)
    if cfg is None:
        row["warning"] = "config unavailable"
        return row
    model_cfg = cfg.get("model", {}).get("primary", {})
    row["model_name"] = str(model_cfg.get("type", ""))
    try:
        model = build_model(model_cfg)
    except Exception as exc:
        row["diagnosis"] = "model_build_failed"
        row["warning"] = str(exc)
        return row
    signature = inspect.signature(model.forward)
    row["forward_signature"] = str(signature)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    row["accepts_missing_mask"] = accepts_kwargs or "missing_mask" in signature.parameters
    row["accepts_missing_modality_metadata"] = accepts_kwargs or "missing_modality_metadata" in signature.parameters
    row["mask_filtered_by_batch"] = not bool(row["accepts_missing_mask"])
    if not row["accepts_missing_mask"]:
        row["diagnosis"] = "missing_mask_filtered_by_batch_signature"
        row["warning"] = "model.forward does not accept missing_mask"
        return row
    if resolution.checkpoint.path is None:
        row["diagnosis"] = "missing_checkpoint"
        row["warning"] = "best checkpoint unavailable; static signature check only"
        return row
    try:
        compare = _compare_full_and_missing(
            cfg,
            model,
            resolution.checkpoint.path,
            split=split,
            device_override=device_override,
        )
    except Exception as exc:
        row["diagnosis"] = "forward_compare_failed"
        row["warning"] = str(exc)
        return row
    row.update(compare)
    if row["full_output_equal_missing_output"] is True:
        row["diagnosis"] = "warning_identical_full_and_missing_logits"
        row["warning"] = "full and missing pattern logits are exactly equal"
    elif row["mask_used_inside_forward"]:
        row["diagnosis"] = "ok"
    else:
        row["diagnosis"] = "mask_not_observed_in_forward_output"
    return row


def _load_config(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    if "configs" in path.parts:
        return load_config(path)
    data = safe_load_yaml(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else None


def _compare_full_and_missing(
    cfg: dict[str, Any],
    model: Any,
    checkpoint: Path,
    *,
    split: str,
    device_override: str | None,
) -> dict[str, Any]:
    if device_override:
        cfg.setdefault("experiment", {})["device"] = device_override
    device = build_device(cfg)
    model.to(device)
    load_model_state(
        checkpoint,
        model,
        role="modular-missing-mask-diagnostic",
        map_location=device,
        strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)),
    )
    model.eval()
    dataloaders = build_dataloaders(cfg)
    split_key = "val" if split == "validation" else split
    if split_key not in dataloaders:
        split_key = "test" if "test" in dataloaders else next(iter(dataloaders))
    batch = next(iter(dataloaders[split_key]))
    model_cfg = cfg["model"]["primary"]
    modalities = list(model_cfg.get("modalities") or ["image", "radar", "gps", "lidar"])
    pattern_name = "missing_gps" if "gps" in modalities else f"missing_{modalities[0]}"
    mask_row = torch.tensor(get_missing_pattern_mask(pattern_name, modalities), dtype=torch.bool, device=device)
    batch_size = _batch_size(batch)
    missing_mask = mask_row.unsqueeze(0).expand(batch_size, -1).contiguous()
    num_pred = int(model_cfg.get("num_pred", cfg.get("model", {}).get("num_pred", 1)))
    seq_length = int(model_cfg.get("seq_length", cfg.get("model", {}).get("seq_length", 8)))
    downsample_ratio = int(model_cfg.get("downsample_ratio", cfg.get("model", {}).get("downsample_ratio", 1)))
    task = cfg.get("experiment", {}).get("task", "image")
    with torch.no_grad():
        full = run_model_step(
            model,
            task,
            batch,
            model_cfg=model_cfg,
            seq_length=seq_length,
            num_pred=num_pred,
            downsample_ratio=downsample_ratio,
            device=device,
        )
        missing = run_model_step(
            model,
            task,
            batch,
            model_cfg=model_cfg,
            seq_length=seq_length,
            num_pred=num_pred,
            downsample_ratio=downsample_ratio,
            device=device,
            extra_model_kwargs={
                "missing_mask": missing_mask,
                "missing_modality_metadata": {
                    "pattern": pattern_name,
                    "debug_missing_mask": True,
                },
            },
        )
    metadata = missing.model_output.diagnostics.get("missing_modality_metadata")
    mask_used = bool(isinstance(metadata, dict) and metadata.get("missing_counts"))
    return {
        "mask_used_inside_forward": mask_used,
        "full_output_equal_missing_output": bool(torch.equal(full.logits.detach().cpu(), missing.logits.detach().cpu())),
    }


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
    if isinstance(batch, (list, tuple)) and batch:
        return _batch_size(batch[0])
    raise ValueError("Could not infer batch size for diagnostic batch.")


if __name__ == "__main__":
    raise SystemExit(main())
