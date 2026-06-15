from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from kd_sensing.baselines.jepa_msac.config import validate_jepa_msac_workflow_config
from kd_sensing.baselines.jepa_msac.data import JepaMsacWindowProtocol, build_scenario32_manifest
from kd_sensing.baselines.jepa_msac.fixture import make_synthetic_jepa_msac_batch
from kd_sensing.baselines.jepa_msac.metrics import evaluate_jepa_msac_predictions
from kd_sensing.baselines.jepa_msac.report import DEFAULT_REPORT_ROOT, build_ablation_row, write_ablation_manifest, write_report
from kd_sensing.config import load_config
from kd_sensing.losses.jepa_msac import jepa_msac_stage2_losses, masked_latent_smooth_l1_loss
from kd_sensing.models.jepa_msac import JepaMsacModel, freeze_jepa_msac_backbone


STAGES = ("pretrain", "heads", "evaluate", "report", "all")


def run_jepa_msac(
    *,
    config_path: str | Path = "configs/pretraining/jepa_msac_s32_smoke.yaml",
    stage: str = "report",
    dry_run: bool = False,
    pretrained_checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    selected_stage = str(stage).strip().lower()
    if selected_stage not in STAGES:
        raise ValueError(f"JEPA-MSAC stage must be one of {STAGES}, got {stage!r}.")
    cfg = load_config(config_path)
    workflow_metadata = validate_jepa_msac_workflow_config(cfg)
    output_root = Path(output_dir) if output_dir is not None else Path(cfg.get("output", {}).get("dir", DEFAULT_REPORT_ROOT))
    protocol = JepaMsacWindowProtocol(
        t_hist=int(workflow_metadata["t_hist"]),
        t_pred=int(workflow_metadata["t_pred"]),
        split_seed=int(cfg.get("workflow", {}).get("jepa_msac", {}).get("split_seed", 42)),
        train_ratio=float(cfg.get("workflow", {}).get("jepa_msac", {}).get("train_ratio", 0.7)),
    )
    manifest = build_scenario32_manifest(
        csv_path=cfg.get("workflow", {}).get("jepa_msac", {}).get("scenario32_csv"),
        output_path=output_root / "scenario32_manifest.json",
        protocol=protocol,
        dry_run=bool(dry_run),
        write=bool(write and (dry_run or selected_stage in {"report", "all"})),
    )
    report: dict[str, Any] = {
        "workflow_family": "jepa_msac",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": selected_stage,
        "dry_run": bool(dry_run),
        "config_path": str(config_path),
        "pretrained_checkpoint": str(pretrained_checkpoint) if pretrained_checkpoint is not None else None,
        "claim_status": "mock/smoke" if dry_run else "unverified",
        "output_dir": str(output_root),
        "workflow_metadata": workflow_metadata,
        "manifest": manifest.to_dict(),
        "caveats": [
            "No paper-aligned long training metrics are promoted by this runner.",
            "Real DeepSense6G data, checkpoints, metrics and figures must stay in ignored local output paths.",
        ],
    }
    if selected_stage in {"pretrain", "all"}:
        report["pretrain_smoke"] = _pretrain_smoke(cfg)
    if selected_stage in {"heads", "all"}:
        report["heads_smoke"] = _heads_smoke(cfg)
    if selected_stage in {"evaluate", "all"}:
        report["metrics"] = _evaluate_smoke(cfg)
    if selected_stage in {"report", "all"}:
        row = build_ablation_row(
            config_path=str(config_path),
            checkpoint_provenance=str(pretrained_checkpoint) if pretrained_checkpoint is not None else None,
            run_status="planned" if dry_run else "blocked",
            claim_status=report["claim_status"],
            latent_dim=int(cfg.get("model", {}).get("primary", {}).get("latent_dim", 64)),
            mask_ratio=float(cfg.get("model", {}).get("primary", {}).get("mask_ratio", 0.5)),
            mask_pattern=str(cfg.get("model", {}).get("primary", {}).get("mask_pattern", "random")),
            loc_aux=bool(cfg.get("model", {}).get("primary", {}).get("localization_guidance", True)),
        )
        if write:
            report["ablation_manifest"] = write_ablation_manifest([row], output_root)
    if write:
        report["report_paths"] = write_report(report, output_root)
    return report


def _model_from_config(cfg: dict[str, Any]) -> JepaMsacModel:
    primary = dict(cfg.get("model", {}).get("primary", {}))
    primary.pop("type", None)
    primary.pop("modalities", None)
    primary.pop("image_profile", None)
    return JepaMsacModel(**primary)


def _pretrain_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    model = _model_from_config(cfg)
    batch = make_synthetic_jepa_msac_batch(
        batch_size=1,
        t_hist=model.t_hist,
        t_pred=model.t_pred,
        num_beams=model.num_beams,
        image_size=16,
    )
    forward = model(**{key: value for key, value in batch.items() if key.endswith("_batch") or key == "rf_history"})
    loss, diagnostics = masked_latent_smooth_l1_loss(
        forward["predicted_target_latent"],
        forward["target_latent"],
        forward["loss_mask"],
        ema_momentum=model.ema_momentum,
    )
    loss.backward()
    before = next(model.target_encoder.parameters(), torch.tensor(0.0)).detach().clone()
    model.update_target_encoder_ema()
    after = next(model.target_encoder.parameters(), torch.tensor(0.0)).detach().clone()
    return {
        "loss": float(loss.detach().cpu().item()),
        "diagnostics": diagnostics,
        "ema_updated": bool(not torch.equal(before, after)) if before.numel() else True,
        "metadata": model.training_strategy_metadata(),
    }


def _heads_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    model = _model_from_config(cfg)
    freeze_metadata = freeze_jepa_msac_backbone(model)
    batch = make_synthetic_jepa_msac_batch(
        batch_size=1,
        t_hist=model.t_hist,
        t_pred=model.t_pred,
        num_beams=model.num_beams,
        image_size=16,
    )
    outputs = model(stage="heads", **{key: value for key, value in batch.items() if key.endswith("_batch") or key == "rf_history"})
    losses = jepa_msac_stage2_losses(outputs, batch["targets"])
    return {
        "S_pred_shape": list(outputs["S_pred"].shape),
        "beam_logits_shape": list(outputs["beam_logits"].shape),
        "predicted_location_shape": list(outputs["predicted_location"].shape),
        "rssi_profile_shape": list(outputs["rssi_profile"].shape),
        "loss_names": sorted(losses),
        "freeze_metadata": freeze_metadata,
        "stage2_metadata": outputs["stage2_metadata"],
    }


def _evaluate_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    model = _model_from_config(cfg)
    batch = make_synthetic_jepa_msac_batch(
        batch_size=2,
        t_hist=model.t_hist,
        t_pred=model.t_pred,
        num_beams=model.num_beams,
        image_size=16,
    )
    outputs = model(stage="heads", **{key: value for key, value in batch.items() if key.endswith("_batch") or key == "rf_history"})
    return evaluate_jepa_msac_predictions(
        outputs,
        batch["targets"],
        beam_power_reference=batch["targets"]["future_rssi_profile"],
        representation_latents=outputs["S_pred"],
    )


__all__ = ["STAGES", "run_jepa_msac"]
