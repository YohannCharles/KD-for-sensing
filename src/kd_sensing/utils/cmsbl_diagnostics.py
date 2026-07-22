import json
from pathlib import Path
from typing import Any


def write_cmsbl_epoch_diagnostics(
    run_dir: str | Path,
    *,
    epoch: int,
    dataset: str,
    modalities: tuple[str, ...],
    capacity_identity: dict[str, Any],
    modality_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    losses: dict[str, float],
    metrics: dict[str, float],
) -> str:
    root = Path(run_dir) / "cmsbl"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "cmsbl_inner_train_v1",
        "claim_eligible": False,
        "state_source": "train_only",
        "dataset": str(dataset),
        "epoch": int(epoch),
        "modalities": list(modalities),
        "capacity_reference": dict(capacity_identity),
        "modality": modality_rows,
        "mask": mask_rows,
        "loss": dict(losses),
        "metrics": dict(metrics),
    }
    json_path = root / f"epoch_{int(epoch):04d}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(json_path)


__all__ = ["write_cmsbl_epoch_diagnostics"]
