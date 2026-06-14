from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from kd_sensing.baselines.amr_net_gps_image.metrics import paper_aligned_metric_summary
from kd_sensing.baselines.amr_net_gps_image.preset import (
    AMR_NET_GPS_IMAGE_DISPLAY_NAME,
    AMR_NET_GPS_IMAGE_PRESET_NAME,
    paper_model_groups,
    validate_amr_net_gps_image_preset_config,
)
from kd_sensing.baselines.amr_net_gps_image.source_audit import (
    SourceAudit,
    build_default_source_audit,
    ensure_claim_status_allowed,
)
from kd_sensing.config import load_config
from kd_sensing.utils.runtime_output_layout import PARTITION_ANALYSIS


DEFAULT_OUTPUT_ROOT = Path("outputs") / PARTITION_ANALYSIS / "amr_net_gps_image"


def run_amr_net_gps_image(
    *,
    config_path: str | Path = "configs/baselines/amr_net_gps_image.yaml",
    output_dir: str | Path | None = None,
    mock: bool = True,
    claim_status: str | None = None,
    command: list[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    validate_amr_net_gps_image_preset_config(cfg)
    audit = build_default_source_audit(claim_status=claim_status or ("mock_smoke" if mock else "blocked_official"))
    run_claim_status = "mock_smoke" if mock else ensure_claim_status_allowed(claim_status or audit.claim_status, audit)
    run_id = _run_id()
    out_root = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / run_id
    metrics = _mock_metrics(cfg, claim_status=run_claim_status) if mock else []
    report = _build_report(
        cfg,
        audit,
        metrics=metrics,
        output_root=out_root,
        mock=mock,
        claim_status=run_claim_status,
        command=command or [],
    )
    if write:
        _write_report_artifacts(report, out_root)
    return report


def _mock_metrics(cfg: Mapping[str, Any], *, claim_status: str) -> list[dict[str, Any]]:
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    generator = torch.Generator().manual_seed(seed)
    labels = torch.tensor([[0], [3], [7], [12]], dtype=torch.long)
    metrics = []
    for index, group in enumerate(paper_model_groups()):
        logits = torch.randn(labels.shape[0], 1, group.num_beams, generator=generator) * 0.01
        for sample_idx, label in enumerate(labels[:, 0]):
            logits[sample_idx, 0, int((int(label) + index) % group.num_beams)] = 1.0
        metrics.append(
            paper_aligned_metric_summary(
                logits,
                labels,
                model_group=group.group_id,
                scene=23,
                split="mock",
                metric_profile=group.metric_profile,
                claim_status=claim_status,
                seed=seed,
                mock_data=True,
            )
        )
    return metrics


def _build_report(
    cfg: Mapping[str, Any],
    audit: SourceAudit,
    *,
    metrics: list[dict[str, Any]],
    output_root: Path,
    mock: bool,
    claim_status: str,
    command: list[str],
) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), Mapping) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), Mapping) else {}
    audit_payload = audit.to_dict()
    warnings = list(audit_payload.get("blocked_reasons", []))
    if dataset_cfg.get("gps_feature_mode") != audit_payload.get("local_substitute", {}).get("gps_feature_mode"):
        warnings.append(
            "local_config_uses_repository_supported_gps_feature_mode_not_author_minmax_lat_lon_exact_protocol"
        )
    file_manifest = [
        "source_audit.json",
        "metrics_summary.json",
        "reproduction_manifest.json",
        "report.md",
    ]
    return {
        "workflow": AMR_NET_GPS_IMAGE_PRESET_NAME,
        "model_name": AMR_NET_GPS_IMAGE_DISPLAY_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mock_data": bool(mock),
        "claim_status": claim_status,
        "source_audit": audit_payload,
        "source_audit_digest": audit_payload["digest"],
        "command": list(command),
        "git_status": _git_status_summary(),
        "scenario": {
            "scene_id": dataset_cfg.get("scene_id", dataset_cfg.get("scene")),
            "scene_slug": dataset_cfg.get("scene_slug"),
            "output_root": str(output_root),
        },
        "enabled_modalities": list(model_cfg.get("modalities", [])),
        "dataset": {
            "type": dataset_cfg.get("type"),
            "data_root": dataset_cfg.get("data_root"),
            "train_csv_name": dataset_cfg.get("train_csv_name"),
            "test_csv_name": dataset_cfg.get("test_csv_name"),
            "gps_feature_mode": dataset_cfg.get("gps_feature_mode"),
            "beam_target_source": dataset_cfg.get("beam_target_source"),
            "use_lidar": bool(dataset_cfg.get("use_lidar", False)),
        },
        "checkpoint_provenance": {
            "checkpoint_path": None,
            "official_weights_status": audit_payload.get("official_weights_status"),
            "mock_data": bool(mock),
        },
        "metric_profile": cfg.get("evaluation", {}).get("metric_profile", "amr_net_gps_image_top1_top3_top5"),
        "model_groups": [group.metadata() for group in paper_model_groups()],
        "metrics": metrics,
        "warnings": warnings,
        "file_manifest": file_manifest,
    }


def _write_report_artifacts(report: Mapping[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "source_audit.json").write_text(
        json.dumps(report["source_audit"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "metrics_summary.json").write_text(
        json.dumps(report["metrics"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "reproduction_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "report.md").write_text(_markdown_report(report), encoding="utf-8")


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report.get('model_name', AMR_NET_GPS_IMAGE_DISPLAY_NAME)} Reproduction Report",
        "",
        f"- Workflow: `{report['workflow']}`",
        f"- Claim status: `{report['claim_status']}`",
        f"- Mock data: `{str(report['mock_data']).lower()}`",
        f"- Source audit digest: `{report['source_audit_digest']}`",
        f"- Scenario: `{report['scenario'].get('scene_slug')}`",
        f"- Enabled modalities: `{', '.join(report.get('enabled_modalities', []))}`",
        "",
        "## Warnings",
    ]
    lines.extend(f"- {item}" for item in report.get("warnings", []))
    lines.extend(["", "## Metrics"])
    for row in report.get("metrics", []):
        lines.append(
            f"- `{row['model_group']}`: top1={row['top1']:.4f}, top3={row['top3']:.4f}, top5={row['top5']:.4f}, DBA={row['DBA']:.4f}"
        )
    lines.append("")
    return "\n".join(lines)


def _git_status_summary() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime metadata.
        return {"available": False, "error": str(exc)}
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return {"available": result.returncode == 0, "dirty": bool(lines), "entries": lines[:50], "entry_count": len(lines)}


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
