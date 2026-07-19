import importlib.util
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_mmw_tie_aware_router_screen",
    ROOT / "scripts/launch_mmw_tie_aware_router_screen.py",
)
assert SPEC is not None and SPEC.loader is not None
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)

SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_mmw_tie_aware_router_screen",
    ROOT / "scripts/summarize_mmw_tie_aware_router_screen.py",
)
assert SUMMARY_SPEC is not None and SUMMARY_SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(SUMMARY)


def _config() -> dict:
    return {
        "model": {"primary": {"fusion_type": "supervised_router"}},
        "loss": {
            "u_mask_beam_jepa": {
                "router_oracle_weight": 0.1,
                "router_oracle_target_mode": "hard_first",
                "router_oracle_temperature": 1.0,
            }
        },
    }


def test_screen_registers_exactly_eight_gpu_candidates() -> None:
    assert len(SCREEN.CANDIDATES) == 8
    assert set(SCREEN.CANDIDATES) == {
        "HardFirstControl",
        "HardConfidenceTie",
        "SoftUniformTie",
        "SoftConfidenceTie",
        "DistanceSoftT05",
        "DistanceSoftT10",
        "DistanceConfidenceT10",
        "UniformFusion",
    }


def test_candidate_overlay_changes_only_router_target_or_uniform_fusion() -> None:
    soft = _config()
    SCREEN.apply_candidate(soft, "DistanceSoftT05", SCREEN.CANDIDATES["DistanceSoftT05"])
    assert soft["model"]["primary"]["fusion_type"] == "supervised_router"
    assert soft["loss"]["u_mask_beam_jepa"]["router_oracle_target_mode"] == "distance_soft"
    assert soft["loss"]["u_mask_beam_jepa"]["router_oracle_temperature"] == 0.5

    uniform = _config()
    SCREEN.apply_candidate(uniform, "UniformFusion", SCREEN.CANDIDATES["UniformFusion"])
    assert uniform["model"]["primary"]["fusion_type"] == "uniform_mean"
    assert uniform["loss"]["u_mask_beam_jepa"]["router_oracle_weight"] == 0.0


def test_adba_first_summary_ranks_adba_and_keeps_top1_secondary(tmp_path: Path) -> None:
    screen_dir = tmp_path / "screen"
    jobs = []
    for candidate_index, candidate in enumerate(SUMMARY.CANDIDATES):
        config = screen_dir / "configs" / f"{candidate}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("evaluation:\n  dba_delta: 5\n  dba_distance_mode: circular\n", encoding="utf-8")
        checkpoint = screen_dir / candidate / "seed1" / "checkpoints" / "last.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(candidate.encode())
        rows = []
        for domain_index in range(15):
            domain = f"weather/scene{domain_index}"
            patterns = [
                ("full", "image,radar,gps,lidar", "0", "0.0", "whole_modality", "whole_modality"),
                ("drop1", "image,radar,gps", "1", "0.0", "whole_modality", "whole_modality"),
                ("drop2", "image,radar", "2", "0.0", "whole_modality", "whole_modality"),
                ("drop3", "image", "3", "0.0", "whole_modality", "whole_modality"),
                ("block", "image,radar,gps,lidar", "0", "0.8", "block", "temporal_missing"),
                ("token20", "image,radar,gps,lidar", "0", "0.2", "modality_frame", "temporal_missing"),
                ("token40", "image,radar,gps,lidar", "0", "0.4", "modality_frame", "temporal_missing"),
                ("token60", "image,radar,gps,lidar", "0", "0.6", "modality_frame", "temporal_missing"),
                ("token80", "image,radar,gps,lidar", "0", "0.8", "modality_frame", "temporal_missing"),
            ]
            for mask_index, (pattern, available, drop, rate, mask_type, family) in enumerate(patterns):
                rows.append(
                    {
                        "method": candidate,
                        "seed": "1",
                        "domain_id": domain,
                        "sample_csv": str(screen_dir / "splits" / domain / "inner_validation.csv"),
                        "sample_csv_sha256": f"sample-{domain_index}",
                        "eval_family": family,
                        "pattern": pattern,
                        "available_modalities": available,
                        "missing_rate": rate,
                        "drop_count": drop,
                        "mask_index": str(mask_index),
                        "mask_type": mask_type,
                        "mask_digest": f"mask-{mask_index}",
                        "mask_cache_checksum": "cache",
                        "mask_cache_seed": "7",
                        "sample_count": "10",
                        "expected_sample_count": "10",
                        "coverage_status": "complete",
                        "partial_request": "False",
                        "screening_role": "local_validation",
                        "checkpoint_role": "last",
                        "checkpoint": str(checkpoint),
                        "dba_distance_mode": "circular",
                        "metric_profile": "progressive_top3_dba_v1",
                        "adba": str(0.7 + candidate_index * 0.01),
                        "top1": str(0.8 - candidate_index * 0.01),
                    }
                )
        while len(rows) < SUMMARY.EXPECTED_ROWS:
            clone = dict(rows[len(rows) % (15 * len(patterns))])
            clone["mask_index"] = str(len(rows))
            clone["mask_digest"] = f"extra-mask-{len(rows)}"
            rows.append(clone)
        metrics = screen_dir / "eval_inner" / candidate / "metrics.csv"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        with metrics.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        jobs.append(
            {
                "method": candidate,
                "status": "done",
                "evaluation_status": "done",
                "training_return_code": 0,
                "evaluation_return_code": 0,
                "config_path": str(config),
            }
        )
    manifest = {
        "protocol": SUMMARY.PROTOCOL_ID,
        "request": {
            "protocol_id": SUMMARY.PROTOCOL_ID,
            "claim_eligible": False,
            "selection_split": "frozen_inner_validation_only",
        },
        "jobs": jobs,
    }
    (screen_dir / "training_manifest_tie_aware_seed1.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = SUMMARY.summarize(screen_dir, tmp_path / "summary", bootstrap_iterations=20)

    assert result["table"][0]["candidate"] == "UniformFusion"
    assert result["table"][0]["top1_Main5"] < result["table"][-1]["top1_Main5"]
    assert result["provenance"]["primary_metric"] == "adba"
    assert result["provenance"]["secondary_metric"] == "top1"
