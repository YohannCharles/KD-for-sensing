import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kd_sensing.data.mmw.codebook_topology import (
    NUM_BEAMS,
    _audit_domain,
    _edge_rows,
    _rsu_yaw,
    make_ula_dft_codebook,
    wrapped_spatial_frequency,
)
from kd_sensing.data.mmw.preparation_splits import compute_split_leakage_diagnostics
from kd_sensing.data.samples import create_samples
from kd_sensing.data.mmw.twc_evidence import build_confirmation_train_domains, build_fixed_mask_cache, prepare_protocol
from kd_sensing.losses.beam_prototype_alignment import make_soft_beam_labels
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config


def test_fixed_mask_cache_is_deterministic_and_contains_required_families():
    first = build_fixed_mask_cache(seed=20260717)
    second = build_fixed_mask_cache(seed=20260717)

    assert first["checksum"] == second["checksum"]
    assert {item["family"] for item in first["conditions"]} == {
        "whole_modality",
        "temporal_missing",
        "joint_missing",
    }
    assert len({(item["family"], item["pattern"], item["mask_digest"]) for item in first["conditions"]}) == len(
        first["conditions"]
    )
    assert any(item["family"] == "joint_missing" and item["drop_count"] == 2 for item in first["conditions"])


def test_mmw_samples_retain_future_beam_path_for_communication_metrics(tmp_path: Path):
    csv_path = tmp_path / "outer.csv"
    future = tmp_path / "future.txt"
    gps = tmp_path / "gps.yaml"
    bs_gps = tmp_path / "bs_gps.yaml"
    np.savetxt(future, np.arange(64, dtype=np.float32))
    gps.write_text("{}\n", encoding="utf-8")
    bs_gps.write_text("{}\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gps1", "bs_gps1", "future_beam1", "future_beam_label1"])
        writer.writeheader()
        writer.writerow(
            {"gps1": "gps.yaml", "bs_gps1": "bs_gps.yaml", "future_beam1": "future.txt", "future_beam_label1": "0"}
        )

    samples = create_samples(csv_path, data_root=tmp_path, enabled_modalities=("gps",), seq_len=1, num_pred=1)

    assert samples.rows is not None
    assert samples.rows[0]["future_beam1"] == "future.txt"


def test_split_leakage_identity_includes_contiguous_segment():
    left = {"contiguous_segment_id": "agent-a", "window_start_frame": "10", "window_end_frame": "15"}
    other_agent = {"contiguous_segment_id": "agent-b", "window_start_frame": "10", "window_end_frame": "15"}
    same_agent = {"contiguous_segment_id": "agent-a", "window_start_frame": "10", "window_end_frame": "15"}

    assert compute_split_leakage_diagnostics([left], [other_agent])["train_test_frame_overlap_count"] == 0
    assert compute_split_leakage_diagnostics([left], [same_agent])["train_test_frame_overlap_count"] == 6


def test_prepare_protocol_is_idempotent_and_detects_cache_tampering(tmp_path: Path):
    domains = _write_domains(tmp_path)
    output = tmp_path / "outputs/cache/protocol"

    first = prepare_protocol(output, project_root=tmp_path, domains=domains, split_seed=17, mask_seed=23)
    second = prepare_protocol(output, project_root=tmp_path, domains=domains, split_seed=17, mask_seed=23)

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert len(first["domains"]) == 15
    for domain in first["domains"]:
        counts = {role: domain["split"][role]["row_count"] for role in ("inner_train", "inner_validation", "outer_evidence")}
        assert all(count > 0 for count in counts.values())
        assert all(item["passed"] for item in domain["partition"]["pair_audits"].values())

    confirmation, confirmation_manifest = build_confirmation_train_domains(first, tmp_path / "outputs/confirmation")
    repeated, repeated_manifest = build_confirmation_train_domains(first, tmp_path / "outputs/confirmation")
    assert len(confirmation) == len(repeated) == 15
    assert confirmation_manifest["manifest_sha256"] == repeated_manifest["manifest_sha256"]
    assert all(Path(item["train_csv_name"]).is_file() for item in confirmation)

    cache = Path(first["fixed_mask_cache"]["path"])
    cache.write_text(cache.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fixed-mask cache changed"):
        prepare_protocol(output, project_root=tmp_path, domains=domains, split_seed=17, mask_seed=23)


def test_ula_dft_endpoint_is_one_local_phase_bin_and_metadata_mismatch_is_rejected(tmp_path: Path):
    codebook = make_ula_dft_codebook()
    edges = _edge_rows(codebook)
    endpoint = next(item for item in edges if item["left_label"] == NUM_BEAMS - 1 and item["right_label"] == 0)

    assert codebook.shape == (64, 64)
    assert wrapped_spatial_frequency(0) == pytest.approx(0.0)
    assert wrapped_spatial_frequency(63) == pytest.approx(1.0 / 32.0)
    assert endpoint["phase_gap_bins"] == pytest.approx(1.0 / NUM_BEAMS)
    assert endpoint["beampattern_overlap"] > 0.0

    data_root = tmp_path / "dataset/MMW/sunny"
    scene = "scene_a"
    metadata = data_root / "Prepared" / scene / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(
            {
                "channel_to_beam": {
                    "algorithm_version": "wrong",
                    "codebook_type": "ula_dft",
                    "num_beams": 64,
                    "tx_antennas": 64,
                    "rx_antennas": 1,
                    "mappings": [{}],
                }
            }
        ),
        encoding="utf-8",
    )
    config = data_root / "Sensor_Data" / scene / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("scenarios: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata mismatch"):
        _audit_domain(
            tmp_path,
            {"id": "sunny/scene_a", "condition": "sunny", "scene": scene, "data_root": "dataset/MMW/sunny"},
            codebook,
            replay_samples_per_domain=1,
            endpoint_samples_per_domain=1,
        )


def test_rsu_yaw_is_audit_only_input(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "scenarios:\n  scene_a:\n    rsu_transform:\n      rotation:\n        yaw: -90\n",
        encoding="utf-8",
    )

    assert _rsu_yaw(config) == pytest.approx(-90.0)
    # Codebook labels remain defined by their local DFT coordinates, not this yaw.
    assert wrapped_spatial_frequency(63) == pytest.approx(1.0 / 32.0)


def test_explicit_bpa_topology_validates_physical_and_permuted_descriptors():
    cfg = _bpa_config(
        {
            "id": "ula_dft_phase_cycle_v1",
            "descriptor_sha256": "a" * 64,
            "audit_path": "outputs/cache/mmw_codebook_topology/v1/audit/topology_manifest.json",
        }
    )
    resolved = u_mask_beam_jepa_config(cfg)
    assert resolved["prototype_topology"]["id"] == "ula_dft_phase_cycle_v1"
    assert resolved["prototype_target_circular"] is True

    permutation = list(range(64))
    permutation[1], permutation[32] = permutation[32], permutation[1]
    permuted = make_soft_beam_labels(
        torch.tensor([0]),
        64,
        1.0,
        circular=True,
        topology_id="permuted_index_v1",
        topology_permutation=permutation,
    )
    cyclic = make_soft_beam_labels(torch.tensor([0]), 64, 1.0, circular=True, topology_id="cyclic_index_v1")
    assert float(permuted[0, 32]) > float(permuted[0, 1])
    assert float(cyclic[0, 1]) > float(cyclic[0, 32])

    invalid = _bpa_config({"id": "permuted_index_v1", "permutation": list(range(63))})
    with pytest.raises(ValueError, match="64-label bijection"):
        u_mask_beam_jepa_config(invalid)


def _write_domains(root: Path) -> list[dict[str, str]]:
    domains = []
    for index in range(15):
        condition = f"weather{index // 5}"
        scene = f"scene{index}"
        source = root / "dataset/MMW" / condition / "Prepared" / scene / "splits/h5p1_strict_v2/train_with_radar_with_bs_gps.csv"
        source.parent.mkdir(parents=True)
        rows = []
        for row_index, start in enumerate(range(0, 60, 12)):
            frames = list(range(start, start + 6))
            rows.append(
                {
                    "condition": condition,
                    "sensor_scenario": scene,
                    "scene_slug": scene,
                    "contiguous_segment_id": f"{condition}:{scene}:segment",
                    "window_start_frame": str(start),
                    "window_end_frame": str(start + 5),
                    "window_frame_ids_json": json.dumps(frames),
                    "sample_id": f"sample-{row_index}",
                    "target_sample_id": f"target-{row_index}",
                    "camera1": f"camera/{row_index}.png",
                    "radar1": f"radar/{row_index}.npy",
                    "gps1": f"gps/{row_index}.yaml",
                    "bs_gps1": f"bs/{row_index}.yaml",
                    "lidar1": f"lidar/{row_index}.npy",
                    "beam1": f"beam/{row_index}.txt",
                    "future_beam1": f"future/{row_index}.txt",
                    "future_beam_label1": str(row_index % 64),
                }
            )
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        domains.append(
            {
                "id": f"{condition}/{scene}",
                "condition": condition,
                "scene": scene,
                "data_root": f"dataset/MMW/{condition}",
                "source_csv": str(source),
            }
        )
    return domains


def _bpa_config(topology: dict) -> dict:
    return {
        "model": {"primary": {"head_type": "prototype", "fusion_type": "supervised_router"}},
        "loss": {
            "u_mask_beam_jepa": {
                "enabled": True,
                "use_beam_prototype_alignment": True,
                "lambda_proto": 0.2,
                "lambda_modality_proto": 0.1,
                "prototype_topology": topology,
            }
        },
    }
