import csv
import importlib.util
from pathlib import Path

import torch

from kd_sensing.config.io import load_config
from kd_sensing.registries import ENCODERS, import_default_components


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def test_lightweight_patchvit_frame_encoder_registry_and_shape():
    import_default_components()
    assert "lightweight_patchvit_frame" in ENCODERS.list()
    encoder = ENCODERS.build(
        {
            "type": "lightweight_patchvit_frame",
            "output_dim": 8,
            "latent_dim": 8,
            "image_channels": 3,
            "patch_size": 16,
            "depth": 1,
            "num_heads": 2,
            "max_tokens": 16,
        }
    )

    features = encoder(torch.randn(2, 3, 3, 32, 32))

    assert features.shape == (2, 3, 8)
    metadata = encoder.training_strategy_metadata()
    assert metadata["encoder"] == "lightweight_patchvit_frame"
    assert metadata["visual_token_encoder"]["visual_encoder_type"] == "patch_vit"
    assert metadata["visual_token_encoder"]["token_count"] == 4
    assert metadata["pooling"] == "mean"


def test_scene31_34_patchvit_ablation_generator_sanity(tmp_path: Path):
    generator = _load_script(
        "generate_scenes31_34_patchvit_ablation",
        ROOT / "scripts/generate_scenes31_34_patchvit_ablation.py",
    )
    out_dir = tmp_path / "generated_configs"
    output_dir = tmp_path / "patchvit_outputs"

    assert generator.main(["--out-dir", str(out_dir), "--output-dir", str(output_dir), "--overwrite", "true"]) == 0

    rows = _read_csv(out_dir / "experiment_manifest.csv")
    by_name = {row["run_name"]: row for row in rows}
    assert set(by_name) == {
        "scenes31_34_patchvit_image_pretrain_seed1",
        "scenes31_34_patchvit_lidar_pretrain_seed1",
        "scenes31_34_proto_randomdrop_subset_patchvit_es40_seed1",
        "scenes31_34_proto_randomdrop_subset_patchvit_jepa_es40_seed1",
    }
    assert (out_dir / "run_budget_manifest.json").exists()

    image_cfg = load_config(_manifest_path(by_name["scenes31_34_patchvit_image_pretrain_seed1"]["config_path"]))
    lidar_cfg = load_config(_manifest_path(by_name["scenes31_34_patchvit_lidar_pretrain_seed1"]["config_path"]))
    plain_cfg = load_config(_manifest_path(by_name["scenes31_34_proto_randomdrop_subset_patchvit_es40_seed1"]["config_path"]))
    jepa_cfg = load_config(
        _manifest_path(by_name["scenes31_34_proto_randomdrop_subset_patchvit_jepa_es40_seed1"]["config_path"])
    )

    assert image_cfg["model"]["primary"]["encoders"]["image"]["type"] == "lightweight_patchvit_frame"
    assert lidar_cfg["model"]["primary"]["encoders"]["lidar"]["type"] == "lightweight_patchvit_frame"
    assert image_cfg["model"]["primary"]["encoders"]["image"]["depth"] == 1
    assert lidar_cfg["model"]["primary"]["encoders"]["lidar"]["depth"] == 1
    assert image_cfg["data"]["dataset"]["scenes"] == [31, 32, 33, 34]
    assert lidar_cfg["data"]["dataset"]["scenes"] == [31, 32, 33, 34]
    assert image_cfg["output"]["dir"] == str(output_dir)
    assert lidar_cfg["output"]["dir"] == str(output_dir)

    plain_primary = plain_cfg["model"]["primary"]
    assert plain_primary["encoders"]["image"]["type"] == "lightweight_patchvit_frame"
    assert plain_primary["encoders"]["lidar"]["type"] == "lightweight_patchvit_frame"
    assert plain_primary["encoder_checkpoint_paths"]["image"].endswith(
        "scenes31_34_patchvit_image_pretrain_seed1/checkpoints/best_top1.pth"
    )
    assert plain_primary["encoder_checkpoint_paths"]["lidar"].endswith(
        "scenes31_34_patchvit_lidar_pretrain_seed1/checkpoints/best_top1.pth"
    )
    assert plain_cfg["training"]["random_modality_dropout"]["mode"] == "random_nonempty_subset"
    assert plain_cfg["loss"]["u_mask_beam_jepa"]["enabled"] is False
    assert plain_primary["use_jepa_loss"] is False

    assert jepa_cfg["model"]["primary"]["use_jepa_loss"] is True
    assert jepa_cfg["loss"]["u_mask_beam_jepa"]["enabled"] is True
    assert jepa_cfg["loss"]["u_mask_beam_jepa"]["use_jepa_loss"] is True
    assert jepa_cfg["loss"]["u_mask_beam_jepa"]["lambda_jepa_global"] == 0.1
