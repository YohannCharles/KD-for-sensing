from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.engine.marf_training import (  # noqa: E402
    ModalitySubsetSampler,
    all_to_subset_kl_loss,
    marf_anchor_entropy,
    marf_anchor_prior_regularization_loss,
    marf_residual_norm_loss,
)
from kd_sensing.engine.trainer import train  # noqa: E402
from kd_sensing.engine.validator import validate  # noqa: E402
import kd_sensing.models.fusion.marf  # noqa: E402,F401
from kd_sensing.registries import MODELS  # noqa: E402


def test_marf_subset_sampler_uses_prior_without_hardcoded_modalities():
    sampler = ModalitySubsetSampler(
        ["image", "radar", "lidar"],
        {"image": 0.1, "radar": 0.9, "lidar": 0.4},
        top_prior_k=2,
        min_keep=1,
        random_keep_prob=0.0,
    )
    top = sampler.sample("top_prior")
    best = sampler.sample("single_best_prior")
    random_top = sampler.sample("random_with_top_prior", batch_size=4)
    drop_one = sampler.sample("drop_one", available_mask=torch.ones(4, 3, dtype=torch.bool))

    assert top.modalities == ("radar", "lidar")
    assert best.modalities == ("radar",)
    assert random_top.mask[:, 1].all()
    assert torch.all(random_top.mask.sum(dim=1) >= 1)
    assert torch.all(drop_one.mask.sum(dim=1) == 2)


def test_marf_losses_handle_ignore_index_and_masks():
    anchor = torch.tensor([[[0.7, 0.3], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])
    residual_weights = torch.tensor([[[0.5, 0.0], [0.2, 0.0]], [[0.0, 0.6], [0.0, 0.4]]])
    residual_delta = torch.ones(2, 2, 2, 4)
    mask = torch.tensor([[True, False], [False, True]])
    prior = torch.tensor([0.8, 0.2])
    subset_logits = torch.randn(2, 2, 5, requires_grad=True)
    all_logits = torch.randn(2, 2, 5)
    labels = torch.tensor([[1, -100], [2, 3]])

    assert marf_residual_norm_loss(residual_delta, residual_weights, mask).ndim == 0
    assert marf_anchor_prior_regularization_loss(anchor, prior, mask).ndim == 0
    assert marf_anchor_entropy(anchor, mask).ndim == 0
    assert all_to_subset_kl_loss(subset_logits, all_logits, labels, ignore_index=-100).ndim == 0
    assert all_to_subset_kl_loss(subset_logits, all_logits, torch.full_like(labels, -100)).item() == pytest.approx(0.0)


def test_validation_subset_all_matches_official_path_and_records_modalities():
    torch.manual_seed(0)
    model = MODELS.build(
        {
            "type": "marf_fusion",
            "modalities": ["gps", "mmwave"],
            "feature_size": 8,
            "d_model": 8,
            "num_classes": 4,
            "num_pred": 1,
            "num_heads": 2,
            "gps_input_size": 3,
            "mmwave_input_size": 64,
            "router": {
                "dataset_prior": {"gps": 0.8, "mmwave": 0.2},
                "prior_anchor_scale": 0.0,
                "prior_residual_scale": 0.0,
            },
        }
    )
    model.eval()
    assert model.horizon == 1
    assert model.router.horizon == 1
    cfg = {
        "experiment": {"task": "fusion"},
        "model": {
            "num_pred": 1,
            "downsample_ratio": 1,
            "seq_length_student": 3,
            "num_classes": 4,
            "student": {"modalities": ["gps", "mmwave"]},
        },
        "training": {"transfer": {"non_blocking": False}, "amp": {"enabled": False}},
        "evaluation": {
            "k_values": [1, 2, 3],
            "dba_delta": 5,
            "modality_subsets": {
                "enabled": True,
                "subsets": ["all", "strong_only", "weak_only"],
                "top_prior_k": 1,
            },
        },
    }
    dataloader = [_fixed_batch()]
    metrics = validate(model, dataloader, cfg, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    with torch.no_grad():
        output = model(gps_batch=dataloader[0]["gps"], mmwave_batch=dataloader[0]["mmwave"])
    assert output["logits"].shape[1] == cfg["model"]["num_pred"]
    assert output["anchor_weights"].shape[1] == cfg["model"]["num_pred"]
    assert output["residual_weights"].shape[1] == cfg["model"]["num_pred"]

    official_top1 = metrics["topk"]["1"][0]
    subset_top1 = metrics["modality_subsets"]["all"]["topk"]["1"][0]
    assert len(metrics["topk"]["1"]) == 1
    assert metrics["total"] == [2]
    assert subset_top1 == pytest.approx(official_top1)
    assert metrics["modality_subsets"]["all"]["modalities"] == ["gps", "mmwave"]
    assert metrics["modality_subsets"]["strong_only"]["modalities"] == ["gps", "mmwave"]
    assert metrics["modality_subsets"]["weak_only"]["modalities"] == ["mmwave"]


def test_marf_synthetic_subset_training_logs_losses(tmp_path: Path):
    cfg = load_config(
        ROOT / "configs/fusion/marf_subset_training.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataset.seq_len=2",
            "data.dataset.num_pred=1",
            "data.dataset.num_classes=4",
            "data.dataset.use_gps=true",
            "data.dataset.use_lidar=false",
            "data.dataset.use_mmwave=true",
            "data.dataset.mmwave_normalize=false",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "data.dataloader.persistent_workers=false",
            "model.feature_size=8",
            "model.num_classes=4",
            "model.seq_length_teacher=2",
            "model.seq_length_student=2",
            "model.num_pred=1",
            "model.teacher.modalities=[\"gps\",\"mmwave\"]",
            "model.student.modalities=[\"gps\",\"mmwave\"]",
            "model.student.feature_size=8",
            "model.student.d_model=8",
            "model.student.num_classes=4",
            "model.student.num_pred=1",
            "model.student.num_heads=2",
            "model.student.gps_input_size=3",
            "model.student.mmwave_input_size=64",
            "model.student.router.dataset_prior={\"gps\":0.8,\"mmwave\":0.2}",
            "teacher.registry_path=null",
            "teacher.load_encoders=false",
            "teacher.freeze_encoders=false",
            "loss.beam_soft.enabled=false",
            "loss.beam_soft.weight=0.0",
            "loss.marf.residual_norm.weight=0.01",
            "loss.marf.prior_regularization.weight=0.01",
            "training.epochs=1",
            "training.subset_training.enabled=true",
            "training.subset_training.modes=[\"top_prior\",\"random_with_top_prior\"]",
            "training.subset_training.ce_weight=0.1",
            "training.subset_training.kd_weight=0.1",
            "scheduler.type=none",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.run_name=marf_subset_smoke",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )
    result = train(cfg)
    first_epoch = result["epoch_logs"][0]

    assert first_epoch["train_batches"] == 2
    assert first_epoch["loss/marf_subset_ce"] >= 0.0
    assert first_epoch["loss/marf_subset_kd"] >= 0.0
    assert "marf/anchor_mean/gps" in first_epoch
    assert "val/subset/all/top1" in first_epoch
    assert first_epoch["val/subset/all/top1"] == pytest.approx(first_epoch["val_acc"])


def _fixed_batch() -> dict[str, torch.Tensor]:
    return {
        "gps": torch.randn(2, 3, 3),
        "mmwave": torch.randn(2, 3, 64),
        "input_beam": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        "target_beam": torch.tensor([[1], [2]], dtype=torch.long),
    }
