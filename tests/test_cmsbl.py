import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from kd_sensing.config import load_config
from kd_sensing.engine.runtime import run_model_step
from kd_sensing.engine.tensorboard_logging import write_tensorboard_method_scalars
from kd_sensing.engine.training_extensions import BatchState, ExtensionContext
from kd_sensing.losses.bcacl_config import primary_model_config_with_bcacl, resolve_bcacl_config
from kd_sensing.losses.cmsbl import (
    NUM_NON_EMPTY_MASKS,
    accumulate_mask_losses,
    auxiliary_schedule_weight,
    capacity_gap_weights,
    fusion_mask_ids,
    hard_mask_weights,
    load_capacity_reference,
    update_mask_loss_ema,
    update_metric_ema,
)
from kd_sensing.losses.cmsbl_config import resolve_cmsbl_config
from kd_sensing.losses.u_mask_beam_jepa import UMaskBeamJEPATrainingExtension, u_mask_beam_jepa_loss
from kd_sensing.registries import ENCODERS, MODELS

import kd_sensing.models.u_mask_beam_jepa  # noqa: F401


MODALITIES = ("image", "radar", "gps", "lidar")
ROOT = Path(__file__).resolve().parents[1]


@ENCODERS.register("cmsbl_test_sequence", force=True)
class _SequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 4, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.projection = nn.Linear(1, self.output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scalar = values.float().reshape(*values.shape[:2], -1).mean(dim=-1, keepdim=True)
        return self.projection(scalar)


def _primary() -> dict:
    return {
        "type": "u_mask_beam_jepa",
        "modalities": list(MODALITIES),
        "d_model": 4,
        "num_classes": 4,
        "num_pred": 1,
        "seq_length": 2,
        "dropout": 0.0,
        "fusion_type": "supervised_router",
        "head_type": "prototype",
        "temporal_pooling": {"enabled": True, "type": "masked_mean"},
        "encoders": {name: {"type": "cmsbl_test_sequence", "output_dim": 4} for name in MODALITIES},
    }


def _bcacl_raw() -> dict:
    return {
        "enabled": True,
        "training_regime": "aux_joint",
        "stage": "aux_joint",
        "projection": {"dim": 6, "layer_norm": True, "dropout": 0.0},
        "private_heads": {"enabled": True},
        "shared_head": {"enabled": True},
        "lambda_shared": 1.0,
    }


def _capacity_file(path: Path, *, split: str = "inner_train") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "cmsbl_capacity_reference_v1",
                "dataset": "mmw",
                "source_split": split,
                "metric": "top1",
                "source_sha256": "a" * 64,
                "modalities": {
                    "image": {"top1": 0.8},
                    "radar": {"top1": 0.7},
                    "gps": {"top1": 0.6},
                    "lidar": {"top1": 0.9},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _cmsbl_raw(stats_path: Path | None = None) -> dict:
    return {
        "enabled": True,
        "aux_schedule": {
            "enabled": True,
            "private": {"start_weight": 1.0, "end_weight": 0.5, "start_epoch": 2, "end_epoch": 4},
            "shared": {"start_weight": 0.8, "end_weight": 0.4, "start_epoch": 2, "end_epoch": 4},
        },
        "capacity_reference": {
            "stats_path": str(stats_path) if stats_path is not None else None,
            "source_split": "inner_train",
        },
        "capacity_gap": {
            "enabled": stats_path is not None,
            "warmup_epochs": 0,
            "ema_momentum": 0.5,
            "min_weight": 1.0,
            "max_weight": 2.0,
        },
        "hard_mask": {
            "enabled": True,
            "warmup_epochs": 0,
            "min_count": 1,
            "ema_momentum": 0.5,
        },
        "diagnostics": {"enabled": True},
    }


def _config(stats_path: Path | None = None) -> dict:
    return {
        "model": {"primary": _primary()},
        "data": {"dataset": {"type": "mmw"}},
        "temporal_missing": {"preserve_unmasked_for_superset": True},
        "loss": {
            "u_mask_beam_jepa": {
                "enabled": True,
                "router_oracle_weight": 0.0,
                "missing_mask": {"mode": "external"},
                "use_beam_prototype_alignment": True,
                "lambda_proto": 0.2,
                "lambda_modality_proto": 0.1,
            }
        },
        "bcacl": _bcacl_raw(),
        "cmsbl": _cmsbl_raw(stats_path),
    }


def test_linear_schedule_clamps_and_interpolates() -> None:
    schedule = {"start_weight": 1.0, "end_weight": 0.2, "start_epoch": 2, "end_epoch": 6}
    assert auxiliary_schedule_weight(schedule, 1) == pytest.approx(1.0)
    assert auxiliary_schedule_weight(schedule, 4) == pytest.approx(0.6)
    assert auxiliary_schedule_weight(schedule, 7) == pytest.approx(0.2)
    constant = {**schedule, "start_weight": 0.5, "end_weight": 0.5}
    assert auxiliary_schedule_weight(constant, 4) == pytest.approx(0.5)


def test_config_has_no_fake_mode_switches(tmp_path: Path) -> None:
    cfg = _config(_capacity_file(tmp_path / "capacity.json"))
    bcacl = resolve_bcacl_config(cfg)
    resolved = resolve_cmsbl_config(cfg, bcacl)
    assert resolved["capacity_reference"]["metric"] == "top1"
    assert resolved["aux_schedule"]["private"]["start_epoch"] == 2

    cfg["cmsbl"]["hard_mask"]["sampling_reweighting"] = True
    with pytest.raises(ValueError, match="does not support fields"):
        resolve_cmsbl_config(cfg, bcacl)


def test_canonical_recipe_strict_overrides_can_enable_v1_and_v3() -> None:
    cfg = load_config(
        ROOT / "configs/mmw/t2.yaml",
        overrides=(
            "bcacl.enabled=true",
            "cmsbl.enabled=true",
            "cmsbl.aux_schedule.enabled=true",
            "cmsbl.hard_mask.enabled=true",
        ),
    )
    resolved = resolve_cmsbl_config(cfg, resolve_bcacl_config(cfg))

    assert resolved["aux_schedule"]["enabled"] is True
    assert resolved["hard_mask"]["enabled"] is True
    assert resolved["capacity_gap"]["enabled"] is False


def test_capacity_reference_provenance_ema_and_gap_weights(tmp_path: Path) -> None:
    path = _capacity_file(tmp_path / "capacity.json")
    cfg = resolve_cmsbl_config(_config(path), resolve_bcacl_config(_config(path)))
    reference, identity = load_capacity_reference(
        cfg["capacity_reference"], dataset="mmw", modalities=MODALITIES
    )
    ema, initialized = update_metric_ema(
        torch.zeros(4),
        torch.zeros(4, dtype=torch.bool),
        torch.tensor([0.4, 0.7, 0.8, 0.3]),
        torch.ones(4, dtype=torch.bool),
        momentum=0.5,
    )
    weights, gaps = capacity_gap_weights(
        reference,
        ema,
        initialized,
        epoch_number=1,
        config=cfg["capacity_gap"],
    )

    assert identity["source_sha256"] == "a" * 64
    assert weights[0] > 1 and weights[1] == pytest.approx(1.0)
    assert gaps[2] == pytest.approx(0.0)

    _capacity_file(path, split="outer_test")
    with pytest.raises(ValueError, match="outer/test"):
        load_capacity_reference(cfg["capacity_reference"], dataset="mmw", modalities=MODALITIES)


def test_mask_ids_ema_and_weights_cover_the_15_non_empty_patterns() -> None:
    masks = torch.tensor(
        [[bool(value & (1 << index)) for index in range(4)] for value in range(1, 16)]
    )
    ids = fusion_mask_ids(masks, MODALITIES)
    sums = torch.zeros(NUM_NON_EMPTY_MASKS, dtype=torch.float64)
    counts = torch.zeros(NUM_NON_EMPTY_MASKS, dtype=torch.long)
    losses = torch.arange(1, 16, dtype=torch.float64)
    accumulate_mask_losses(sums, counts, losses, ids)
    ema, initialized, cumulative = update_mask_loss_ema(
        torch.zeros(15),
        torch.zeros(15, dtype=torch.bool),
        torch.zeros(15, dtype=torch.long),
        sums.float(),
        counts,
        momentum=0.5,
        min_count=1,
    )
    weights, _ = hard_mask_weights(
        ema,
        cumulative,
        initialized,
        epoch_number=1,
        config={
            "warmup_epochs": 0,
            "min_count": 1,
            "gamma": 0.5,
            "min_weight": 0.75,
            "max_weight": 1.75,
            "normalize_mean_to_one": True,
            "full_mask_min_weight": 1.0,
            "eps": 1e-6,
        },
    )

    assert ids.tolist() == list(range(1, 16))
    assert cumulative.tolist() == [1] * 15
    assert weights[-1] > weights[0]
    assert weights.mean() == pytest.approx(1.0, abs=1e-5)


def test_disabled_weighting_keeps_the_scalar_loss_path() -> None:
    logits = torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]], requires_grad=True)
    output = {
        "logits": logits,
        "output_features": torch.randn(2, 3),
        "modality_features": torch.randn(2, 1, 3),
        "missing_mask": torch.ones(2, 1, dtype=torch.bool),
    }
    first = u_mask_beam_jepa_loss(output, torch.tensor([[0], [1]]), router_oracle_weight=0.0)
    second = u_mask_beam_jepa_loss(output, torch.tensor([[0], [1]]), router_oracle_weight=0.0)
    assert torch.equal(first["loss"], second["loss"])
    assert resolve_cmsbl_config({}, {}) == {"enabled": False}


def test_cmsbl_epoch_metrics_reuse_the_existing_tensorboard_writer() -> None:
    class Writer:
        def __init__(self) -> None:
            self.values = []
            self.flushed = False

        def add_scalar(self, key: str, value: float, step: int) -> None:
            self.values.append((key, value, step))

        def flush(self) -> None:
            self.flushed = True

    writer = Writer()
    write_tensorboard_method_scalars(
        writer,
        {"cmsbl/lambda_private": 0.75, "unrelated": 1.0},
        3,
    )
    assert writer.values == [("cmsbl/lambda_private", 0.75, 3)]
    assert writer.flushed is True


def test_extension_updates_train_state_writes_one_json_and_resumes(tmp_path: Path) -> None:
    capacity = _capacity_file(tmp_path / "capacity.json")
    cfg = _config(capacity)
    model = MODELS.build(primary_model_config_with_bcacl(cfg)).train()
    context = ExtensionContext(
        cfg=cfg,
        task="fusion",
        model_cfg=cfg["model"],
        training_cfg={},
        primary_model=model,
        task_criterion=nn.CrossEntropyLoss(),
        run_dir=tmp_path,
        device=torch.device("cpu"),
        num_pred=1,
        num_classes=4,
        seq_length=2,
        non_blocking=False,
    )
    extension = UMaskBeamJEPATrainingExtension()
    state = extension.setup(context)
    fusion_temporal = torch.tensor(
        [
            [[True, True, True, True], [True, True, True, True]],
            [[False, True, True, False], [False, True, True, False]],
        ]
    )
    original = {
        "image": torch.ones(2, 2, 3, 2, 2),
        "radar_ra": torch.ones(2, 2, 1, 128, 64),
        "radar_da": torch.ones(2, 2, 1, 128, 64),
        "gps": torch.ones(2, 2, 3),
        "lidar": torch.ones(2, 2, 3, 2, 2),
    }
    batch = {
        **original,
        "target_beam": torch.tensor([0, 1]),
        "modality_temporal_mask": fusion_temporal,
        "available_modalities": fusion_temporal.any(dim=1),
        "temporal_superset_payload": {
            "inputs": original,
            "base_mask": torch.ones(2, 2, 4, dtype=torch.bool),
            "modalities": MODALITIES,
        },
    }
    labels = torch.tensor([[0], [1]])
    controls = extension.before_forward(context, state, batch, labels, epoch=0, step=0)
    step = run_model_step(
        model,
        "fusion",
        batch,
        seq_length=2,
        num_pred=1,
        device=torch.device("cpu"),
        extra_model_kwargs=controls.model_kwargs,
    )
    result = extension.compute_base_loss(
        context,
        state,
        BatchState(
            epoch=0,
            step=0,
            batch=batch,
            labels=labels,
            primary_output=step.model_output,
            primary_logits=step.logits,
            controls=controls,
        ),
    )
    assert result is not None and torch.isfinite(result.total_loss)
    result.total_loss.backward()
    metrics = extension.after_epoch(context, state, epoch=0)

    assert metrics["cmsbl/mask/15/count"] == 1.0
    diagnostic = tmp_path / "cmsbl" / "epoch_0001.json"
    assert diagnostic.is_file()
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert payload["state_source"] == "train_only"
    assert list((tmp_path / "cmsbl").iterdir()) == [diagnostic]

    saved = extension.state_dict(state)
    restored = extension.setup(context)
    restored["cmsbl_epoch_mask_counts"].fill_(3)
    extension.load_state_dict(restored, saved)
    assert torch.equal(restored["cmsbl_metric_ema"], state["cmsbl_metric_ema"])
    assert torch.equal(restored["cmsbl_mask_counts"], state["cmsbl_mask_counts"])
    assert torch.count_nonzero(restored["cmsbl_epoch_mask_counts"]) == 0
