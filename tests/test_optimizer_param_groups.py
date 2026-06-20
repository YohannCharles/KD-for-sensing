import pytest
import torch
import torch.nn as nn

from kd_sensing.engine.optim import build_optimizer, optimizer_param_group_summary


class _TinyGroupedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                "image": nn.Linear(4, 4),
                "gps": nn.Linear(4, 4),
            }
        )
        self.pooler = nn.Linear(4, 4)
        self.heads = nn.ModuleDict({"beam": nn.Linear(4, 2)})


def _cfg(parameter_groups=None, **optimizer_overrides):
    optimizer_cfg = {"type": "adam"}
    if parameter_groups is not None:
        optimizer_cfg["parameter_groups"] = parameter_groups
    optimizer_cfg.update(optimizer_overrides)
    return {
        "training": {
            "lr": 0.001,
            "weight_decay": 0.01,
            "optimizer": optimizer_cfg,
        }
    }


def test_optimizer_parameter_groups_build_named_groups_and_summary():
    model = _TinyGroupedModel()
    optimizer = build_optimizer(
        _cfg(
            [
                {
                    "name": "image_encoder",
                    "module_patterns": ["encoders.image"],
                    "lr": 0.0001,
                    "weight_decay": 0.001,
                },
                {
                    "name": "gps_encoder",
                    "module_patterns": ["encoders.gps"],
                    "lr": 0.0002,
                    "weight_decay": 0.002,
                },
            ]
        ),
        model,
    )

    summary = optimizer_param_group_summary(optimizer)

    assert [group["name"] for group in summary] == ["image_encoder", "gps_encoder", "main"]
    assert [group["lr"] for group in summary] == [0.0001, 0.0002, 0.001]
    assert [group["weight_decay"] for group in summary] == [0.001, 0.002, 0.01]
    assert summary[0]["param_count"] == sum(param.numel() for param in model.encoders["image"].parameters())
    assert summary[1]["param_count"] == sum(param.numel() for param in model.encoders["gps"].parameters())
    assert summary[2]["param_count"] == sum(param.numel() for param in model.pooler.parameters()) + sum(
        param.numel() for param in model.heads.parameters()
    )


def test_optimizer_parameter_groups_reject_unmatched_duplicate_and_required_all_matched():
    with pytest.raises(ValueError, match="did not match any trainable parameters.*missing.branch"):
        build_optimizer(
            _cfg(
                [
                    {
                        "name": "missing",
                        "module_patterns": ["missing.branch"],
                    }
                ]
            ),
            _TinyGroupedModel(),
        )

    with pytest.raises(ValueError, match="matched by multiple optimizer parameter groups.*encoders.image.weight"):
        build_optimizer(
            _cfg(
                [
                    {"name": "all_encoders", "module_patterns": ["encoders"]},
                    {"name": "image_encoder", "module_patterns": ["encoders.image"]},
                ]
            ),
            _TinyGroupedModel(),
        )

    with pytest.raises(ValueError, match="require_all_matched.*pooler.weight"):
        build_optimizer(
            _cfg(
                [
                    {
                        "name": "image_encoder",
                        "module_patterns": ["encoders.image"],
                    }
                ],
                require_all_matched=True,
            ),
            _TinyGroupedModel(),
        )


def test_optimizer_parameter_groups_keep_default_main_group_and_allow_non_strict_patterns():
    default_optimizer = build_optimizer({"training": {"lr": 0.003, "weight_decay": 0.004}}, _TinyGroupedModel())
    default_summary = optimizer_param_group_summary(default_optimizer)

    assert len(default_summary) == 1
    assert default_summary[0]["name"] == "main"
    assert default_summary[0]["lr"] == 0.003
    assert default_summary[0]["weight_decay"] == 0.004

    relaxed_optimizer = build_optimizer(
        _cfg(
            [
                {
                    "name": "image_encoder",
                    "module_patterns": ["encoders.image", "missing.branch"],
                    "lr": 0.0001,
                }
            ],
            strict=False,
        ),
        _TinyGroupedModel(),
    )
    relaxed_summary = optimizer_param_group_summary(relaxed_optimizer)

    assert [group["name"] for group in relaxed_summary] == ["image_encoder", "main"]
    assert relaxed_summary[0]["lr"] == 0.0001
