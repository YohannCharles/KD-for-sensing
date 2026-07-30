import pytest
import torch

from kd_sensing.engine.optim import build_optimizer


@pytest.mark.parametrize(("name", "expected"), [("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)])
def test_build_optimizer_uses_only_trainable_parameters(name, expected) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    model[0].requires_grad_(False)

    optimizer = build_optimizer(
        {"training": {"lr": 0.01, "weight_decay": 0.02, "optimizer": {"type": name}}},
        model,
    )

    assert isinstance(optimizer, expected)
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == {
        id(parameter) for parameter in model[1].parameters()
    }
