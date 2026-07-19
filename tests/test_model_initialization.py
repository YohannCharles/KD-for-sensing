from pathlib import Path

import pytest
import torch
import torch.nn as nn

from kd_sensing.engine.model_initialization import (
    enforce_frozen_module_eval,
    initialize_model_from_checkpoint,
)
from kd_sensing.utils.checkpoint import CheckpointLoadError, checkpoint_file_digest, publish_checkpoint


class _SourceModel(nn.Module):
    def __init__(self, *, expert_width: int = 3) -> None:
        super().__init__()
        self.expert = nn.Sequential(
            nn.Linear(2, expert_width),
            nn.BatchNorm1d(expert_width),
            nn.Dropout(0.5),
        )
        self.current_router = nn.Linear(expert_width, 1)


class _CandidateModel(_SourceModel):
    def __init__(self, *, expert_width: int = 3) -> None:
        super().__init__(expert_width=expert_width)
        self.candidate_router = nn.Linear(expert_width, 1)


def _checkpoint(tmp_path: Path, model: nn.Module, *, role: str = "last") -> tuple[Path, str]:
    path, _ = publish_checkpoint(
        {
            "checkpoint_schema_version": 1,
            "checkpoint_role": role,
            "state_dict": model.state_dict(),
        },
        tmp_path,
        "last.pth",
    )
    digest, _ = checkpoint_file_digest(path)
    return path, digest


def _training_config(path: Path, digest: str, **overrides) -> dict:
    initialization = {
        "path": str(path),
        "sha256": digest,
        "role": "last",
        "checkpoint_schema_version": 1,
        "required_prefixes": ["expert", "current_router"],
        "allowed_missing_prefixes": ["candidate_router"],
        "freeze_prefixes": ["expert", "current_router"],
    }
    initialization.update(overrides)
    return {"resume": False, "initialization_checkpoint": initialization}


def test_initialization_loads_allowlisted_missing_and_freezes_expert_runtime(tmp_path: Path) -> None:
    torch.manual_seed(1)
    source = _SourceModel()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
    checkpoint, digest = _checkpoint(tmp_path, source)
    torch.manual_seed(2)
    candidate = _CandidateModel()
    candidate_before = candidate.candidate_router.weight.detach().clone()

    load_info = initialize_model_from_checkpoint(candidate, _training_config(checkpoint, digest))

    assert load_info is not None
    assert load_info["role"] == "initialization"
    assert load_info["source_sha256"] == digest
    assert load_info["missing_keys"] == ["candidate_router.bias", "candidate_router.weight"]
    assert torch.equal(candidate.expert[0].weight, source.expert[0].weight)
    assert torch.equal(candidate.current_router.weight, source.current_router.weight)
    assert torch.equal(candidate.candidate_router.weight, candidate_before)
    assert all(not parameter.requires_grad for parameter in candidate.expert.parameters())
    assert all(not parameter.requires_grad for parameter in candidate.current_router.parameters())
    assert all(parameter.requires_grad for parameter in candidate.candidate_router.parameters())

    candidate.train()
    enforce_frozen_module_eval(candidate)
    assert candidate.training is True
    assert candidate.expert.training is False
    assert candidate.expert[1].training is False
    assert candidate.expert[2].training is False
    assert candidate.current_router.training is False
    assert candidate.candidate_router.training is True


def test_frozen_parameters_and_running_statistics_do_not_change_on_step(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path, _SourceModel())
    candidate = _CandidateModel()
    initialize_model_from_checkpoint(candidate, _training_config(checkpoint, digest))
    frozen_parameters = {
        name: parameter.detach().clone()
        for name, parameter in candidate.named_parameters()
        if name.startswith(("expert.", "current_router."))
    }
    running_mean = candidate.expert[1].running_mean.detach().clone()
    running_var = candidate.expert[1].running_var.detach().clone()
    optimizer = torch.optim.Adam(candidate.candidate_router.parameters(), lr=0.1)

    candidate.train()
    enforce_frozen_module_eval(candidate)
    features = candidate.expert(torch.randn(8, 2))
    loss = candidate.candidate_router(features).square().mean()
    loss.backward()
    optimizer.step()

    for name, expected in frozen_parameters.items():
        assert torch.equal(dict(candidate.named_parameters())[name], expected)
    assert torch.equal(candidate.expert[1].running_mean, running_mean)
    assert torch.equal(candidate.expert[1].running_var, running_var)


def test_initialization_is_mutually_exclusive_with_resume(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path, _SourceModel())
    config = _training_config(checkpoint, digest)
    config["resume"] = str(checkpoint)

    with pytest.raises(ValueError, match="mutually exclusive"):
        initialize_model_from_checkpoint(_CandidateModel(), config)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"sha256": "0" * 64}, "SHA256 mismatch"),
        ({"role": "best"}, "role mismatch"),
        ({"checkpoint_schema_version": 2}, "schema mismatch"),
    ],
)
def test_initialization_rejects_source_identity_mismatch(tmp_path: Path, override: dict, message: str) -> None:
    checkpoint, digest = _checkpoint(tmp_path, _SourceModel())

    with pytest.raises(CheckpointLoadError, match=message):
        initialize_model_from_checkpoint(_CandidateModel(), _training_config(checkpoint, digest, **override))


def test_initialization_rejects_non_allowlisted_missing_key(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path, _SourceModel())
    config = _training_config(checkpoint, digest, allowed_missing_prefixes=[])

    with pytest.raises(CheckpointLoadError, match="non-allowlisted missing keys"):
        initialize_model_from_checkpoint(_CandidateModel(), config)


def test_initialization_requires_an_explicit_state_dict(tmp_path: Path) -> None:
    checkpoint, _ = publish_checkpoint(
        {
            "checkpoint_schema_version": 1,
            "checkpoint_role": "last",
            "weights": _SourceModel().state_dict(),
        },
        tmp_path,
        "last.pth",
    )
    digest, _ = checkpoint_file_digest(checkpoint)

    with pytest.raises(CheckpointLoadError, match="state_dict must be a mapping"):
        initialize_model_from_checkpoint(_CandidateModel(), _training_config(checkpoint, digest))


def test_initialization_rejects_unexpected_source_key(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path, _CandidateModel())

    with pytest.raises(CheckpointLoadError, match="unexpected state_dict keys"):
        initialize_model_from_checkpoint(
            _SourceModel(),
            _training_config(checkpoint, digest, allowed_missing_prefixes=[]),
        )


def test_initialization_rejects_required_prefix_or_shape_drift(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path, _SourceModel(expert_width=4))
    config = _training_config(checkpoint, digest)

    with pytest.raises(CheckpointLoadError, match="shape mismatch"):
        initialize_model_from_checkpoint(_CandidateModel(expert_width=3), config)

    missing_required = _training_config(checkpoint, digest, required_prefixes=["not_present"])
    with pytest.raises(CheckpointLoadError, match="required prefix"):
        initialize_model_from_checkpoint(_CandidateModel(expert_width=4), missing_required)
