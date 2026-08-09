from torch.utils.data import Dataset

from kd_sensing.engine.data_factory import (
    build_dataloader,
    dataloader_generator_metadata,
    resolve_dataloader_generator_seed,
)


class _IndexedDataset(Dataset):
    def __init__(self, identity: str, length: int = 32) -> None:
        self.schema_identity = identity
        self.split = "train"
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> int:
        return index


def _loader_config(**extra) -> dict:
    return {
        "train_batch_size": 8,
        "validation_batch_size": 8,
        "test_batch_size": 8,
        "num_workers": 0,
        **extra,
    }


def test_explicit_generator_seed_preserves_order_across_dataset_fingerprints() -> None:
    config = _loader_config(generator_seeds={"train": 1234})
    first = build_dataloader(_IndexedDataset("first"), config, split="train", experiment_seed=1)
    second = build_dataloader(_IndexedDataset("second"), config, split="train", experiment_seed=1)

    first_order = [int(index) for batch in first for index in batch]
    second_order = [int(index) for batch in second for index in batch]
    assert first_order == second_order
    assert first.generator_metadata["algorithm"] == "explicit-v1"
    assert first.generator_metadata["derived_seed"] == 1234
    assert first.generator_metadata["explicit_seed"] == 1234
    assert first.generator_metadata["dataset_fingerprint"] != second.generator_metadata["dataset_fingerprint"]


def test_default_generator_seed_remains_fingerprint_derived() -> None:
    first = dataloader_generator_metadata(_IndexedDataset("first"), split="train", base_seed=1)
    repeated = dataloader_generator_metadata(_IndexedDataset("first"), split="train", base_seed=1)
    second = dataloader_generator_metadata(_IndexedDataset("second"), split="train", base_seed=1)

    assert first == repeated
    assert first["algorithm"] == "sha256-v1"
    assert "explicit_seed" not in first
    assert first["derived_seed"] != second["derived_seed"]


def test_generator_seed_config_rejects_invalid_values() -> None:
    for config in (
        {"generator_seeds": []},
        {"generator_seeds": {"development": 1}},
        {"generator_seeds": {"train": True}},
        {"generator_seeds": {"train": -1}},
        {"generator_seeds": {"train": 1 << 63}},
    ):
        try:
            resolve_dataloader_generator_seed(config, split="train")
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid generator seed config to fail: {config}")
