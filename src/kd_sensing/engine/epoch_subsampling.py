from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Sampler


CONFIG_KEY = "training.epoch_subsampling"
SAMPLER_VERSION = "epoch_subsample_v1"
ORDER_RANDOM = "random"
ORDER_SORTED = "sorted"
ORDER_LOCALITY = "locality"
ORDER_BLOCK_SHUFFLE = "block_shuffle"
ORDER_ALIASES = {
    "random": ORDER_RANDOM,
    "shuffle": ORDER_RANDOM,
    "shuffled": ORDER_RANDOM,
    "sorted": ORDER_SORTED,
    "index": ORDER_SORTED,
    "locality": ORDER_LOCALITY,
    "source": ORDER_LOCALITY,
    "source_block": ORDER_LOCALITY,
    "cache_locality": ORDER_LOCALITY,
    "block_shuffle": ORDER_BLOCK_SHUFFLE,
}


@dataclass(frozen=True)
class EpochSubsamplingPlan:
    enabled: bool
    full_train_samples: int
    effective_train_samples: int
    strategy: str
    seed: int | None
    rotate_each_epoch: bool
    shuffle: bool
    order: str
    block_size: int | None = None
    fraction: float | None = None
    num_samples: int | None = None

    @property
    def full_epoch(self) -> bool:
        return self.effective_train_samples >= self.full_train_samples

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sampler": "EpochSubsampleSampler" if self.enabled else None,
            "sampler_version": SAMPLER_VERSION if self.enabled else None,
            "strategy": self.strategy,
            "full_train_samples": int(self.full_train_samples),
            "effective_train_samples": int(self.effective_train_samples),
            "fraction": self.fraction,
            "num_samples": self.num_samples,
            "seed": self.seed,
            "rotate_each_epoch": bool(self.rotate_each_epoch),
            "shuffle": bool(self.shuffle),
            "order": self.order,
            "locality_strategy": self.order,
            "block_size": self.block_size,
            "full_epoch": bool(self.full_epoch),
            "full_epoch_degenerate": bool(self.enabled and self.full_epoch),
        }


class EpochSubsampleSampler(Sampler[int]):
    """Reproducible without-replacement sampler for train epoch subsampling."""

    def __init__(
        self,
        *,
        dataset_length: int,
        effective_num_samples: int,
        seed: int,
        rotate_each_epoch: bool = True,
        shuffle: bool = True,
        order: str | None = None,
        locality_keys: Sequence[Any] | None = None,
        block_size: int | None = None,
        strategy: str = "num_samples",
        fraction: float | None = None,
        num_samples: int | None = None,
    ) -> None:
        dataset_length = int(dataset_length)
        effective_num_samples = int(effective_num_samples)
        if dataset_length <= 0:
            raise ValueError(f"{CONFIG_KEY} requires a non-empty train dataset.")
        if effective_num_samples <= 0 or effective_num_samples > dataset_length:
            raise ValueError(
                f"{CONFIG_KEY} effective sample count must be in [1, {dataset_length}], "
                f"got {effective_num_samples}."
            )
        self.dataset_length = dataset_length
        self.effective_num_samples = effective_num_samples
        self.seed = int(seed)
        self.rotate_each_epoch = bool(rotate_each_epoch)
        self.shuffle = bool(shuffle)
        self.order = _normalize_order(order, shuffle=self.shuffle)
        self.locality_keys = tuple(locality_keys) if locality_keys is not None else None
        if self.locality_keys is not None and len(self.locality_keys) != dataset_length:
            raise ValueError(
                f"{CONFIG_KEY}.order={self.order} received {len(self.locality_keys)} locality keys "
                f"for dataset length {dataset_length}."
            )
        self.block_size = _parse_block_size(block_size)
        self.strategy = str(strategy)
        self.fraction = fraction
        self.num_samples = num_samples
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        if self.order == ORDER_SORTED and self.effective_num_samples >= self.dataset_length:
            return iter(range(self.dataset_length))

        effective_epoch = self.epoch if self.rotate_each_epoch else 0
        generator = torch.Generator()
        generator.manual_seed(_epoch_seed(self.seed, effective_epoch))
        selected = torch.randperm(self.dataset_length, generator=generator)[: self.effective_num_samples].tolist()
        selected = self._order_selected([int(index) for index in selected], generator=generator)
        return iter(int(index) for index in selected)

    def __len__(self) -> int:
        return self.effective_num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def full_epoch(self) -> bool:
        return self.effective_num_samples >= self.dataset_length

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "sampler": self.__class__.__name__,
            "sampler_version": SAMPLER_VERSION,
            "strategy": self.strategy,
            "full_train_samples": int(self.dataset_length),
            "effective_train_samples": int(self.effective_num_samples),
            "fraction": self.fraction,
            "num_samples": self.num_samples,
            "seed": int(self.seed),
            "rotate_each_epoch": bool(self.rotate_each_epoch),
            "shuffle": bool(self.shuffle),
            "order": self.order,
            "locality_strategy": self.order,
            "block_size": self.block_size,
            "full_epoch": bool(self.full_epoch),
            "full_epoch_degenerate": bool(self.full_epoch),
        }

    def _order_selected(self, selected: list[int], *, generator: torch.Generator) -> list[int]:
        if self.order == ORDER_RANDOM:
            return selected
        if self.order == ORDER_SORTED:
            return sorted(selected)
        if self.order == ORDER_LOCALITY:
            return sorted(selected, key=self._locality_key)
        if self.order == ORDER_BLOCK_SHUFFLE:
            ordered = sorted(selected, key=self._locality_key)
            block_size = self.block_size or max(1, min(len(ordered), 64))
            blocks = [ordered[index : index + block_size] for index in range(0, len(ordered), block_size)]
            if len(blocks) <= 1:
                return ordered
            block_order = torch.randperm(len(blocks), generator=generator).tolist()
            return [item for block_index in block_order for item in blocks[int(block_index)]]
        return selected

    def _locality_key(self, index: int) -> Any:
        if self.locality_keys is None:
            return int(index)
        return self.locality_keys[int(index)]


def validate_epoch_subsampling_config(cfg: dict[str, Any]) -> None:
    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, dict):
        return
    _parse_epoch_subsampling_section(training_cfg.get("epoch_subsampling", {}))


def build_epoch_subsample_sampler(
    dataset: Any,
    subsampling_cfg: dict[str, Any] | None,
    *,
    experiment_seed: int | None,
) -> EpochSubsampleSampler | None:
    plan = resolve_epoch_subsampling_plan(
        subsampling_cfg,
        dataset_length=len(dataset),
        experiment_seed=experiment_seed,
    )
    if not plan.enabled:
        return None
    return EpochSubsampleSampler(
        dataset_length=plan.full_train_samples,
        effective_num_samples=plan.effective_train_samples,
        seed=int(plan.seed if plan.seed is not None else 0),
        rotate_each_epoch=plan.rotate_each_epoch,
        shuffle=plan.shuffle,
        order=plan.order,
        locality_keys=_dataset_locality_keys(dataset, plan.order),
        block_size=plan.block_size,
        strategy=plan.strategy,
        fraction=plan.fraction,
        num_samples=plan.num_samples,
    )


def resolve_epoch_subsampling_plan(
    subsampling_cfg: dict[str, Any] | None,
    *,
    dataset_length: int,
    experiment_seed: int | None,
) -> EpochSubsamplingPlan:
    parsed = _parse_epoch_subsampling_section(subsampling_cfg or {})
    dataset_length = int(dataset_length)
    if not parsed["enabled"]:
        return EpochSubsamplingPlan(
            enabled=False,
            full_train_samples=max(dataset_length, 0),
            effective_train_samples=max(dataset_length, 0),
            strategy="full",
            seed=None,
            rotate_each_epoch=bool(parsed["rotate_each_epoch"]),
            shuffle=bool(parsed["shuffle"]),
            order=parsed["order"],
            block_size=parsed["block_size"],
        )
    if dataset_length <= 0:
        raise ValueError(f"{CONFIG_KEY} requires a non-empty train dataset.")

    seed = parsed["seed"]
    if seed is None:
        seed = int(experiment_seed if experiment_seed is not None else 0)

    fraction = parsed["fraction"]
    num_samples = parsed["num_samples"]
    if fraction is not None:
        effective = max(1, int(dataset_length * float(fraction)))
        effective = min(effective, dataset_length)
        strategy = "fraction"
    else:
        effective = min(int(num_samples), dataset_length)
        strategy = "num_samples"

    return EpochSubsamplingPlan(
        enabled=True,
        full_train_samples=dataset_length,
        effective_train_samples=effective,
        strategy=strategy,
        seed=seed,
        rotate_each_epoch=bool(parsed["rotate_each_epoch"]),
        shuffle=bool(parsed["shuffle"]),
        order=parsed["order"],
        block_size=parsed["block_size"],
        fraction=fraction,
        num_samples=num_samples,
    )


def set_train_sampler_epoch(dataloader: DataLoader, epoch: int) -> None:
    setter = getattr(getattr(dataloader, "sampler", None), "set_epoch", None)
    if callable(setter):
        setter(int(epoch))


def epoch_subsampling_metadata_from_loader(
    dataloader: DataLoader,
    *,
    include_disabled: bool = False,
) -> dict[str, Any]:
    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, EpochSubsampleSampler):
        return sampler.metadata()
    if not include_disabled:
        return {}
    dataset = getattr(dataloader, "dataset", None)
    dataset_length = len(dataset) if dataset is not None else 0
    return EpochSubsamplingPlan(
        enabled=False,
        full_train_samples=dataset_length,
        effective_train_samples=dataset_length,
        strategy="full",
        seed=None,
        rotate_each_epoch=False,
        shuffle=True,
        order=ORDER_RANDOM,
        block_size=None,
    ).metadata()


def epoch_subsampling_epoch_log(dataloader: DataLoader) -> dict[str, Any]:
    metadata = epoch_subsampling_metadata_from_loader(dataloader, include_disabled=True)
    sampler = getattr(dataloader, "sampler", None)
    sampler_epoch = int(sampler.epoch) if isinstance(sampler, EpochSubsampleSampler) else None
    metadata["sampler_epoch"] = sampler_epoch
    return {
        "train_epoch_subsampling": metadata,
        "train_epoch_subsampling_enabled": bool(metadata["enabled"]),
        "train_full_samples": int(metadata["full_train_samples"]),
        "train_effective_samples": int(metadata["effective_train_samples"]),
        "train_sampler_epoch": sampler_epoch,
        "train_epoch_subsampling_full_epoch": bool(metadata["full_epoch"]),
    }


def _parse_epoch_subsampling_section(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{CONFIG_KEY} must be a mapping.")
    enabled = bool(raw.get("enabled", False))
    fraction = _parse_fraction(raw.get("fraction"))
    num_samples = _parse_num_samples(raw.get("num_samples"))
    if fraction is not None and num_samples is not None:
        raise ValueError(f"{CONFIG_KEY} requires exactly one of fraction or num_samples, got both.")
    if enabled and fraction is None and num_samples is None:
        raise ValueError(f"{CONFIG_KEY} requires fraction or num_samples when enabled=true.")
    seed = _parse_seed(raw.get("seed"))
    shuffle = bool(raw.get("shuffle", True))
    order = _normalize_order(raw.get("order", raw.get("locality")), shuffle=shuffle)
    block_size = _parse_block_size(raw.get("block_size", raw.get("locality_block_size")))
    return {
        "enabled": enabled,
        "fraction": fraction,
        "num_samples": num_samples,
        "seed": seed,
        "rotate_each_epoch": bool(raw.get("rotate_each_epoch", True)),
        "shuffle": shuffle,
        "order": order,
        "block_size": block_size,
    }


def _parse_fraction(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{CONFIG_KEY}.fraction must be a number in (0, 1].")
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{CONFIG_KEY}.fraction must be a number in (0, 1].") from exc
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{CONFIG_KEY}.fraction must be in (0, 1], got {value!r}.")
    return fraction


def _parse_num_samples(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{CONFIG_KEY}.num_samples must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{CONFIG_KEY}.num_samples must be a positive integer.") from exc
    if not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{CONFIG_KEY}.num_samples must be a positive integer, got {value!r}.")
    return int(numeric)


def _parse_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{CONFIG_KEY}.seed must be an integer or null.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{CONFIG_KEY}.seed must be an integer or null.") from exc
    if not numeric.is_integer():
        raise ValueError(f"{CONFIG_KEY}.seed must be an integer or null, got {value!r}.")
    return int(numeric)


def _normalize_order(value: Any, *, shuffle: bool) -> str:
    if value is None:
        return ORDER_RANDOM if shuffle else ORDER_SORTED
    normalized = str(value).strip().lower().replace("-", "_")
    order = ORDER_ALIASES.get(normalized)
    if order is None:
        raise ValueError(
            f"{CONFIG_KEY}.order must be one of random, sorted, locality, or block_shuffle; got {value!r}."
        )
    return order


def _parse_block_size(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{CONFIG_KEY}.block_size must be a positive integer or null.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{CONFIG_KEY}.block_size must be a positive integer or null.") from exc
    if not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{CONFIG_KEY}.block_size must be a positive integer, got {value!r}.")
    return int(numeric)


def _dataset_locality_keys(dataset: Any, order: str) -> Sequence[Any] | None:
    if order not in {ORDER_LOCALITY, ORDER_BLOCK_SHUFFLE}:
        return None
    provider = getattr(dataset, "epoch_subsampling_locality_keys", None)
    if callable(provider):
        return provider()
    return None


def _epoch_seed(seed: int, epoch: int) -> int:
    return (int(seed) + max(int(epoch), 0)) % (2**63 - 1)


__all__ = [
    "EpochSubsampleSampler",
    "EpochSubsamplingPlan",
    "build_epoch_subsample_sampler",
    "epoch_subsampling_epoch_log",
    "epoch_subsampling_metadata_from_loader",
    "resolve_epoch_subsampling_plan",
    "set_train_sampler_epoch",
    "validate_epoch_subsampling_config",
]
