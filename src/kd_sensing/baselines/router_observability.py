"""Frozen-U0 Router observability screen.

The frozen U0 encoders are mask independent: ``UMaskBeamJEPA.forward`` encodes all
four modalities unconditionally and only applies the mask during temporal masking
and pooling.  That lets us cache ``latent_sequence`` and the pre-projection
features once and replay the entire U0 head -- reliability, unimodal logits,
router scalars, fusion -- for any mask at negligible cost.

Every arm therefore shares bit-identical representations, so an arm-to-arm delta
can only come from the router input.  This is the attribution isolation that the
BTMA ablation could not achieve, because there encoder training randomness varied
between arms.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.baselines.prototype_decision_adapter import MASKS, numpy_metrics
from kd_sensing.data.corruption_conditions import CONDITION_IDS
from kd_sensing.modalities import MODALITY_ORDER
from kd_sensing.models.router_quality_branch import (
    ARMS,
    PROTOTYPE_STATE_KEYS,
    RouterObservabilityModel,
    uses_quality_branch,
)


SETTINGS: tuple[str, ...] = ("N", "C")
SETTING_DESCRIPTIONS: dict[str, str] = {
    "N": "Full-pool natural weather with the 15 canonical masks",
    "C": "Setting N plus one pre-drawn graded corruption condition per sample",
}
ROUTER_SEEDS: tuple[int, ...] = (1, 2, 3)
MASK_SCHEDULE_SEED = 1  # Fixed across every arm and router seed.
CONDITION_DRAW_SEED = 20260726
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

FULL_MASK = (1, 1, 1, 1)
MASK_PATTERNS: tuple[tuple[str, tuple[int, ...]], ...] = tuple((key, pattern) for key, _, pattern in MASKS)
NON_FULL_KEYS: tuple[str, ...] = tuple(key for key, pattern in MASK_PATTERNS if tuple(pattern) != FULL_MASK)

# Hook the input of each encoder's final output linear layer.  These paths are the
# ones audited in outputs/prototype_collapse_diagnostics/layer_manifest.md.
HOOK_TARGETS: dict[str, str] = {
    "image": "encoders.image.projection.1",
    "radar": "encoders.radar.fc_layer.9",
    "gps": "encoders.gps.net.4",
    "lidar": "encoders.lidar.projection.1",
}
# Output of the per-modality top-level projection, i.e. exactly the tensors that
# UMaskBeamJEPA.forward stacks into `latent_sequence` before temporal pooling.
LATENT_TARGETS: dict[str, str] = {name: f"encoder_projections.{name}" for name in MODALITY_ORDER}

# Pre-registered gate metrics.  Higher is better for the two primaries.
PRIMARY_METRICS: tuple[str, ...] = ("full_top1", "all14_top1")
NON_REGRESSION_METRICS: tuple[str, ...] = ("all14_within3", "all14_mae")


@dataclass(frozen=True)
class ArmRun:
    setting: str
    arm: str
    seed: int

    @property
    def key(self) -> str:
        return f"setting_{self.setting}/{self.arm}/seed_{self.seed}"

    @property
    def directory(self) -> str:
        return f"setting_{self.setting}/{self.arm}/seed_{self.seed}"


def all_runs(settings: Sequence[str] = SETTINGS, arms: Sequence[str] = ARMS, seeds: Sequence[int] = ROUTER_SEEDS) -> list[ArmRun]:
    return [ArmRun(setting, arm, int(seed)) for setting in settings for arm in arms for seed in seeds]


# --------------------------------------------------------------------------
# frozen head replay
# --------------------------------------------------------------------------


def resolve_hook_modules(model: nn.Module) -> dict[str, nn.Linear]:
    """Resolve and validate the audited pre-projection hook points."""
    modules = dict(model.named_modules())
    resolved: dict[str, nn.Linear] = {}
    for name, path in HOOK_TARGETS.items():
        module = modules.get(path)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"Pre-projection hook {path!r} for {name} is not an nn.Linear; refusing to guess.")
        resolved[name] = module
    return resolved


def preprojection_dims(model: nn.Module) -> dict[str, int]:
    return {name: int(module.in_features) for name, module in resolve_hook_modules(model).items()}


class EncoderCapture:
    """Capture ``latent_sequence`` and pre-projection features during one forward.

    Both quantities are mask independent, so a single clean pass per sample is
    enough to replay every mask afterwards.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.preprojection: dict[str, torch.Tensor] = {}
        self.latent: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        modules = dict(model.named_modules())
        for name, module in resolve_hook_modules(model).items():
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
        for name, path in LATENT_TARGETS.items():
            module = modules.get(path)
            if module is None:
                raise ValueError(f"Latent hook {path!r} is absent from the frozen U0.")
            self._handles.append(module.register_forward_hook(self._post_hook(name)))

    def _pre_hook(self, name: str):
        def hook(module: nn.Module, args: tuple[Any, ...]) -> None:
            self.preprojection[name] = args[0].detach()

        return hook

    def _post_hook(self, name: str):
        def hook(module: nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> None:
            self.latent[name] = output.detach()

        return hook

    def collect(self, batch: int, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        missing = [name for name in MODALITY_ORDER if name not in self.latent or name not in self.preprojection]
        if missing:
            raise ValueError(f"Encoder capture did not fire for {missing}.")
        latent = torch.stack([self.latent[name].reshape(batch, steps, -1) for name in MODALITY_ORDER], dim=2)
        preprojection = {
            name: self.preprojection[name].reshape(batch, steps, -1) for name in MODALITY_ORDER
        }
        self.preprojection.clear()
        self.latent.clear()
        return latent, preprojection

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "EncoderCapture":
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


def expand_mask(pattern: Sequence[int], batch: int, device: torch.device) -> torch.Tensor:
    value = torch.tensor(tuple(int(bit) for bit in pattern), dtype=torch.bool, device=device)
    if value.numel() != len(MODALITY_ORDER):
        raise ValueError(f"Mask pattern must cover {len(MODALITY_ORDER)} modalities.")
    return value.unsqueeze(0).expand(batch, -1)


class FrozenU0Head(nn.Module):
    """Replay U0's post-encoder path from cached features for an arbitrary mask.

    The frozen U0 module itself supplies ``_modality_reliability``, ``_head_logits``
    and ``_router_features``; nothing is reimplemented, so only the cached encoder
    outputs can diverge from a live forward pass.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):  # noqa: D102 - keep the backbone frozen
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        latent_sequence: torch.Tensor,
        preprojection_sequence: Mapping[str, torch.Tensor],
        available: torch.Tensor,
    ) -> dict[str, Any]:
        if latent_sequence.ndim != 4:
            raise ValueError(f"latent_sequence must be [B, T, M, D], got {tuple(latent_sequence.shape)}.")
        batch, steps, modalities, dimension = latent_sequence.shape
        mask = available.to(dtype=torch.bool, device=latent_sequence.device)
        cell_mask = torch.ones(batch, steps, modalities, dtype=torch.bool, device=latent_sequence.device)
        cell_mask = cell_mask & mask.unsqueeze(1)
        if not bool(cell_mask.any(dim=(1, 2)).all()):
            raise ValueError("Every sample must keep at least one available temporal cell.")
        weights = cell_mask.to(dtype=latent_sequence.dtype).unsqueeze(-1)
        latent = (latent_sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        reliability = self.model._modality_reliability(latent, mask)
        unimodal_logits = self.model._head_logits(latent.reshape(-1, dimension)).reshape(batch, modalities, -1)
        scalars = self.model._router_features(unimodal_logits, reliability, mask)
        state = self.model.prototype_bank.describe(latent.reshape(-1, dimension))
        prototype_state = {
            key: state[key].reshape(batch, modalities) for key in PROTOTYPE_STATE_KEYS
        }
        pooled: dict[str, torch.Tensor] = {}
        for index, name in enumerate(MODALITY_ORDER):
            sequence = preprojection_sequence[name]
            column = cell_mask[:, :, index].to(dtype=sequence.dtype).unsqueeze(-1)
            pooled[name] = (sequence * column).sum(dim=1) / column.sum(dim=1).clamp_min(1.0)
        return {
            "latent": latent,
            "available": mask,
            "unimodal_logits": unimodal_logits,
            "scalars": scalars,
            "prototype_state": prototype_state,
            "preprojection": pooled,
        }

    @torch.no_grad()
    def reference_logits(self, replayed: Mapping[str, Any]) -> torch.Tensor:
        """Fuse using U0's own trained router; the untrained reference row."""
        _, weights = self.model.route_from_features(replayed["scalars"], replayed["available"])
        return (weights.unsqueeze(-1) * replayed["unimodal_logits"]).sum(dim=1)


# --------------------------------------------------------------------------
# deterministic schedules
# --------------------------------------------------------------------------


TEMPORAL_MASK_KEYS: tuple[str, ...] = ("temporal_mask", "modality_temporal_mask", "available_modalities")


def assert_dense_temporal_inputs(inputs: Mapping[str, Any]) -> None:
    """Fail closed if the batch carries a temporal mask.

    Cache replay reconstructs ``cell_mask`` as ``ones & available``, exactly like
    the frozen-U0 Adapter workflow.  A dataset-supplied temporal mask would make
    that reconstruction wrong, so refuse instead of silently diverging.
    """
    for key in TEMPORAL_MASK_KEYS:
        value = inputs.get(key)
        if value is None:
            continue
        if not bool(torch.as_tensor(value).all()):
            raise ValueError(
                f"Router observability requires dense temporal inputs, but {key!r} drops cells. "
                "Cached replay cannot reproduce a dataset-supplied temporal mask."
            )


def _stable_hash(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


_MASK_SCHEDULE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def mask_schedule(sample_ids: Sequence[str], epoch: int) -> np.ndarray:
    """Uniform draw over all 15 canonical masks, identical for every arm and seed."""
    fingerprint = hashlib.sha256("\n".join(str(value) for value in sample_ids).encode("utf-8")).hexdigest()
    key = (fingerprint, int(epoch))
    cached = _MASK_SCHEDULE_CACHE.get(key)
    if cached is None:
        cached = np.asarray(
            [_stable_hash(MASK_SCHEDULE_SEED, epoch, sample_id) % len(MASK_PATTERNS) for sample_id in sample_ids],
            dtype=np.int64,
        )
        _MASK_SCHEDULE_CACHE[key] = cached
    return cached


def draw_conditions(sample_ids: Sequence[str]) -> list[str]:
    """One fixed corruption condition per sample, drawn before any metric is read."""
    return [CONDITION_IDS[_stable_hash(CONDITION_DRAW_SEED, sample_id) % len(CONDITION_IDS)] for sample_id in sample_ids]


def cross_sample_permutation(count: int, *, seed: int, tag: str) -> np.ndarray:
    """Derangement-style permutation used by the q3 capacity control.

    Fixed points are removed by rotating them among themselves, which keeps the
    result a bijection.  Repairing each fixed point against its neighbour instead
    collides whenever two fixed points are cyclically adjacent, and a q3 batch
    that silently duplicates one sample while dropping another is not the capacity
    control it claims to be.
    """
    generator = np.random.default_rng(_stable_hash(seed, tag) % (2**32))
    permutation = generator.permutation(count)
    if count < 2:
        return permutation
    fixed = np.flatnonzero(permutation == np.arange(count))
    if fixed.size == 1:
        # A lone fixed point has nothing to rotate against; swap it with a neighbour.
        partner = (int(fixed[0]) + 1) % count
        permutation[[int(fixed[0]), partner]] = permutation[[partner, int(fixed[0])]]
    elif fixed.size > 1:
        permutation[fixed] = permutation[np.roll(fixed, 1)]
    return permutation


# --------------------------------------------------------------------------
# cache storage precision
# --------------------------------------------------------------------------

# The cache is built and replayed in float32, and the whole screen is float32.
#
# The retained-U0 evaluation path normally runs under a bfloat16 autocast, but
# bfloat16 is not a usable reference here: running the same frozen model twice,
# once in float32 and once under autocast, moves the fused logits by up to 1.5e-1,
# which is larger than the cache round-trip itself.  Measured against a float32
# live forward, the float32 replay is bit-exact (0.0 across all 15 masks), so
# float32 makes the load-bearing equivalence property exactly verifiable instead
# of merely approximately.  The cost is 947 MB per setting rather than 474 MB.
#
# Consequence for reading the numbers: this screen's absolute Top-1 is not
# directly comparable to the published bfloat16 A0 row.  The frozen-U0 reference
# row is therefore recomputed through this same float32 path, and every arm-to-arm
# comparison happens inside it.
CACHE_DTYPE = torch.float32


def quantize_for_cache(value: torch.Tensor) -> torch.Tensor:
    """Round-trip a tensor through the cache's storage precision."""
    return value.to(CACHE_DTYPE)


def pack_cache_array(value: torch.Tensor) -> np.ndarray:
    """Serialise a tensor at exactly the precision the arms will consume."""
    return value.to(CACHE_DTYPE).contiguous().cpu().numpy()


def unpack_cache_array(value: np.ndarray, device: torch.device) -> torch.Tensor:
    """Invert :func:`pack_cache_array`."""
    return torch.as_tensor(np.ascontiguousarray(value), device=device).to(CACHE_DTYPE)


# --------------------------------------------------------------------------
# cache access
# --------------------------------------------------------------------------


class RepresentationCache:
    """GPU-resident frozen representations for one split of one setting."""

    def __init__(self, payload: Mapping[str, np.ndarray], device: torch.device) -> None:
        self.sample_id = np.asarray(payload["sample_id"], dtype=str)
        self.domain = np.asarray(payload["domain"], dtype=str)
        self.weather = np.asarray(payload["weather"], dtype=str)
        self.condition = np.asarray(payload["condition"], dtype=str)
        self.label = torch.as_tensor(np.asarray(payload["label"], dtype=np.int64), device=device)
        self.latent_sequence = unpack_cache_array(np.asarray(payload["latent_sequence"]), device)
        self.preprojection = {
            name: unpack_cache_array(np.asarray(payload[f"preprojection_{name}"]), device)
            for name in MODALITY_ORDER
        }
        forced = payload.get("forced_missing")
        self.forced_missing = (
            torch.as_tensor(np.asarray(forced, dtype=bool), device=device)
            if forced is not None
            else torch.zeros((len(self.sample_id), len(MODALITY_ORDER)), dtype=torch.bool, device=device)
        )
        self.device = device

    def __len__(self) -> int:
        return len(self.sample_id)

    def slice(self, index: torch.Tensor) -> dict[str, Any]:
        return {
            "latent_sequence": self.latent_sequence[index],
            "preprojection": {name: value[index] for name, value in self.preprojection.items()},
            "label": self.label[index],
            "forced_missing": self.forced_missing[index],
        }

    def available(self, index: torch.Tensor, pattern: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Combine the mask pattern with any modality forced missing by corruption."""
        if isinstance(pattern, torch.Tensor):
            scheduled = pattern.to(dtype=torch.bool, device=self.device)
        else:
            scheduled = expand_mask(pattern, int(index.numel()), self.device)
        mask = scheduled & ~self.forced_missing[index]
        scheduled = scheduled.expand_as(mask)
        # A corruption-forced removal must never empty a sample, because pooling
        # over an all-zero mask is undefined.  The restored modality is drawn from
        # the scheduled pattern and never from outside it: reviving a hard-masked
        # modality would quietly give setting C a different mask distribution than
        # setting N, which is exactly the comparison this screen rests on.
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            rows = torch.nonzero(empty, as_tuple=False).flatten()
            mask[rows, scheduled[rows].float().argmax(dim=1)] = True
        return mask


def load_cache(path: Path, device: torch.device) -> RepresentationCache:
    if not path.is_file():
        raise FileNotFoundError(f"Representation cache is absent: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return RepresentationCache({name: payload[name] for name in payload.files}, device)


# --------------------------------------------------------------------------
# training and evaluation
# --------------------------------------------------------------------------


def build_model(arm: str, head: FrozenU0Head, cache: RepresentationCache, *, seed: int) -> RouterObservabilityModel:
    probe_index = torch.arange(min(2, len(cache)), device=cache.device)
    probe = head(
        cache.latent_sequence[probe_index],
        {name: value[probe_index] for name, value in cache.preprojection.items()},
        cache.available(probe_index, FULL_MASK),
    )
    # Seed immediately before construction so only the router init consumes the RNG.
    torch.manual_seed(seed)
    return RouterObservabilityModel(
        arm,
        scalar_feature_count=int(probe["scalars"].shape[-1]),
        preprojection_dims={name: int(value.shape[-1]) for name, value in cache.preprojection.items()},
    ).to(cache.device)


def train_arm(
    model: RouterObservabilityModel,
    head: FrozenU0Head,
    cache: RepresentationCache,
    *,
    seed: int,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, math.ceil(len(cache) / batch_size))
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, steps_per_epoch)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    patterns = torch.tensor([[int(bit) for bit in pattern] for _, pattern in MASK_PATTERNS], device=cache.device)
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        order = torch.as_tensor(
            np.random.default_rng(_stable_hash(seed, "order", epoch) % (2**32)).permutation(len(cache)),
            device=cache.device,
        )
        schedule = torch.as_tensor(mask_schedule(cache.sample_id.tolist(), epoch), device=cache.device)
        loss_sum = 0.0
        count = 0
        for start in range(0, len(cache), batch_size):
            index = order[start : start + batch_size]
            available = cache.available(index, patterns[schedule[index]])
            replayed = head(
                cache.latent_sequence[index],
                {name: value[index] for name, value in cache.preprojection.items()},
                available,
            )
            permutation = None
            if model.arm == "q3":
                permutation = torch.as_tensor(
                    cross_sample_permutation(int(index.numel()), seed=seed, tag=f"train:{epoch}:{start}"),
                    device=cache.device,
                )
            optimizer.zero_grad(set_to_none=True)
            output = model(
                replayed["scalars"],
                replayed["available"],
                replayed["unimodal_logits"],
                prototype_state=replayed["prototype_state"],
                preprojection=replayed["preprojection"],
                permutation=permutation,
            )
            loss = F.cross_entropy(output["logits"].float(), cache.label[index])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite router loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            size = int(index.numel())
            loss_sum += float(loss.detach()) * size
            count += size
        history.append({"epoch": epoch + 1, "loss": loss_sum / max(count, 1), "lr": optimizer.param_groups[0]["lr"]})
    return history


@torch.no_grad()
def fit_quality_mean(model: RouterObservabilityModel, head: FrozenU0Head, cache: RepresentationCache) -> None:
    """Train-only mean quality embedding for the inference-time ablation."""
    if not uses_quality_branch(model.arm):
        return
    model.eval()
    totals = torch.zeros_like(model.quality_mean)
    count = 0
    for start in range(0, len(cache), BATCH_SIZE):
        index = torch.arange(start, min(start + BATCH_SIZE, len(cache)), device=cache.device)
        replayed = head(
            cache.latent_sequence[index],
            {name: value[index] for name, value in cache.preprojection.items()},
            cache.available(index, FULL_MASK),
        )
        assert model.quality_branch is not None
        totals += model.quality_branch(replayed["preprojection"]).sum(dim=0).to(totals.dtype)
        count += int(index.numel())
    model.set_quality_mean(totals / max(count, 1))


@torch.no_grad()
def evaluate_arm(
    model: RouterObservabilityModel | None,
    head: FrozenU0Head,
    cache: RepresentationCache,
    *,
    seed: int,
    ablate_quality: bool = False,
) -> dict[str, Any]:
    """Enumerate all 15 canonical masks; ``model=None`` evaluates U0's own router."""
    if model is not None:
        model.eval()
    per_mask: dict[str, dict[str, float]] = {}
    per_condition: dict[str, dict[str, list[float]]] = {}
    for key, pattern in MASK_PATTERNS:
        logits_chunks: list[np.ndarray] = []
        for start in range(0, len(cache), BATCH_SIZE):
            index = torch.arange(start, min(start + BATCH_SIZE, len(cache)), device=cache.device)
            available = cache.available(index, pattern)
            replayed = head(
                cache.latent_sequence[index],
                {name: value[index] for name, value in cache.preprojection.items()},
                available,
            )
            if model is None:
                logits = head.reference_logits(replayed)
            else:
                permutation = None
                if model.arm == "q3":
                    permutation = torch.as_tensor(
                        cross_sample_permutation(int(index.numel()), seed=seed, tag=f"eval:{key}:{start}"),
                        device=cache.device,
                    )
                logits = model(
                    replayed["scalars"],
                    replayed["available"],
                    replayed["unimodal_logits"],
                    prototype_state=replayed["prototype_state"],
                    preprojection=replayed["preprojection"],
                    permutation=permutation,
                    ablate_quality=ablate_quality,
                )["logits"]
            logits_chunks.append(logits.float().cpu().numpy())
        logits_all = np.concatenate(logits_chunks)
        target = cache.label.cpu().numpy()
        per_mask[key] = numpy_metrics(logits_all, target)
        if key == "full":
            correct = logits_all.argmax(axis=1) == target
            for condition, hit in zip(cache.condition.tolist(), correct.tolist()):
                per_condition.setdefault(condition, {"top1": []})["top1"].append(float(hit))

    summary: dict[str, Any] = {"per_mask": per_mask}
    for name in ("top1", "top3", "within3", "mae", "adba", "loss"):
        summary[f"full_{name}"] = float(per_mask["full"][name])
        values = [per_mask[key][name] for key in NON_FULL_KEYS]
        summary[f"all14_{name}"] = float(np.mean(values))
        summary[f"all14_worst_{name}"] = float(np.min(values) if name in {"top1", "top3", "within3", "adba"} else np.max(values))
    summary["per_condition_full_top1"] = {
        condition: float(np.mean(values["top1"])) for condition, values in sorted(per_condition.items())
    }
    return summary


# --------------------------------------------------------------------------
# pre-registered gates
# --------------------------------------------------------------------------


def _interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.min()), float(array.max())


def evaluate_gates(summaries: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Q2 must beat both Q1 and Q3 on both primaries with non-overlapping seed ranges."""
    rows: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        treatment = [float(item[metric]) for item in summaries["q2"]]
        low_treatment, high_treatment = _interval(treatment)
        for control in ("q1", "q3"):
            reference = [float(item[metric]) for item in summaries[control]]
            low_reference, high_reference = _interval(reference)
            rows.append(
                {
                    "gate": f"q2_beats_{control}_{metric}",
                    "metric": metric,
                    "treatment_mean": float(np.mean(treatment)),
                    "treatment_min": low_treatment,
                    "treatment_max": high_treatment,
                    "control": control,
                    "control_mean": float(np.mean(reference)),
                    "control_min": low_reference,
                    "control_max": high_reference,
                    "separated": bool(low_treatment > high_reference),
                    "passed": bool(low_treatment > high_reference),
                }
            )
    for metric in NON_REGRESSION_METRICS:
        treatment = float(np.mean([float(item[metric]) for item in summaries["q2"]]))
        reference = float(np.mean([float(item[metric]) for item in summaries["q1"]]))
        higher_is_better = not metric.endswith("mae")
        passed = treatment >= reference if higher_is_better else treatment <= reference
        rows.append(
            {
                "gate": f"q2_no_regression_{metric}",
                "metric": metric,
                "treatment_mean": treatment,
                "treatment_min": treatment,
                "treatment_max": treatment,
                "control": "q1",
                "control_mean": reference,
                "control_min": reference,
                "control_max": reference,
                "separated": False,
                "passed": bool(passed),
            }
        )
    return rows


def direction_survives(gates: Sequence[Mapping[str, Any]]) -> bool:
    return bool(gates) and all(bool(row["passed"]) for row in gates)


__all__ = [
    "ArmRun",
    "BATCH_SIZE",
    "CONDITION_DRAW_SEED",
    "EPOCHS",
    "FULL_MASK",
    "EncoderCapture",
    "FrozenU0Head",
    "HOOK_TARGETS",
    "LATENT_TARGETS",
    "MASK_PATTERNS",
    "MASK_SCHEDULE_SEED",
    "NON_FULL_KEYS",
    "NON_REGRESSION_METRICS",
    "PRIMARY_METRICS",
    "ROUTER_SEEDS",
    "RepresentationCache",
    "SETTINGS",
    "SETTING_DESCRIPTIONS",
    "TEMPORAL_MASK_KEYS",
    "all_runs",
    "assert_dense_temporal_inputs",
    "build_model",
    "cross_sample_permutation",
    "direction_survives",
    "draw_conditions",
    "evaluate_arm",
    "evaluate_gates",
    "expand_mask",
    "fit_quality_mean",
    "load_cache",
    "mask_schedule",
    "preprojection_dims",
    "resolve_hook_modules",
    "train_arm",
]
