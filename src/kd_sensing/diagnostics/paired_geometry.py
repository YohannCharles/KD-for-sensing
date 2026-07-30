"""Paired Full-to-Missing geometry utilities for frozen sensing features."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from kd_sensing.diagnostics.prototype_deformation import normalize


GROUP_NAMES = ("G0", "G1", "G2", "G3")


def predictive_statistics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    scores = torch.as_tensor(logits, dtype=torch.float32)
    target = torch.as_tensor(labels, dtype=torch.long, device=scores.device).reshape(-1)
    if scores.ndim < 2 or scores.shape[0] != target.numel() or scores.shape[-1] != 64:
        raise ValueError("logits must be [N,...,64] and labels must be [N].")
    expanded = target.reshape(target.numel(), *([1] * (scores.ndim - 2)), 1).expand(*scores.shape[:-1], 1)
    target_score = scores.gather(-1, expanded).squeeze(-1)
    other = scores.clone()
    other.scatter_(-1, expanded, -torch.inf)
    probability = torch.softmax(scores, dim=-1)
    prediction = scores.argmax(dim=-1)
    return {
        "prediction": prediction,
        "correct": prediction.eq(target.reshape(target.numel(), *([1] * (scores.ndim - 2)))),
        "target_rank": 1 + scores.gt(target_score.unsqueeze(-1)).sum(dim=-1),
        "target_margin": target_score - other.max(dim=-1).values,
        "entropy": -(probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum(dim=-1),
        "nll": -torch.log_softmax(scores, dim=-1).gather(-1, expanded).squeeze(-1),
        "probability": probability,
    }


def classification_groups(
    full_prediction: torch.Tensor,
    missing_prediction: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    missing = torch.as_tensor(missing_prediction, dtype=torch.long)
    full = torch.as_tensor(full_prediction, dtype=torch.long, device=missing.device).reshape(-1)
    target = torch.as_tensor(labels, dtype=torch.long, device=missing.device).reshape(-1)
    if missing.ndim == 1:
        missing = missing[:, None]
    if full.shape != target.shape or missing.shape[0] != target.numel():
        raise ValueError("Full, Missing, and target identities do not align.")
    full_correct = full.eq(target)[:, None]
    missing_correct = missing.eq(target[:, None])
    groups = torch.full(missing.shape, 2, dtype=torch.int8, device=missing.device)
    groups[full_correct & missing_correct] = 0
    groups[full_correct & ~missing_correct] = 1
    groups[~full_correct & missing_correct] = 3
    return groups


def signed_cycle_offset(
    target: torch.Tensor,
    prediction: torch.Tensor,
    labels_by_position: tuple[int, ...],
) -> torch.Tensor:
    labels = tuple(int(value) for value in labels_by_position)
    if len(labels) != len(set(labels)) or set(labels) != set(range(len(labels))):
        raise ValueError("cycle labels must be a permutation of contiguous class IDs.")
    target_tensor = torch.as_tensor(target, dtype=torch.long)
    prediction_tensor = torch.as_tensor(prediction, dtype=torch.long, device=target_tensor.device)
    if target_tensor.shape != prediction_tensor.shape:
        raise ValueError("cycle target and prediction must align.")
    position = torch.empty(len(labels), dtype=torch.long, device=target_tensor.device)
    position[torch.tensor(labels, dtype=torch.long, device=target_tensor.device)] = torch.arange(
        len(labels), device=target_tensor.device
    )
    raw = position[prediction_tensor] - position[target_tensor]
    half = len(labels) // 2
    return torch.remainder(raw + half, len(labels)) - half


def validate_pair_alignment(
    sample_ids: list[str],
    values: torch.Tensor,
    *,
    expected_masks: int = 15,
) -> None:
    features = torch.as_tensor(values)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be unique within a split.")
    if features.ndim < 2 or features.shape[0] != len(sample_ids) or features.shape[1] != int(expected_masks):
        raise ValueError("paired values must align as [sample, mask, ...].")


def validate_safety_contract(contract: Mapping[str, Any]) -> None:
    expected = {
        "csi_used": False,
        "channel_input_used": False,
        "f1_used": False,
        "outer_test_accessed": False,
        "future_beam_power_role": "label_side_evaluation_metric_only",
    }
    observed = {key: contract.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"invalid sensing-only safety contract: {observed}")


def nonempty_subset_utilities(
    utility: torch.Tensor,
    mask_bits: torch.Tensor,
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    """Enumerate valid first- and second-order utility comparisons without an empty subset."""
    values = torch.as_tensor(utility)
    bits = torch.as_tensor(mask_bits, dtype=torch.bool, device=values.device)
    if values.ndim < 2 or bits.shape != (values.shape[1], 4):
        raise ValueError("subset utility requires values [N,M,...] and mask bits [M,4].")
    lookup = {tuple(int(item) for item in row): index for index, row in enumerate(bits.tolist())}
    if len(lookup) != bits.shape[0] or (0, 0, 0, 0) in lookup:
        raise ValueError("mask bits must be unique non-empty four-modality subsets.")
    marginal: list[dict[str, int]] = []
    interaction: list[dict[str, int]] = []
    for base_index, row in enumerate(bits.tolist()):
        base = tuple(int(item) for item in row)
        for modality in range(4):
            if base[modality]:
                continue
            union = list(base)
            union[modality] = 1
            union_index = lookup.get(tuple(union))
            if union_index is not None:
                marginal.append({"base": base_index, "union": union_index, "modality": modality})
        for left, right in combinations(range(4), 2):
            if base[left] or base[right]:
                continue
            left_bits = list(base)
            left_bits[left] = 1
            right_bits = list(base)
            right_bits[right] = 1
            both_bits = list(base)
            both_bits[left] = both_bits[right] = 1
            if tuple(left_bits) in lookup and tuple(right_bits) in lookup and tuple(both_bits) in lookup:
                interaction.append(
                    {
                        "base": base_index,
                        "left": lookup[tuple(left_bits)],
                        "right": lookup[tuple(right_bits)],
                        "both": lookup[tuple(both_bits)],
                        "modality_left": left,
                        "modality_right": right,
                    }
                )
    return marginal, interaction


def highest_confuser(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    scores = torch.as_tensor(logits, dtype=torch.float32).clone()
    target = torch.as_tensor(labels, dtype=torch.long, device=scores.device).reshape(-1)
    if scores.ndim != 2 or scores.shape != (target.numel(), 64):
        raise ValueError("confuser logits must be [N,64].")
    scores.scatter_(1, target[:, None], -torch.inf)
    return scores.argmax(dim=1)


def final_geometry(
    full_raw: torch.Tensor,
    missing_raw: torch.Tensor,
    labels: torch.Tensor,
    missing_logits: torch.Tensor,
    prototypes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    raw_full = torch.as_tensor(full_raw, dtype=torch.float32)
    raw_missing = torch.as_tensor(missing_raw, dtype=torch.float32, device=raw_full.device)
    target = torch.as_tensor(labels, dtype=torch.long, device=raw_full.device).reshape(-1)
    scores = torch.as_tensor(missing_logits, dtype=torch.float32, device=raw_full.device)
    bank = normalize(torch.as_tensor(prototypes, dtype=torch.float32, device=raw_full.device))
    if raw_full.shape != raw_missing.shape or raw_full.ndim != 2 or raw_full.shape[0] != target.numel():
        raise ValueError("paired final features must align as [N,D].")
    full = normalize(raw_full)
    missing = normalize(raw_missing)
    cosine = (full * missing).sum(dim=1).clamp(-1.0, 1.0)
    confuser = highest_confuser(scores, target)
    singular = torch.linalg.svdvals(bank)
    rank = int((singular > singular.max() * 1e-7).sum())
    basis = torch.linalg.svd(bank, full_matrices=False).Vh[:rank].t()
    full_projection = full @ basis
    missing_projection = missing @ basis
    return {
        "cosine": cosine,
        "angular_distance": torch.acos(cosine),
        "euclidean_distance": (missing - full).norm(dim=1),
        "raw_norm_change": raw_missing.norm(dim=1) - raw_full.norm(dim=1),
        "delta_target": (missing * bank[target]).sum(dim=1) - (full * bank[target]).sum(dim=1),
        "delta_confuser": (missing * bank[confuser]).sum(dim=1) - (full * bank[confuser]).sum(dim=1),
        "full_projection_energy": full_projection.square().sum(dim=1),
        "missing_projection_energy": missing_projection.square().sum(dim=1),
        "full_orthogonal_residual": (full - full_projection @ basis.t()).norm(dim=1),
        "missing_orthogonal_residual": (missing - missing_projection @ basis.t()).norm(dim=1),
        "confuser": confuser,
    }


def decision_decomposition(
    full_raw: torch.Tensor,
    missing_raw: torch.Tensor,
    full_logits: torch.Tensor,
    missing_logits: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    full = normalize(torch.as_tensor(full_raw, dtype=torch.float32))
    missing = normalize(torch.as_tensor(missing_raw, dtype=torch.float32, device=full.device))
    scores_full = torch.as_tensor(full_logits, dtype=torch.float32, device=full.device)
    scores_missing = torch.as_tensor(missing_logits, dtype=torch.float32, device=full.device)
    target = torch.as_tensor(labels, dtype=torch.long, device=full.device).reshape(-1)
    bank = normalize(torch.as_tensor(prototypes, dtype=torch.float32, device=full.device))
    if full.shape != missing.shape or scores_full.shape != scores_missing.shape or scores_full.shape != (target.numel(), 64):
        raise ValueError("decision decomposition inputs do not align.")
    confuser = scores_missing.argmax(dim=1)
    correct = confuser.eq(target)
    if bool(correct.any()):
        confuser[correct] = highest_confuser(scores_missing[correct], target[correct])
    difference = bank[target] - bank[confuser]
    direction = normalize(difference)
    delta = missing - full
    signed = (delta * direction).sum(dim=1)
    parallel = signed[:, None] * direction
    orthogonal = delta - parallel
    energy = delta.square().sum(dim=1)
    production_left = (
        scores_missing.gather(1, target[:, None]).squeeze(1)
        - scores_missing.gather(1, confuser[:, None]).squeeze(1)
        - scores_full.gather(1, target[:, None]).squeeze(1)
        + scores_full.gather(1, confuser[:, None]).squeeze(1)
    )
    full_fp64 = full.double()
    missing_fp64 = missing.double()
    bank_fp64 = bank.double()
    exact_full = full_fp64 @ bank_fp64.t() / float(temperature)
    exact_missing = missing_fp64 @ bank_fp64.t() / float(temperature)
    left = (
        exact_missing.gather(1, target[:, None]).squeeze(1)
        - exact_missing.gather(1, confuser[:, None]).squeeze(1)
        - exact_full.gather(1, target[:, None]).squeeze(1)
        + exact_full.gather(1, confuser[:, None]).squeeze(1)
    )
    right = ((missing_fp64 - full_fp64) * (bank_fp64[target] - bank_fp64[confuser])).sum(dim=1) / float(
        temperature
    )
    return {
        "confuser": confuser,
        "signed_parallel": signed,
        "absolute_parallel": signed.abs(),
        "orthogonal_norm": orthogonal.norm(dim=1),
        "parallel_energy_ratio": parallel.square().sum(dim=1) / energy.clamp_min(1e-12),
        "delta_direction_cosine": signed / delta.norm(dim=1).clamp_min(1e-12),
        "identity_left": left,
        "identity_right": right,
        "identity_absolute_error": (left - right).abs(),
        "production_identity_left": production_left.double(),
        "production_identity_absolute_error": (production_left.double() - right).abs(),
        "parallel_vector": parallel,
        "orthogonal_vector": orthogonal,
    }


def representation_spectrum(features: torch.Tensor, *, dead_variance_ratio: float = 1e-6) -> dict[str, Any]:
    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("representation spectrum requires [N,D] with N >= 2.")
    centered = values - values.mean(dim=0, keepdim=True)
    if centered.shape[0] < centered.shape[1]:
        gram = centered.matmul(centered.t()) / max(1, centered.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    else:
        covariance = centered.t().matmul(centered) / max(1, centered.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    probability = eigenvalues / total
    entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum()
    variance = centered.square().mean(dim=0)
    mean_variance = variance.mean().clamp_min(torch.finfo(variance.dtype).tiny)
    normalized = normalize(values)
    count = normalized.shape[0]
    pairwise_cosine = (
        normalized.sum(dim=0).square().sum() - count
    ) / max(1, count * (count - 1))
    return {
        "sample_count": count,
        "dimension": values.shape[1],
        "effective_rank": float(torch.exp(entropy)),
        "stable_rank": float(total / eigenvalues.max().clamp_min(torch.finfo(eigenvalues.dtype).tiny)),
        "participation_ratio": float(total.square() / eigenvalues.square().sum().clamp_min(1e-12)),
        "top1_energy": float(eigenvalues[:1].sum() / total),
        "top5_energy": float(eigenvalues[:5].sum() / total),
        "top10_energy": float(eigenvalues[:10].sum() / total),
        "dead_dimension_ratio": float(variance.lt(mean_variance * float(dead_variance_ratio)).float().mean()),
        "feature_variance_mean": float(variance.mean()),
        "feature_variance_std": float(variance.std(unbiased=False)),
        "feature_variance_min": float(variance.min()),
        "feature_variance_max": float(variance.max()),
        "mean_pairwise_cosine": float(pairwise_cosine),
        "eigenvalues": eigenvalues,
    }


def scatter_ratio(features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    values = torch.as_tensor(features, dtype=torch.float32)
    target = torch.as_tensor(labels, dtype=torch.long, device=values.device).reshape(-1)
    if values.ndim != 2 or values.shape[0] != target.numel():
        raise ValueError("scatter ratio inputs do not align.")
    global_mean = values.mean(dim=0)
    within = values.new_tensor(0.0)
    between = values.new_tensor(0.0)
    centroids = []
    for label in target.unique(sorted=True):
        selected = values[target.eq(label)]
        centroid = selected.mean(dim=0)
        centroids.append(centroid)
        within += (selected - centroid).square().sum()
        between += selected.shape[0] * (centroid - global_mean).square().sum()
    return {
        "within_class_scatter": float(within / max(1, values.shape[0])),
        "between_class_scatter": float(between / max(1, values.shape[0])),
        "fisher_ratio": float(between / within.clamp_min(1e-12)),
        "centroid_separation": float(torch.pdist(torch.stack(centroids)).mean()) if len(centroids) > 1 else 0.0,
    }


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    first = torch.as_tensor(left, dtype=torch.float32)
    second = torch.as_tensor(right, dtype=torch.float32, device=first.device)
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0] or first.shape[0] < 2:
        raise ValueError("linear CKA requires aligned [N,D] matrices.")
    first = first - first.mean(dim=0, keepdim=True)
    second = second - second.mean(dim=0, keepdim=True)
    if first.shape[0] < min(first.shape[1], second.shape[1]):
        left_gram = first.matmul(first.t())
        right_gram = second.matmul(second.t())
        numerator = (left_gram * right_gram).sum()
        denominator = left_gram.square().sum().sqrt() * right_gram.square().sum().sqrt()
    else:
        cross = first.t().matmul(second)
        left_cov = first.t().matmul(first)
        right_cov = second.t().matmul(second)
        numerator = cross.square().sum()
        denominator = left_cov.square().sum().sqrt() * right_cov.square().sum().sqrt()
    return float(numerator / denominator.clamp_min(1e-12))


def minimal_interpolation_alpha(
    missing: torch.Tensor,
    destination: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    steps: int = 100,
) -> torch.Tensor:
    source = torch.as_tensor(missing, dtype=torch.float32)
    target_feature = torch.as_tensor(destination, dtype=torch.float32, device=source.device)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=source.device).reshape(-1)
    bank = normalize(torch.as_tensor(prototypes, dtype=torch.float32, device=source.device))
    if source.shape != target_feature.shape or source.shape[0] != labels_tensor.numel() or int(steps) <= 0:
        raise ValueError("counterfactual inputs do not align.")
    result = torch.full((source.shape[0],), torch.inf, dtype=torch.float32, device=source.device)
    unresolved = torch.ones(source.shape[0], dtype=torch.bool, device=source.device)
    for index in range(int(steps) + 1):
        alpha = index / int(steps)
        feature = normalize((1.0 - alpha) * source + alpha * target_feature)
        recovered = feature.matmul(bank.t()).argmax(dim=1).eq(labels_tensor) & unresolved
        result[recovered] = float(alpha)
        unresolved &= ~recovered
        if not bool(unresolved.any()):
            break
    return result


def fit_pca_directions(features: torch.Tensor, *, max_components: int = 32) -> dict[str, torch.Tensor]:
    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA requires [N,D] with N >= 2.")
    mean = values.mean(dim=0)
    _, singular, basis = torch.pca_lowrank(values - mean, q=min(int(max_components), values.shape[0] - 1, values.shape[1]), center=False)
    return {"mean": mean, "basis": basis, "singular_values": singular}


def project_rows(features: torch.Tensor, basis: torch.Tensor, components: int) -> torch.Tensor:
    values = torch.as_tensor(features)
    directions = torch.as_tensor(basis, dtype=values.dtype, device=values.device)[:, : int(components)]
    return values.matmul(directions).matmul(directions.t())


def cosine_knn(
    query: torch.Tensor,
    index: torch.Tensor,
    *,
    k: int,
    device: torch.device | str = "cpu",
    chunk_size: int = 256,
    exclude_self: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_device = torch.device(device)
    queries = normalize(torch.as_tensor(query, dtype=torch.float32)).to(target_device)
    database = normalize(torch.as_tensor(index, dtype=torch.float32)).to(target_device)
    if queries.ndim != 2 or database.ndim != 2 or queries.shape[1] != database.shape[1]:
        raise ValueError("cosine kNN requires query/index matrices with the same feature dimension.")
    if exclude_self and queries.shape[0] != database.shape[0]:
        raise ValueError("self exclusion requires identical query/index row counts.")
    requested = int(k) + int(exclude_self)
    if requested >= database.shape[0]:
        raise ValueError("k must be smaller than the index size.")
    all_values = []
    all_indices = []
    for start in range(0, queries.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), queries.shape[0])
        similarity = queries[start:stop].matmul(database.t())
        if exclude_self:
            row = torch.arange(stop - start, device=target_device)
            similarity[row, torch.arange(start, stop, device=target_device)] = -torch.inf
        values, indices = similarity.topk(requested, dim=1)
        all_values.append(values[:, : int(k)].cpu())
        all_indices.append(indices[:, : int(k)].cpu())
    return torch.cat(all_values), torch.cat(all_indices)


def ridge_probe_fit(features: torch.Tensor, labels: torch.Tensor, *, l2: float = 1e-3, classes: int = 64) -> dict[str, torch.Tensor]:
    values = torch.as_tensor(features, dtype=torch.float64)
    target = torch.as_tensor(labels, dtype=torch.long, device=values.device).reshape(-1)
    if values.ndim != 2 or values.shape[0] != target.numel():
        raise ValueError("ridge probe inputs do not align.")
    mean = values.mean(dim=0)
    scale = values.std(dim=0).clamp_min(1e-6)
    standardized = (values - mean) / scale
    design = torch.cat((standardized, torch.ones(values.shape[0], 1, dtype=values.dtype, device=values.device)), dim=1)
    identity = torch.eye(design.shape[1], dtype=values.dtype, device=values.device)
    identity[-1, -1] = 0.0
    one_hot = F.one_hot(target, num_classes=int(classes)).to(values.dtype)
    weight = torch.linalg.solve(design.t().matmul(design) + float(l2) * identity, design.t().matmul(one_hot))
    return {"mean": mean.float(), "scale": scale.float(), "weight": weight.float()}


def ridge_probe_predict(features: torch.Tensor, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    values = torch.as_tensor(features, dtype=torch.float32)
    mean = torch.as_tensor(state["mean"], dtype=values.dtype, device=values.device)
    scale = torch.as_tensor(state["scale"], dtype=values.dtype, device=values.device)
    weight = torch.as_tensor(state["weight"], dtype=values.dtype, device=values.device)
    design = torch.cat(((values - mean) / scale, torch.ones(values.shape[0], 1, device=values.device)), dim=1)
    return design.matmul(weight)


def fit_logistic_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 8192,
    seed: int = 44001,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float32)
    target = np.asarray(labels, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or values.shape[0] != target.size or set(np.unique(target)) != {0.0, 1.0}:
        raise ValueError("logistic probe requires finite binary [N,D] inputs.")
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    standardized = (values - mean) / scale
    target_device = torch.device(device)
    torch.manual_seed(int(seed))
    model = torch.nn.Linear(values.shape[1], 1).to(target_device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=float(l2))
    positive = float(target.sum())
    negative = float(target.size - positive)
    class_weights = torch.tensor((target.size / (2.0 * negative), target.size / (2.0 * positive)), dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed))
    x_cpu = torch.from_numpy(standardized)
    y_cpu = torch.from_numpy(target)
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(target.size, generator=generator)
        for start in range(0, target.size, int(batch_size)):
            selected = order[start : start + int(batch_size)]
            x = x_cpu[selected].to(target_device)
            y = y_cpu[selected].to(target_device)
            logits = model(x).squeeze(1)
            weights = torch.where(y > 0.5, class_weights[1], class_weights[0]).to(target_device)
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return {
        "mean": mean,
        "scale": scale,
        "weight": model.weight.detach().cpu().numpy().reshape(-1),
        "bias": float(model.bias.detach().cpu()),
        "l2": float(l2),
        "epochs": int(epochs),
        "seed": int(seed),
    }


def predict_logistic_probe(features: np.ndarray, state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    logits = ((values - np.asarray(state["mean"])) / np.asarray(state["scale"])) @ np.asarray(state["weight"])
    logits = logits + float(state["bias"])
    probability = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    probability[~positive] = exponential / (1.0 + exponential)
    return logits, probability


def binary_probe_metrics(labels: np.ndarray, probability: np.ndarray, *, bins: int = 15) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.int8).reshape(-1)
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    if target.size != score.size or set(np.unique(target)) != {0, 1} or not np.isfinite(score).all():
        raise ValueError("binary metrics require aligned finite labels and probabilities.")
    ranks = rankdata(score, method="average")
    positive = target == 1
    negative = ~positive
    roc_auc = (ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2.0) / (positive.sum() * negative.sum())
    order = np.argsort(-score, kind="stable")
    sorted_target = target[order]
    tp = np.cumsum(sorted_target)
    fp = np.cumsum(1 - sorted_target)
    recall = tp / positive.sum()
    precision = tp / np.maximum(1, tp + fp)
    pr_auc = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    prediction = score >= 0.5
    true_positive = int(np.sum(prediction & positive))
    true_negative = int(np.sum(~prediction & negative))
    false_positive = int(np.sum(prediction & negative))
    false_negative = int(np.sum(~prediction & positive))
    top_count = max(1, int(math.ceil(0.2 * target.size)))
    selected = order[:top_count]
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, int(bins) + 1)[:-1]:
        chosen = (score > lower) & (score <= lower + 1.0 / int(bins))
        if np.any(chosen):
            ece += float(chosen.mean() * abs(score[chosen].mean() - target[chosen].mean()))
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": pr_auc,
        "balanced_accuracy": 0.5 * (true_positive / positive.sum() + true_negative / negative.sum()),
        "f1": 2.0 * true_positive / max(1, 2 * true_positive + false_positive + false_negative),
        "recall_at_20pct": float(target[selected].sum() / positive.sum()),
        "precision_at_20pct": float(target[selected].mean()),
        "brier": float(np.mean((score - target) ** 2)),
        "ece": float(ece),
        "prevalence": float(target.mean()),
    }


def bootstrap_mean_interval(
    values: np.ndarray | torch.Tensor,
    *,
    replicates: int = 1000,
    seed: int = 44001,
    confidence: float = 0.95,
) -> dict[str, float]:
    array = np.asarray(torch.as_tensor(values).cpu(), dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be non-empty and finite.")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), 100):
        stop = min(start + 100, int(replicates))
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        estimates[start:stop] = array[indices].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(estimates, tail)),
        "ci_high": float(np.quantile(estimates, 1.0 - tail)),
    }


def validate_train_only_selection(selection: Mapping[str, Any]) -> None:
    source_roles = {str(value) for value in selection.get("source_roles", ())}
    forbidden = source_roles - {"train", "split_independent"}
    if forbidden:
        raise ValueError(f"selection contains non-train roles: {sorted(forbidden)}")
    if selection.get("validation_leakage_oracle") is not False:
        raise ValueError("formal selection must set validation_leakage_oracle=false.")


__all__ = [
    "GROUP_NAMES",
    "binary_probe_metrics",
    "bootstrap_mean_interval",
    "classification_groups",
    "cosine_knn",
    "decision_decomposition",
    "final_geometry",
    "fit_logistic_probe",
    "fit_pca_directions",
    "highest_confuser",
    "linear_cka",
    "minimal_interpolation_alpha",
    "nonempty_subset_utilities",
    "predict_logistic_probe",
    "predictive_statistics",
    "project_rows",
    "representation_spectrum",
    "ridge_probe_fit",
    "ridge_probe_predict",
    "scatter_ratio",
    "validate_pair_alignment",
    "validate_safety_contract",
    "validate_train_only_selection",
]
