from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


SELECTION_MODES = ("fixed_same_for_all", "random_per_sample", "learned_lookup", "all_candidates_oracle")


class PrototypePilotSelector(nn.Module):
    def __init__(self, num_prototypes: int, num_candidate_patterns: int, *, num_selected_patterns: int = 4) -> None:
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.num_candidate_patterns = int(num_candidate_patterns)
        self.num_selected_patterns = int(num_selected_patterns)
        if min(self.num_prototypes, self.num_candidate_patterns, self.num_selected_patterns) <= 0:
            raise ValueError("Prototype pilot selector dimensions must be positive.")
        if self.num_selected_patterns > self.num_candidate_patterns:
            raise ValueError("num_selected_patterns cannot exceed num_candidate_patterns.")
        self.pilot_logits = nn.Parameter(torch.zeros(self.num_prototypes, self.num_candidate_patterns))

    def forward(
        self,
        proto_id: torch.Tensor,
        candidate_g: torch.Tensor,
        *,
        mode: str = "learned_lookup",
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        num_selected_patterns: int | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = str(mode)
        if mode not in SELECTION_MODES:
            raise ValueError(f"Unsupported pilot selection mode {mode!r}.")
        values = torch.as_tensor(candidate_g)
        ids = torch.as_tensor(proto_id, device=values.device, dtype=torch.long).reshape(-1)
        if values.ndim != 3 or values.shape[0] != ids.numel() or values.shape[1] != self.num_candidate_patterns:
            raise ValueError("candidate_g and proto_id must have shapes [B,R,K] and [B].")
        if bool(((ids < 0) | (ids >= self.num_prototypes)).any().item()):
            raise ValueError("proto_id is outside the prototype lookup table.")

        requested = self.num_selected_patterns if num_selected_patterns is None else int(num_selected_patterns)
        if requested <= 0 or requested > self.num_candidate_patterns:
            raise ValueError("Runtime num_selected_patterns must be within the candidate pattern budget.")
        count = self.num_candidate_patterns if mode == "all_candidates_oracle" else requested
        if mode == "fixed_same_for_all" or mode == "all_candidates_oracle":
            pattern_ids = torch.arange(count, device=values.device).expand(ids.numel(), -1)
            selected = _gather_candidates(values, pattern_ids)
        elif mode == "random_per_sample":
            random_scores = torch.rand(
                ids.numel(), self.num_candidate_patterns, device=values.device, generator=generator
            )
            pattern_ids = random_scores.topk(count, dim=-1).indices
            selected = _gather_candidates(values, pattern_ids)
        elif self.training:
            pattern_ids, selection = _straight_through_gumbel_topk(
                self.pilot_logits[ids], count, temperature=float(temperature), generator=generator
            )
            selected = torch.einsum("bmr,brk->bmk", selection.to(values.dtype), values)
        else:
            pattern_ids = self.pilot_logits[ids].topk(count, dim=-1).indices
            selected = _gather_candidates(values, pattern_ids)
        return {"selected_y": selected, "pattern_ids": pattern_ids}

    def usage_regularization(self) -> torch.Tensor:
        probabilities = self.pilot_logits.softmax(dim=-1).mean(dim=0)
        return (probabilities * (probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log())).sum() + torch.log(
            probabilities.new_tensor(float(self.num_candidate_patterns))
        )

    def lookup(self, num_selected_patterns: int | None = None) -> torch.Tensor:
        count = self.num_selected_patterns if num_selected_patterns is None else int(num_selected_patterns)
        if count <= 0 or count > self.num_candidate_patterns:
            raise ValueError("Lookup width must be within the candidate pattern budget.")
        return self.pilot_logits.detach().topk(count, dim=-1).indices.cpu()

    def export_lookup(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        num_selected_patterns: int | None = None,
    ) -> Path:
        if metadata:
            source = str(metadata.get("source_split", metadata.get("split", ""))).lower()
            if "train" not in source or source in {"validation", "test", "outer_test"}:
                raise ValueError("Prototype pilot lookup export is train-only.")
            if metadata.get("outer_test_accessed") is True:
                raise ValueError("Prototype pilot lookup export is train-only and must not access outer test.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lookup = self.lookup(num_selected_patterns).tolist()
        payload: dict[str, Any] = {f"prototype_{index}": patterns for index, patterns in enumerate(lookup)}
        if metadata:
            payload["_metadata"] = metadata
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target


def load_prototype_pilot_lookup(
    path: str | Path,
    *,
    num_prototypes: int,
    num_selected_patterns: int,
    num_candidate_patterns: int,
) -> torch.Tensor:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [payload.get(f"prototype_{index}") for index in range(int(num_prototypes))]
    lookup = torch.as_tensor(rows, dtype=torch.long)
    expected = (int(num_prototypes), int(num_selected_patterns))
    if tuple(lookup.shape) != expected:
        raise ValueError(f"Prototype pilot lookup must have shape {expected}.")
    if bool(((lookup < 0) | (lookup >= int(num_candidate_patterns))).any().item()):
        raise ValueError("Prototype pilot lookup contains an invalid pattern id.")
    if any(torch.unique(row).numel() != row.numel() for row in lookup):
        raise ValueError("Each prototype lookup row must contain distinct pattern ids.")
    return lookup


def select_from_lookup(proto_id: torch.Tensor, candidate_g: torch.Tensor, lookup: torch.Tensor) -> dict[str, torch.Tensor]:
    values = torch.as_tensor(candidate_g)
    ids = torch.as_tensor(proto_id, device=values.device, dtype=torch.long).reshape(-1)
    table = torch.as_tensor(lookup, device=values.device, dtype=torch.long)
    pattern_ids = table[ids]
    return {"selected_y": _gather_candidates(values, pattern_ids), "pattern_ids": pattern_ids}


def _straight_through_gumbel_topk(
    logits: torch.Tensor,
    count: int,
    *,
    temperature: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("Gumbel temperature must be positive.")
    uniform = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype, generator=generator)
    perturbed = logits - torch.log(-torch.log(uniform.clamp_(1e-6, 1.0 - 1e-6)))
    available = torch.ones_like(logits, dtype=torch.bool)
    hard_ids = []
    selections = []
    for _ in range(int(count)):
        masked = perturbed.masked_fill(~available, torch.finfo(logits.dtype).min)
        soft = F.softmax(masked / float(temperature), dim=-1)
        ids = masked.argmax(dim=-1)
        hard = F.one_hot(ids, num_classes=logits.shape[-1]).to(logits.dtype)
        selections.append(hard + soft - soft.detach())
        hard_ids.append(ids)
        available = available.scatter(1, ids[:, None], False)
    return torch.stack(hard_ids, dim=1), torch.stack(selections, dim=1)


def _gather_candidates(values: torch.Tensor, pattern_ids: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[batch, pattern_ids]


__all__ = [
    "PrototypePilotSelector",
    "SELECTION_MODES",
    "load_prototype_pilot_lookup",
    "select_from_lookup",
]
