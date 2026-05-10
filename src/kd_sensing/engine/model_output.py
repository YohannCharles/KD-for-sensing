from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    input_features: torch.Tensor | None
    output_features: torch.Tensor | None
    diagnostics: dict[str, Any]


_LOGIT_KEYS = ("logits", "pred", "prediction", "outputs")
_INPUT_FEATURE_KEYS = ("input_features", "features", "token_features", "encoder_features")
_OUTPUT_FEATURE_KEYS = ("output_features", "out_features", "fusion_memory", "memory")


def adapt_model_output(output: Any) -> ModelOutput:
    """Normalize model outputs without inventing missing feature tensors."""

    if isinstance(output, dict):
        logits = _first_tensor(output, _LOGIT_KEYS)
        if logits is None:
            raise ValueError(f"Model output dict must contain one of {list(_LOGIT_KEYS)}.")
        input_features = _first_tensor(output, _INPUT_FEATURE_KEYS)
        output_features = _first_tensor(output, _OUTPUT_FEATURE_KEYS)
        diagnostics = {
            key: value
            for key, value in output.items()
            if key not in set(_LOGIT_KEYS + _INPUT_FEATURE_KEYS + _OUTPUT_FEATURE_KEYS)
        }
        return ModelOutput(
            logits=logits,
            input_features=input_features,
            output_features=output_features,
            diagnostics=diagnostics,
        )

    if isinstance(output, (tuple, list)) and len(output) >= 3:
        logits, input_features, output_features = output[:3]
        if not torch.is_tensor(logits):
            raise TypeError("Legacy model output first item must be a Tensor of logits.")
        diagnostics = getattr(output, "diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        return ModelOutput(
            logits=logits,
            input_features=input_features if torch.is_tensor(input_features) else None,
            output_features=output_features if torch.is_tensor(output_features) else None,
            diagnostics=diagnostics,
        )

    if torch.is_tensor(output):
        return ModelOutput(
            logits=output,
            input_features=None,
            output_features=None,
            diagnostics={},
        )

    raise TypeError(f"Unsupported model output type: {type(output).__name__}.")


def select_prediction_slots(logits: torch.Tensor, num_pred: int) -> torch.Tensor:
    """Return the final future slots from long time-series logits."""

    horizon = int(num_pred)
    if horizon <= 0:
        raise ValueError(f"num_pred must be positive, got {num_pred}.")
    if logits.ndim != 3:
        raise ValueError(f"Model logits must have shape [B, T, C], got {tuple(logits.shape)}.")
    if logits.shape[1] < horizon:
        raise ValueError(
            f"Model logits provide {logits.shape[1]} slots but future labels require {horizon} slots."
        )
    if logits.shape[1] == horizon:
        return logits
    return logits[:, -horizon:, :]


def _first_tensor(output: dict[str, Any], keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        value = output.get(key)
        if torch.is_tensor(value):
            return value
    return None
