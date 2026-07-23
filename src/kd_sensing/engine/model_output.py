from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    input_features: torch.Tensor | None
    output_features: torch.Tensor | None
    diagnostics: dict[str, Any]


def adapt_model_output(output: dict[str, Any]) -> ModelOutput:
    """Adapt the canonical current model output dictionary."""

    if not isinstance(output, dict):
        raise TypeError("Retained models must return a dictionary.")
    logits = output.get("logits")
    if not torch.is_tensor(logits):
        raise ValueError("Model output must contain Tensor field 'logits'.")
    input_features = output.get("input_features")
    output_features = output.get("output_features")
    return ModelOutput(
        logits=logits,
        input_features=input_features if torch.is_tensor(input_features) else None,
        output_features=output_features if torch.is_tensor(output_features) else None,
        diagnostics={
            key: value
            for key, value in output.items()
            if key not in {"logits", "input_features", "output_features"}
        },
    )


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
