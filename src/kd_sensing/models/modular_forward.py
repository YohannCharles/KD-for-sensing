from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModularForwardInputs:
    raw_inputs: dict[str, torch.Tensor | None]
    reliability_inputs: dict[str, Any]
    modality_valid_inputs: dict[str, torch.Tensor | None]
    modality_dropout_inputs: dict[str, torch.Tensor | None]


@dataclass(frozen=True)
class EncoderProjectorStage:
    encoded: dict[str, torch.Tensor]
    projected: dict[str, torch.Tensor]
    encoder_auxiliary_features: dict[str, dict[str, torch.Tensor]]
    encoder_runtime_metadata: dict[str, Any]


@dataclass(frozen=True)
class CoreInputStage:
    core_input: torch.Tensor
    availability_mask: torch.Tensor | None
    input_features: torch.Tensor
    has_token_features: bool


@dataclass(frozen=True)
class LogitPostProcessStage:
    logits: torch.Tensor
    geometry_prior_payload: dict[str, Any] | None
    geometry_fusion_payload: dict[str, Any] | None
    rerank_payload: dict[str, Any] | None


def collect_forward_inputs(**kwargs: Any) -> ModularForwardInputs:
    raw_inputs = {
        "image": kwargs["image_batch"],
        "radar": kwargs["radar_batch"],
        "gps": kwargs["gps_batch"],
        "lidar": kwargs["lidar_batch"],
        "mmwave": kwargs["mmwave_batch"],
        "csi": kwargs["csi_batch"],
    }
    reliability_inputs = {
        "image_valid_mask": kwargs["image_valid_mask"],
        "radar_valid_mask": kwargs["radar_valid_mask"],
        "image_observability_score": kwargs["image_observability_score"],
        "gps_valid_mask": kwargs["gps_valid_mask"],
        "lidar_valid_mask": kwargs["lidar_valid_mask"],
        "gps_delay_steps": kwargs["gps_delay_steps"],
        "image_dropout_mask": kwargs["image_dropout_mask"],
        "radar_dropout_mask": kwargs["radar_dropout_mask"],
        "gps_dropout_mask": kwargs["gps_dropout_mask"],
        "lidar_dropout_mask": kwargs["lidar_dropout_mask"],
        "gps_counterfactual_mask": kwargs["gps_counterfactual_mask"],
        "benchmark_condition_metadata": kwargs["benchmark_condition_metadata"],
    }
    modality_valid_inputs = {
        "image": kwargs["image_valid_mask"],
        "radar": kwargs["radar_valid_mask"],
        "gps": kwargs["gps_valid_mask"],
        "lidar": kwargs["lidar_valid_mask"],
        "mmwave": None,
        "csi": None,
    }
    modality_dropout_inputs = {
        "image": kwargs["image_dropout_mask"],
        "radar": kwargs["radar_dropout_mask"],
        "gps": kwargs["gps_dropout_mask"],
        "lidar": kwargs["lidar_dropout_mask"],
        "mmwave": None,
        "csi": None,
    }
    return ModularForwardInputs(
        raw_inputs=raw_inputs,
        reliability_inputs=reliability_inputs,
        modality_valid_inputs=modality_valid_inputs,
        modality_dropout_inputs=modality_dropout_inputs,
    )


def run_encoder_projector_stage(
    model: Any,
    raw_inputs: dict[str, torch.Tensor | None],
    reliability_inputs: dict[str, Any],
) -> EncoderProjectorStage:
    encoded: dict[str, torch.Tensor] = {}
    projected: dict[str, torch.Tensor] = {}
    encoder_auxiliary_features: dict[str, dict[str, torch.Tensor]] = {}
    encoder_runtime_metadata: dict[str, Any] = {}
    batch_size = None
    seq_len = None
    pending = list(model.modalities)
    while pending:
        progressed = False
        for modality in list(pending):
            encoder = model.encoders[modality]
            dependencies = encoder_context_dependencies(encoder)
            source = encoder_context_source(encoder)
            if not encoder_dependencies_satisfied(dependencies, source=source, encoded=encoded, projected=projected):
                continue
            tensor = raw_inputs[modality]
            if tensor is None:
                raise ValueError(f"Modular sequence model requires '{modality}' input because it is enabled.")
            context_kwargs = encoder_context_kwargs(
                encoder,
                modality=modality,
                raw_tensor=tensor,
                dependencies=dependencies,
                raw_inputs=raw_inputs,
                encoded=encoded,
                projected=projected,
            )
            context_kwargs.update(
                encoder_reliability_kwargs(
                    encoder,
                    modality=modality,
                    reliability_inputs=reliability_inputs,
                )
            )
            features = encoder(tensor, **context_kwargs) if context_kwargs else encoder(tensor)
            collect_encoder_runtime_metadata(
                encoder,
                modality=modality,
                encoder_auxiliary_features=encoder_auxiliary_features,
                encoder_runtime_metadata=encoder_runtime_metadata,
            )
            batch_size, seq_len = check_temporal_features(features, modality, batch_size, seq_len)
            encoded[modality] = features
            projected_features = model.projectors[modality](features)
            check_projected_features(projected_features, modality, model.d_model)
            projected[modality] = projected_features
            pending.remove(modality)
            progressed = True
        if not progressed:
            unmet = {
                modality: unmet_context_dependencies(
                    encoder_context_dependencies(model.encoders[modality]),
                    source=encoder_context_source(model.encoders[modality]),
                    encoded=encoded,
                    projected=projected,
                )
                for modality in pending
            }
            raise ValueError(
                "Unable to satisfy modular sequence encoder condition dependencies; "
                f"pending modalities={pending}, unmet dependencies={unmet}. "
                "Check for missing condition modalities or circular dependencies."
            )
    return EncoderProjectorStage(
        encoded=encoded,
        projected=projected,
        encoder_auxiliary_features=encoder_auxiliary_features,
        encoder_runtime_metadata=encoder_runtime_metadata,
    )


def collect_encoder_runtime_metadata(
    encoder: nn.Module,
    *,
    modality: str,
    encoder_auxiliary_features: dict[str, dict[str, torch.Tensor]],
    encoder_runtime_metadata: dict[str, Any],
) -> None:
    temporal_aux_metadata = getattr(encoder, "last_temporal_auxiliary_metadata", None)
    if isinstance(temporal_aux_metadata, dict) and bool(temporal_aux_metadata.get("enabled", False)):
        current_latent = getattr(encoder, "last_current_latent", None)
        predicted_latent = getattr(encoder, "last_temporal_predicted_latent", None)
        if isinstance(current_latent, torch.Tensor) and isinstance(predicted_latent, torch.Tensor):
            encoder_auxiliary_features[modality] = {
                "current_latent": current_latent,
                "temporal_predicted_latent": predicted_latent,
            }
        encoder_runtime_metadata[modality] = {
            "temporal_auxiliary": temporal_aux_metadata,
        }
    predictive_diagnostics = getattr(encoder, "last_predictive_gps_query_diagnostics", None)
    if isinstance(predictive_diagnostics, dict):
        encoder_runtime_metadata.setdefault(modality, {})["predictive_gps_query"] = predictive_diagnostics
    visual_token_diagnostics = getattr(encoder, "last_visual_token_diagnostics", None)
    if isinstance(visual_token_diagnostics, dict) and visual_token_diagnostics:
        encoder_runtime_metadata.setdefault(modality, {})["visual_tokens"] = visual_token_diagnostics


def assemble_core_input_stage(
    model: Any,
    projected: dict[str, torch.Tensor],
    *,
    modality_valid_inputs: dict[str, torch.Tensor | None],
    modality_dropout_inputs: dict[str, torch.Tensor | None],
    modality_availability_overrides: dict[str, torch.Tensor | None] | None = None,
) -> CoreInputStage:
    ordered = [projected[modality] for modality in model.modalities]
    has_token_features = any(features.ndim == 4 for features in ordered)
    if has_token_features:
        return assemble_token_core_input(
            model,
            projected,
            ordered,
            modality_valid_inputs,
            modality_dropout_inputs,
            modality_availability_overrides or {},
        )
    if len(ordered) == 1:
        core_input = ordered[0]
        availability_mask = core_input_availability_mask(
            projected,
            model.modalities,
            valid_masks=modality_valid_inputs,
            dropout_masks=modality_dropout_inputs,
            availability_overrides=modality_availability_overrides or {},
            token_features=False,
        )
        if availability_mask is not None:
            core_input = core_input * availability_mask[:, 0, :].unsqueeze(-1).to(dtype=core_input.dtype)
        return CoreInputStage(core_input, availability_mask, core_input, False)
    core_input = torch.stack(ordered, dim=1)
    availability_mask = core_input_availability_mask(
        projected,
        model.modalities,
        valid_masks=modality_valid_inputs,
        dropout_masks=modality_dropout_inputs,
        availability_overrides=modality_availability_overrides or {},
        token_features=False,
    )
    if availability_mask is not None:
        core_input = core_input * availability_mask.unsqueeze(-1).to(dtype=core_input.dtype)
    input_features = torch.cat([core_input[:, index, :, :] for index in range(int(core_input.shape[1]))], dim=-1)
    return CoreInputStage(core_input, availability_mask, input_features, False)


def assemble_token_core_input(
    model: Any,
    projected: dict[str, torch.Tensor],
    ordered: list[torch.Tensor],
    modality_valid_inputs: dict[str, torch.Tensor | None],
    modality_dropout_inputs: dict[str, torch.Tensor | None],
    modality_availability_overrides: dict[str, torch.Tensor | None],
) -> CoreInputStage:
    token_pieces = [features if features.ndim == 4 else features.unsqueeze(2) for features in ordered]
    modality_masks = [
        modality_availability_from_inputs(
            modality,
            projected[modality],
            valid_mask=modality_valid_inputs.get(modality),
            dropout_mask=modality_dropout_inputs.get(modality),
            availability_override=modality_availability_overrides.get(modality),
        )
        for modality in model.modalities
    ]
    masked_for_input = [
        features * mask.unsqueeze(2).unsqueeze(-1).to(dtype=features.dtype)
        for features, mask in zip(token_pieces, modality_masks)
    ]
    if bool(getattr(model.representation_core, "supports_spatial_modality_tokens", False)):
        max_tokens = max(int(features.shape[2]) for features in token_pieces)
        padded_tokens: list[torch.Tensor] = []
        padded_masks: list[torch.Tensor] = []
        for features, mask in zip(masked_for_input, modality_masks):
            token_count = int(features.shape[2])
            padded_tokens.append(pad_modality_tokens(features, max_tokens=max_tokens))
            token_mask = mask.unsqueeze(2).expand(-1, -1, token_count)
            padded_masks.append(pad_modality_token_mask(token_mask, max_tokens=max_tokens))
        core_input = torch.stack(padded_tokens, dim=1).contiguous()
        availability_mask = torch.stack(padded_masks, dim=1).contiguous()
    else:
        token_features = torch.cat(token_pieces, dim=2)
        core_input = token_features.permute(0, 2, 1, 3).contiguous()
        availability_mask = core_input_availability_mask(
            projected,
            model.modalities,
            valid_masks=modality_valid_inputs,
            dropout_masks=modality_dropout_inputs,
            availability_overrides=modality_availability_overrides,
            token_features=True,
        )
        if availability_mask is not None:
            core_input = core_input * availability_mask.unsqueeze(-1).to(dtype=core_input.dtype)
    input_features = torch.cat(
        [
            features.mean(dim=2)
            for features in masked_for_input
        ],
        dim=-1,
    )
    return CoreInputStage(core_input, availability_mask, input_features, True)


def run_core_head_stage(
    model: Any,
    core_input: torch.Tensor,
    availability_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bool(getattr(model.representation_core, "supports_missing_modality_metadata", False)):
        if core_input.ndim == 3:
            core_input = core_input.unsqueeze(1)
        output_features = model.representation_core(core_input, modality_available=availability_mask)
    else:
        output_features = model.representation_core(core_input)
    return output_features, model.heads["beam"](output_features)


def post_process_logits_stage(
    model: Any,
    image_logits: torch.Tensor,
    *,
    gps_batch: torch.Tensor | None,
    image_valid_mask: torch.Tensor | None,
    image_observability_score: torch.Tensor | None,
    gps_valid_mask: torch.Tensor | None,
    gps_delay_steps: torch.Tensor | None,
    gps_counterfactual_mask: torch.Tensor | None,
) -> LogitPostProcessStage:
    logits = image_logits
    geometry_prior_payload: dict[str, Any] | None = None
    geometry_fusion_payload: dict[str, Any] | None = None
    rerank_payload: dict[str, Any] | None = None
    if model.geometry_prior is not None and model.geometry_prior_fusion is not None:
        geometry_prior_payload = model.geometry_prior(
            gps_batch,
            target_time=int(image_logits.shape[1]),
            gps_valid_mask=gps_valid_mask,
            gps_delay_steps=gps_delay_steps,
            gps_counterfactual_mask=gps_counterfactual_mask,
        )
        geometry_fusion_payload = model.geometry_prior_fusion(
            image_logits=image_logits,
            prior_logits=geometry_prior_payload["logits"],
            prior_distribution=geometry_prior_payload.get("distribution"),
            prior_availability_mask=geometry_prior_payload.get("availability_mask"),
            image_valid_mask=image_valid_mask,
            image_observability_score=image_observability_score,
            gps_valid_mask=gps_valid_mask,
            gps_delay_steps=gps_delay_steps,
            gps_counterfactual_mask=gps_counterfactual_mask,
        )
        logits = geometry_fusion_payload["logits"]
    if model.reranker is not None:
        rerank_payload = model.reranker(
            anchor_logits=image_logits,
            geometry_prior_logits=geometry_prior_payload["logits"] if geometry_prior_payload is not None else None,
            image_observability_score=image_observability_score,
            gps_valid_mask=gps_valid_mask,
            gps_delay_steps=gps_delay_steps,
            gps_counterfactual_mask=gps_counterfactual_mask,
        )
        logits = rerank_payload["logits"]
    return LogitPostProcessStage(logits, geometry_prior_payload, geometry_fusion_payload, rerank_payload)


def assemble_model_output_stage(
    model: Any,
    *,
    logits: torch.Tensor,
    image_logits: torch.Tensor,
    input_features: torch.Tensor,
    output_features: torch.Tensor,
    core_input: torch.Tensor,
    availability_mask: torch.Tensor | None,
    has_token_features: bool,
    encoded: dict[str, torch.Tensor],
    projected: dict[str, torch.Tensor],
    encoder_auxiliary_features: dict[str, dict[str, torch.Tensor]],
    encoder_runtime_metadata: dict[str, Any],
    geometry_prior_payload: dict[str, Any] | None,
    geometry_fusion_payload: dict[str, Any] | None,
    rerank_payload: dict[str, Any] | None,
    missing_modality_metadata_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = {
        "logits": logits,
        "input_features": input_features,
        "output_features": output_features,
        "modalities": model.modalities,
        "modality_features": projected,
        "encoder_features": encoded,
        "image_profile": model.image_profile,
    }
    if availability_mask is not None or bool(getattr(model.representation_core, "supports_missing_modality_metadata", False)):
        output_metadata = missing_modality_output_metadata(
            availability_mask,
            modalities=model.modalities,
        )
        if isinstance(missing_modality_metadata_input, dict):
            output_metadata["input_metadata"] = dict(missing_modality_metadata_input)
        output["missing_modality_metadata"] = output_metadata
    if has_token_features:
        output["token_features"] = core_input
    attach_geometry_outputs(output, image_logits, geometry_prior_payload, geometry_fusion_payload, rerank_payload)
    attach_runtime_outputs(output, encoder_auxiliary_features, encoder_runtime_metadata)
    attach_auxiliary_outputs(model, output, output_features)
    return output


def attach_geometry_outputs(
    output: dict[str, Any],
    image_logits: torch.Tensor,
    geometry_prior_payload: dict[str, Any] | None,
    geometry_fusion_payload: dict[str, Any] | None,
    rerank_payload: dict[str, Any] | None,
) -> None:
    if geometry_prior_payload is not None and geometry_fusion_payload is not None:
        fusion_diagnostics = dict(geometry_fusion_payload.get("diagnostics", {}))
        output.update(
            {
                "anchor_logits": image_logits,
                "image_logits": image_logits,
                "geometry_prior_logits": geometry_prior_payload["logits"],
                "geometry_prior_distribution": geometry_prior_payload["distribution"],
                "geometry_prior_entropy": geometry_prior_payload["entropy"],
                "geometry_prior_topk_indices": geometry_prior_payload["topk_indices"],
                "geometry_prior_topk_probabilities": geometry_prior_payload["topk_probabilities"],
                "geometry_prior_availability_mask": geometry_prior_payload["availability_mask"],
                "geometry_prior_unavailable_reason": geometry_prior_payload["unavailable_reason"],
                "geometry_prior_diagnostics": {
                    "entropy": geometry_prior_payload["entropy"],
                    "topk_indices": geometry_prior_payload["topk_indices"],
                    "availability_mask": geometry_prior_payload["availability_mask"],
                    "unavailable_reason": geometry_prior_payload["unavailable_reason"],
                    "metadata": geometry_prior_payload["metadata"],
                },
                "geometry_prior_fusion_diagnostics": fusion_diagnostics,
                "branch_weights": fusion_diagnostics.get("branch_weights"),
            }
        )
    elif rerank_payload is not None:
        output["anchor_logits"] = image_logits
    if rerank_payload is not None:
        rerank_diagnostics = dict(rerank_payload.get("diagnostics", {}))
        output.update(
            {
                "rerank_logits": rerank_diagnostics.get("rerank_logits", rerank_payload["logits"]),
                "safe_rerank_diagnostics": rerank_diagnostics,
                "candidate_ids": rerank_diagnostics.get("candidate_ids"),
                "candidate_source_mask": rerank_diagnostics.get("candidate_source_mask"),
                "selected_source": rerank_diagnostics.get("selected_source"),
                "target_rank_delta": rerank_diagnostics.get("target_rank_delta"),
                "fallback_reason": rerank_diagnostics.get("fallback_reason_code"),
                "gate_confidence": rerank_diagnostics.get("gate_confidence"),
                "condition_id_consumed": False,
            }
        )


def attach_runtime_outputs(
    output: dict[str, Any],
    encoder_auxiliary_features: dict[str, dict[str, torch.Tensor]],
    encoder_runtime_metadata: dict[str, Any],
) -> None:
    if encoder_auxiliary_features:
        output["encoder_auxiliary_features"] = encoder_auxiliary_features
    if encoder_runtime_metadata:
        output["runtime_metadata"] = {"encoder_temporal_auxiliary": encoder_runtime_metadata}
        predictive_runtime = {
            modality: metadata["predictive_gps_query"]
            for modality, metadata in encoder_runtime_metadata.items()
            if isinstance(metadata, dict) and isinstance(metadata.get("predictive_gps_query"), dict)
        }
        if predictive_runtime:
            output["predictive_gps_query_diagnostics"] = predictive_runtime


def attach_auxiliary_outputs(model: Any, output: dict[str, Any], output_features: torch.Tensor) -> None:
    feature_consistency_diagnostics = getattr(model.representation_core, "last_feature_consistency_diagnostics", None)
    if isinstance(feature_consistency_diagnostics, dict):
        output["feature_consistency_diagnostics"] = feature_consistency_diagnostics
    token_readout_diagnostics = getattr(model.representation_core, "last_token_readout_diagnostics", None)
    if isinstance(token_readout_diagnostics, dict):
        output["token_readout_diagnostics"] = token_readout_diagnostics
    amber_full_auxiliary = getattr(model.representation_core, "last_amber_full_auxiliary", None)
    if isinstance(amber_full_auxiliary, dict):
        output["amber_full_auxiliary"] = amber_full_auxiliary
    amber_full_attention_mask = getattr(model.representation_core, "last_amber_full_attention_mask", None)
    if torch.is_tensor(amber_full_attention_mask):
        output["amber_full_attention_key_padding_mask"] = amber_full_attention_mask
    amr_lite_gate_stats = getattr(model.representation_core, "last_amr_lite_gate_stats", None)
    if isinstance(amr_lite_gate_stats, list):
        output["amr_lite_gate_stats"] = amr_lite_gate_stats
    output.update(model.auxiliary_heads(output_features))


def encoder_context_dependencies(encoder: nn.Module) -> tuple[str, ...]:
    raw = getattr(encoder, "required_context_modalities", ())
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = (raw,)
    return tuple(str(item) for item in raw)


def encoder_context_source(encoder: nn.Module) -> str:
    source = str(getattr(encoder, "context_feature_source", "projected")).strip().lower()
    if source == "none":
        return source
    if source not in {"projected", "encoded", "raw"}:
        raise ValueError(
            "Encoder requested unsupported condition feature source "
            f"{source!r}; supported sources are 'projected', 'encoded', and 'raw'."
        )
    return source


def component_training_strategy_metadata(
    component: nn.Module,
    cfg: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    raw = component.training_strategy_metadata() if hasattr(component, "training_strategy_metadata") else {}
    metadata = dict(raw) if isinstance(raw, dict) else {}
    registry_type = cfg.get("type")
    if registry_type not in (None, ""):
        registry_type = str(registry_type)
        metadata.setdefault("type", registry_type)
        metadata.setdefault("registry_type", registry_type)
        if role == "encoder":
            metadata.setdefault("encoder", registry_type)
        elif role == "projector":
            metadata.setdefault("projector", registry_type)
        elif role == "representation_core":
            metadata.setdefault("core", registry_type)
        elif role == "head":
            metadata.setdefault("head", registry_type)
    metadata.setdefault("class", component.__class__.__name__)
    metadata.setdefault("component_role", role)
    if "consumes_reliability_metadata" not in metadata:
        metadata["consumes_reliability_metadata"] = component_consumes_reliability_metadata(component, metadata)
    return metadata


def component_consumes_reliability_metadata(component: nn.Module, metadata: dict[str, Any] | None = None) -> bool:
    metadata = metadata or {}
    for key in (
        "consumes_reliability_metadata",
        "supports_reliability_metadata",
        "supports_observability_metadata",
        "consumes_missing_modality_metadata",
    ):
        if key in metadata:
            return bool(metadata.get(key))
    temporal_fallback = metadata.get("temporal_fallback")
    if isinstance(temporal_fallback, dict) and bool(temporal_fallback.get("enabled", False)):
        return True
    return bool(
        getattr(component, "consumes_reliability_metadata", False)
        or getattr(component, "supports_reliability_metadata", False)
        or getattr(component, "supports_observability_metadata", False)
        or getattr(component, "supports_missing_modality_metadata", False)
    )


def encoder_dependencies_satisfied(
    dependencies: tuple[str, ...],
    *,
    source: str,
    encoded: dict[str, torch.Tensor],
    projected: dict[str, torch.Tensor],
) -> bool:
    if source == "raw":
        return True
    if source == "none":
        return not dependencies
    if source == "encoded":
        return all(dependency in encoded for dependency in dependencies)
    if source == "projected":
        return all(dependency in projected for dependency in dependencies)
    return False


def unmet_context_dependencies(
    dependencies: tuple[str, ...],
    *,
    source: str,
    encoded: dict[str, torch.Tensor],
    projected: dict[str, torch.Tensor],
) -> list[str]:
    if source == "raw":
        return []
    if source == "none":
        return [] if not dependencies else list(dependencies)
    if source == "encoded":
        return [dependency for dependency in dependencies if dependency not in encoded]
    if source == "projected":
        return [dependency for dependency in dependencies if dependency not in projected]
    return list(dependencies)


def encoder_context_kwargs(
    encoder: nn.Module,
    *,
    modality: str,
    raw_tensor: torch.Tensor,
    dependencies: tuple[str, ...],
    raw_inputs: dict[str, torch.Tensor | None],
    encoded: dict[str, torch.Tensor],
    projected: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not dependencies:
        return {}
    source = encoder_context_source(encoder)
    kwarg_names = getattr(encoder, "context_feature_kwargs", {})
    if not isinstance(kwarg_names, dict):
        kwarg_names = {}
    context_kwargs: dict[str, torch.Tensor] = {}
    for dependency in dependencies:
        if source == "projected":
            feature = projected[dependency]
        elif source == "encoded":
            feature = encoded[dependency]
        elif source == "raw":
            feature = raw_inputs.get(dependency)
            if feature is None:
                raise ValueError(
                    f"Encoder for modality '{modality}' requested raw condition feature from '{dependency}', "
                    "but that raw batch input is missing."
                )
        else:
            raise ValueError(
                f"Encoder for modality '{modality}' requested unsupported condition feature source "
                f"{source!r}; supported sources are 'projected', 'encoded', and 'raw'."
            )
        check_condition_feature_shape(
            modality=modality,
            dependency=dependency,
            raw_tensor=raw_tensor,
            condition_features=feature,
            source=source,
        )
        kwarg = str(kwarg_names.get(dependency, f"{dependency}_condition_features"))
        context_kwargs[kwarg] = feature
    return context_kwargs


def encoder_reliability_kwargs(
    encoder: nn.Module,
    *,
    modality: str,
    reliability_inputs: dict[str, Any],
) -> dict[str, Any]:
    if modality != "image":
        return {}
    if not bool(getattr(encoder, "supports_observability_metadata", False)):
        return {}
    return {key: value for key, value in reliability_inputs.items() if value is not None}


def check_condition_feature_shape(
    *,
    modality: str,
    dependency: str,
    raw_tensor: torch.Tensor,
    condition_features: torch.Tensor,
    source: str,
) -> None:
    if source != "raw" and condition_features.ndim != 3:
        raise ValueError(
            f"Condition feature for modality '{dependency}' must have shape [B, T, D], "
            f"got {tuple(condition_features.shape)} while encoding '{modality}'."
        )
    if source == "raw" and condition_features.ndim < 2:
        raise ValueError(
            f"Raw condition feature for modality '{dependency}' must expose batch/time dimensions, "
            f"got {tuple(condition_features.shape)} while encoding '{modality}'."
        )
    if raw_tensor.ndim < 2:
        raise ValueError(
            f"Modular sequence input for modality '{modality}' must expose batch/time dimensions, "
            f"got {tuple(raw_tensor.shape)}."
        )
    raw_batch_time = tuple(int(value) for value in raw_tensor.shape[:2])
    condition_batch_time = tuple(int(value) for value in condition_features.shape[:2])
    if raw_batch_time != condition_batch_time:
        raise ValueError(
            "Condition feature batch/time dimensions must match the conditioned encoder input; "
            f"modality '{modality}' input shape {tuple(raw_tensor.shape)}, "
            f"condition modality '{dependency}' feature shape {tuple(condition_features.shape)}."
        )


def core_input_availability_mask(
    projected: dict[str, torch.Tensor],
    modalities: tuple[str, ...],
    *,
    valid_masks: dict[str, torch.Tensor | None],
    dropout_masks: dict[str, torch.Tensor | None],
    availability_overrides: dict[str, torch.Tensor | None],
    token_features: bool,
) -> torch.Tensor | None:
    pieces: list[torch.Tensor] = []
    for modality in modalities:
        features = projected[modality]
        mask = modality_availability_from_inputs(
            modality,
            features,
            valid_mask=valid_masks.get(modality),
            dropout_mask=dropout_masks.get(modality),
            availability_override=availability_overrides.get(modality),
        )
        if token_features:
            token_count = int(features.shape[2]) if features.ndim == 4 else 1
            mask = mask.unsqueeze(2).expand(-1, -1, token_count)
        pieces.append(mask)
    if not pieces:
        return None
    if token_features:
        return torch.cat(pieces, dim=2).permute(0, 2, 1).contiguous()
    return torch.stack(pieces, dim=1)


def pad_modality_tokens(features: torch.Tensor, *, max_tokens: int) -> torch.Tensor:
    token_count = int(features.shape[2])
    if token_count == int(max_tokens):
        return features
    pad_shape = (*tuple(features.shape[:2]), int(max_tokens) - token_count, int(features.shape[-1]))
    pad = torch.zeros(pad_shape, dtype=features.dtype, device=features.device)
    return torch.cat([features, pad], dim=2)


def pad_modality_token_mask(mask: torch.Tensor, *, max_tokens: int) -> torch.Tensor:
    token_count = int(mask.shape[2])
    if token_count == int(max_tokens):
        return mask
    pad_shape = (*tuple(mask.shape[:2]), int(max_tokens) - token_count)
    pad = torch.zeros(pad_shape, dtype=torch.bool, device=mask.device)
    return torch.cat([mask, pad], dim=2)


def modality_availability_from_inputs(
    modality: str,
    features: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None,
    dropout_mask: torch.Tensor | None,
    availability_override: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, seq_len = int(features.shape[0]), int(features.shape[1])
    if valid_mask is not None:
        mask = coerce_temporal_mask(
            valid_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=features.device,
            name=f"{modality}_valid_mask",
        )
    elif dropout_mask is not None:
        mask = ~coerce_temporal_mask(
            dropout_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=features.device,
            name=f"{modality}_dropout_mask",
        )
    else:
        mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=features.device)
    if availability_override is not None:
        override = coerce_temporal_mask(
            availability_override,
            batch_size=batch_size,
            seq_len=seq_len,
            device=features.device,
            name=f"{modality}_availability_override",
        )
        mask = mask & override
    return mask


def coerce_temporal_mask(
    mask: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    value = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [B, T] or [B], got {tuple(value.shape)}.")
    if int(value.shape[0]) != int(batch_size):
        raise ValueError(f"{name} batch size must be {batch_size}, got {tuple(value.shape)}.")
    if int(value.shape[1]) == int(seq_len):
        return value
    if int(value.shape[1]) == 1:
        return value.expand(-1, int(seq_len))
    raise ValueError(f"{name} time dimension must be 1 or {seq_len}, got {tuple(value.shape)}.")


def coerce_core_availability_mask(
    mask: torch.Tensor,
    *,
    features: torch.Tensor,
    core_name: str,
) -> torch.Tensor:
    value = torch.as_tensor(mask, dtype=torch.bool, device=features.device)
    expected = tuple(int(item) for item in features.shape[:3])
    if value.ndim != 3 or tuple(int(item) for item in value.shape) != expected:
        raise ValueError(f"{core_name} modality_available must have shape {expected}, got {tuple(value.shape)}.")
    return value


def missing_modality_output_metadata(
    availability_mask: torch.Tensor | None,
    *,
    modalities: tuple[str, ...],
) -> dict[str, Any]:
    if availability_mask is None:
        return {"available": True, "modalities": list(modalities), "missing_counts": {}}
    available = availability_mask.detach()
    if available.ndim == 4:
        available = available.any(dim=3)
    missing = ~available
    counts: dict[str, int] = {}
    for index, modality in enumerate(modalities):
        if index >= int(missing.shape[1]):
            break
        counts[modality] = int(missing[:, index, :].sum().cpu().item())
    return {
        "available": True,
        "modalities": list(modalities),
        "availability_mask": availability_mask,
        "missing_counts": counts,
        "provenance": "input_valid_or_dropout_masks",
    }


def check_temporal_features(
    features: torch.Tensor,
    modality: str,
    batch_size: int | None,
    seq_len: int | None,
) -> tuple[int, int]:
    if features.ndim not in {3, 4}:
        raise ValueError(
            f"{modality} encoder output must have shape [B, T, D] or [B, T, K, D], got {tuple(features.shape)}."
        )
    current_batch, current_seq = int(features.shape[0]), int(features.shape[1])
    if batch_size is not None and (current_batch != batch_size or current_seq != seq_len):
        raise ValueError(
            "Modular sequence modalities must share batch/time dimensions; "
            f"modality '{modality}' produced shape {tuple(features.shape)}, "
            f"expected batch={batch_size}, time={seq_len}."
        )
    return current_batch, current_seq


def check_projected_features(features: torch.Tensor, modality: str, d_model: int) -> None:
    if features.ndim not in {3, 4} or int(features.shape[-1]) != int(d_model):
        raise ValueError(
            f"{modality} projector output must have shape [B, T, {int(d_model)}] or "
            f"[B, T, K, {int(d_model)}], got {tuple(features.shape)}."
        )
