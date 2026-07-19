from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.data.mmw.twc_router_joint_stress import STATE_CORRUPT, STATE_DROP
from kd_sensing.data.mmw.twc_router_joint_training import load_router_joint_training_panel
from kd_sensing.data.missing_mask import sample_missing_mask
from kd_sensing.data.temporal_missing import apply_modality_temporal_mask_to_batch
from kd_sensing.data.temporal_missing_contract import TEMPORAL_SUPERSET_PAYLOAD_KEY
from kd_sensing.engine.evaluation_pass_runtime import sample_ids_from_batch
from kd_sensing.engine.training_extensions import BaseLossResult, BatchState, ExtensionContext, ForwardControls, TrainingExtension
from kd_sensing.losses.modality_alignment_contrastive import amber_cma_analogue_loss
from kd_sensing.losses.router_reliability import paired_router_reliability_loss
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.losses.u_mask_beam_jepa_prototype import add_prototype_alignment_losses


def u_mask_beam_jepa_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    prototype_bank: torch.nn.Module | None = None,
    use_beam_prototype_alignment: bool = False,
    lambda_proto: float = 0.0,
    lambda_modality_proto: float = 0.0,
    beam_label_sigma: float = 1.0,
    prototype_target_circular: bool = True,
    prototype_topology_id: str | None = None,
    prototype_topology_permutation: list[int] | tuple[int, ...] | None = None,
    use_amber_cma_analogue: bool = False,
    lambda_amber_cma: float = 0.2,
    amber_cma_temperature: float = 0.2,
    sample_ids: list[str] | tuple[str, ...] | None = None,
    superset_output: dict[str, torch.Tensor] | None = None,
    use_superset_confidence_gated_kl: bool = False,
    lambda_superset_consistency: float = 0.0,
    superset_temperature: float = 2.0,
    router_oracle_weight: float = 0.1,
    router_oracle_target_mode: str = "hard_first",
    router_oracle_temperature: float = 1.0,
    router_oracle_beam_temperature: float = 1.0,
    circular_beam_distance: bool = True,
    beam_powers: torch.Tensor | None = None,
) -> dict[str, Any]:
    if use_beam_prototype_alignment and use_amber_cma_analogue:
        raise ValueError("BPA and the AMBER CMA analogue are mutually exclusive.")
    if float(lambda_amber_cma) < 0.0 or float(lambda_superset_consistency) < 0.0:
        raise ValueError("auxiliary loss weights must be non-negative.")
    if float(amber_cma_temperature) <= 0.0 or float(superset_temperature) <= 0.0:
        raise ValueError("auxiliary loss temperatures must be positive.")

    logits = _as_prediction_logits(output["logits"])
    labels = _as_prediction_labels(labels, logits)
    loss_beam = _beam_supervised_loss(logits, labels)
    loss = loss_beam

    loss, prototype_diagnostics = add_prototype_alignment_losses(
        loss,
        output,
        labels,
        prototype_bank=prototype_bank,
        enabled=use_beam_prototype_alignment,
        lambda_proto=lambda_proto,
        lambda_modality_proto=lambda_modality_proto,
        beam_label_sigma=beam_label_sigma,
        prototype_target_circular=prototype_target_circular,
        prototype_topology_id=prototype_topology_id,
        prototype_topology_permutation=prototype_topology_permutation,
    )

    zero = logits.sum() * 0.0
    loss_amber_cma = zero
    diagnostics: dict[str, float] = dict(prototype_diagnostics)
    if use_amber_cma_analogue:
        loss_amber_cma, cma_diagnostics = amber_cma_analogue_loss(
            output["output_features"],
            output["modality_features"],
            output["missing_mask"],
            sample_ids,
            temperature=float(amber_cma_temperature),
        )
        weighted = float(lambda_amber_cma) * loss_amber_cma
        loss = loss + weighted
        diagnostics.update(cma_diagnostics)
        diagnostics["loss/amber_cma_weighted"] = float(weighted.detach().cpu().item())

    loss_superset = zero
    if use_superset_confidence_gated_kl:
        if superset_output is None or not torch.is_tensor(superset_output.get("logits")):
            raise ValueError("Enabled superset KL requires a same-model superset output.")
        loss_superset, raw_kl, gate = _confidence_gated_temperature_kl(
            logits,
            superset_output["logits"].detach(),
            labels,
            temperature=float(superset_temperature),
        )
        loss = loss + float(lambda_superset_consistency) * loss_superset
        diagnostics.update(
            {
                "loss/superset_consistency": float(loss_superset.detach().cpu().item()),
                "superset_consistency/raw_kl": float(raw_kl.detach().cpu().item()),
                "superset_consistency/weighted_kl": float(loss_superset.detach().cpu().item()),
                "superset_consistency/gate_mean": float(gate.mean().detach().cpu().item()),
                "superset_consistency/gate_active_ratio": float(gate.gt(0).float().mean().detach().cpu().item()),
            }
        )

    if float(router_oracle_weight) == 0.0:
        loss_router_oracle = zero
        router_diagnostics = {
            "loss/router_oracle": 0.0,
            "router_oracle_active_ratio": 0.0,
            "router_oracle_tie_ratio": 0.0,
            "router_oracle_target_entropy": 0.0,
            "router_oracle_first_modality_target_mass": 0.0,
            "router_oracle_enabled": 0.0,
            "router_oracle_disabled": 1.0,
        }
    else:
        loss_router_oracle, router_diagnostics = _router_oracle_loss(
            output,
            labels,
            circular_beam_distance=bool(circular_beam_distance),
            target_mode=str(router_oracle_target_mode),
            temperature=float(router_oracle_temperature),
            beam_temperature=float(router_oracle_beam_temperature),
            beam_powers=beam_powers,
        )
        router_diagnostics["router_oracle_enabled"] = 1.0
        router_diagnostics["router_oracle_disabled"] = 0.0
    loss = loss + float(router_oracle_weight) * loss_router_oracle
    diagnostics.update(router_diagnostics)
    diagnostics["loss/router_oracle_weighted"] = float(
        (float(router_oracle_weight) * loss_router_oracle).detach().cpu().item()
    )

    diagnostics.update(
        _loss_diagnostics(
            logits,
            labels,
            loss_beam,
        )
    )
    return {
        "loss": loss,
        "loss_beam": loss_beam,
        "loss_amber_cma": loss_amber_cma,
        "loss_superset": loss_superset,
        "loss_router_oracle": loss_router_oracle,
        "diagnostics": diagnostics,
    }


class UMaskBeamJEPATrainingExtension(TrainingExtension):
    name = "u_mask_beam_jepa"

    def setup(self, context: ExtensionContext) -> dict[str, Any]:
        config = u_mask_beam_jepa_config(context.cfg)
        return {
            "config": config,
            "online_superset": None,
            "router_quality_pair": None,
            "router_quality_pair_scheduled": False,
            "dynamic_router_pair": None,
            "dynamic_router_panel": _load_dynamic_router_panel(config),
        }

    def state_dict(self, state: Any) -> dict[str, Any]:
        config = state.get("config", {}) if isinstance(state, dict) else {}
        return {"config": config}

    def load_state_dict(self, state: Any, payload: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("u_mask_beam_jepa extension state must be a mapping.")
        state.clear()
        state.update(
            {
                "config": dict(payload.get("config", {})),
                "online_superset": None,
                "router_quality_pair": None,
                "router_quality_pair_scheduled": False,
                "dynamic_router_pair": None,
                "dynamic_router_panel": _load_dynamic_router_panel(dict(payload.get("config", {}))),
            }
        )

    def before_forward(
        self,
        context: ExtensionContext,
        state: Any,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        epoch: int,
        step: int = 0,
    ) -> ForwardControls:
        config = state.get("config", {}) if isinstance(state, dict) else {}
        if not bool(config.get("enabled", False)):
            return ForwardControls()
        modalities = tuple(getattr(context.primary_model, "modalities", ()))
        if not modalities:
            raise ValueError("u_mask_beam_jepa requires primary_model.modalities.")
        mask_config = config.get("missing_mask", {})
        if mask_config.get("mode", "random") == "external":
            mask = _external_missing_mask(batch, labels, modalities, context.device)
        else:
            mask = sample_missing_mask(
                int(labels.shape[0]),
                len(modalities),
                mask_config.get("p_missing", 0.25),
                always_available_indices=mask_config.get("always_available_indices"),
                ensure_at_least_one=bool(mask_config.get("ensure_at_least_one", True)),
                device=context.device,
            )
        superset = config.get("superset_consistency", {})
        pairing = config.get("router_quality_pairing", {})
        dynamic = config.get("dynamic_router", {})
        dynamic_pairing = dynamic.get("paired_joint", {}) if isinstance(dynamic, dict) else {}
        superset_active = bool(
            isinstance(superset, dict) and superset.get("enabled") and superset.get("confidence_gated_kl")
        )
        pairing_active = bool(
            isinstance(pairing, dict)
            and pairing.get("enabled")
            and int(epoch) >= int(pairing.get("start_epoch_index", 0))
        )
        state["router_quality_pair_scheduled"] = pairing_active
        dynamic_pairing_active = bool(
            isinstance(dynamic, dict)
            and dynamic.get("enabled")
            and isinstance(dynamic_pairing, dict)
            and dynamic_pairing.get("enabled")
        )
        if superset_active or pairing_active:
            state["online_superset"] = _online_superset(context, batch, modalities)
        else:
            state["online_superset"] = None
        if pairing_active:
            state["router_quality_pair"] = _online_router_quality_pair(
                context,
                batch,
                modalities,
                state["online_superset"],
                epoch=epoch,
                step=step,
                config=pairing,
            )
        else:
            state["router_quality_pair"] = None
        if dynamic_pairing_active:
            state["dynamic_router_pair"] = _online_dynamic_router_pair(
                context,
                batch,
                modalities,
                state.get("dynamic_router_panel"),
                epoch=epoch,
                step=step,
                config=dynamic_pairing,
            )
        else:
            state["dynamic_router_pair"] = None
        return ForwardControls(model_kwargs={"missing_mask": mask})

    def compute_base_loss(self, context: ExtensionContext, state: Any, batch_state: BatchState) -> BaseLossResult | None:
        config = state.get("config", {}) if isinstance(state, dict) else {}
        if not bool(config.get("enabled", False)):
            return None
        output = {
            "logits": batch_state.primary_logits,
            "output_features": batch_state.primary_output.output_features,
            "input_features": batch_state.primary_output.input_features,
            **batch_state.primary_output.diagnostics,
        }
        superset = config.get("superset_consistency", {})
        beam_powers = batch_state.batch.get("future_beam_power")
        result = u_mask_beam_jepa_loss(
            output,
            batch_state.labels,
            prototype_bank=getattr(context.primary_model, "prototype_bank", None),
            use_beam_prototype_alignment=bool(config.get("use_beam_prototype_alignment", False)),
            lambda_proto=float(config.get("lambda_proto", 0.0)),
            lambda_modality_proto=float(config.get("lambda_modality_proto", 0.0)),
            beam_label_sigma=float(config.get("beam_label_sigma", 1.0)),
            prototype_target_circular=bool(config.get("prototype_target_circular", True)),
            prototype_topology_id=str(config.get("prototype_topology", {}).get("id", "")) or None,
            prototype_topology_permutation=config.get("prototype_topology", {}).get("permutation"),
            use_amber_cma_analogue=bool(config.get("use_amber_cma_analogue", False)),
            lambda_amber_cma=float(config.get("lambda_amber_cma", 0.2)),
            amber_cma_temperature=float(config.get("amber_cma_temperature", 0.2)),
            sample_ids=(sample_ids_from_batch(batch_state.batch) if config.get("use_amber_cma_analogue", False) else None),
            superset_output=state.get("online_superset"),
            use_superset_confidence_gated_kl=bool(
                isinstance(superset, dict) and superset.get("enabled") and superset.get("confidence_gated_kl")
            ),
            lambda_superset_consistency=float(superset.get("kl_weight", 0.0)),
            superset_temperature=float(superset.get("temperature", 2.0)),
            router_oracle_weight=float(config.get("router_oracle_weight", 0.1)),
            router_oracle_target_mode=str(config.get("router_oracle_target_mode", "hard_first")),
            router_oracle_temperature=float(config.get("router_oracle_temperature", 1.0)),
            router_oracle_beam_temperature=float(config.get("router_oracle_beam_temperature", 1.0)),
            circular_beam_distance=bool(config.get("circular_beam_distance", True)),
            beam_powers=beam_powers,
        )
        total = result["loss"]
        pairing = config.get("router_quality_pairing", {})
        if bool(state.get("router_quality_pair_scheduled")):
            paired_loss, paired_diagnostics = _paired_router_quality_loss(
                state.get("router_quality_pair"),
                batch_state.labels,
                beam_powers,
                temperature=float(pairing.get("target_temperature", 0.1)),
                utility_mode=str(pairing.get("utility_mode", "argmax")),
                beam_temperature=float(pairing.get("beam_temperature", 1.0)),
                max_target_entropy=pairing.get("max_target_entropy"),
                utility_weight=float(pairing.get("utility_weight", 0.0)),
                monotonic_weight=float(pairing.get("monotonic_weight", 0.0)),
                margin_scale=float(pairing.get("monotonic_margin_scale", 0.25)),
                quality_drop_epsilon=float(pairing.get("quality_drop_epsilon", 0.01)),
            )
            total = total + paired_loss
            result["diagnostics"].update(paired_diagnostics)
            result["diagnostics"]["router_pair_scheduled_active"] = 1.0
        elif isinstance(pairing, dict) and pairing.get("enabled"):
            result["diagnostics"].update(
                {
                    "loss/router_pair_utility": 0.0,
                    "loss/router_pair_utility_weighted": 0.0,
                    "loss/router_pair_monotonic": 0.0,
                    "loss/router_pair_monotonic_weighted": 0.0,
                    "router_pair_active_ratio": 0.0,
                    "router_pair_scheduled_active": 0.0,
                }
            )
        dynamic = config.get("dynamic_router", {})
        if isinstance(dynamic, dict) and dynamic.get("enabled"):
            dynamic_pair = state.get("dynamic_router_pair")
            if not isinstance(dynamic_pair, dict):
                raise ValueError("Enabled dynamic Router requires a same-availability Joint pair.")
            topology = config.get("prototype_topology", {})
            paired = dynamic.get("paired_joint", {})
            dynamic_loss, dynamic_diagnostics = paired_router_reliability_loss(
                dynamic_pair["control"],
                dynamic_pair["joint"],
                batch_state.labels,
                source=str(dynamic.get("supervision", "label_topology")),
                beam_powers=beam_powers,
                beam_temperature=float(dynamic.get("utility_temperature", 1.0)),
                beam_label_sigma=float(config.get("beam_label_sigma", 1.0)),
                circular=bool(config.get("prototype_target_circular", True)),
                topology_id=str(topology.get("id", "")) or None,
                topology_permutation=topology.get("permutation"),
                quality_weight=float(dynamic.get("quality_regression_weight", 0.1)),
                fused_utility_weight=float(dynamic.get("fused_utility_weight", 0.1)),
                monotonic_weight=float(paired.get("monotonic_weight", 0.05)),
                frame_rank_weight=float(dynamic.get("frame_rank_weight", 0.0)),
                residual_anchor_weight=float(dynamic.get("residual_anchor_weight", 0.0)),
                quality_drop_epsilon=float(paired.get("quality_drop_epsilon", 0.01)),
                monotonic_margin_scale=float(paired.get("monotonic_margin_scale", 0.25)),
            )
            total = total + dynamic_loss
            result["diagnostics"].update(dynamic_diagnostics)
        return BaseLossResult(
            total_loss=total,
            task_loss=result["loss_beam"],
            auxiliary_loss=total - result["loss_beam"],
            diagnostics=result["diagnostics"],
        )


def _load_dynamic_router_panel(config: dict[str, Any]) -> dict[str, Any] | None:
    dynamic = config.get("dynamic_router", {})
    paired = dynamic.get("paired_joint", {}) if isinstance(dynamic, dict) else {}
    if not (isinstance(dynamic, dict) and dynamic.get("enabled") and isinstance(paired, dict) and paired.get("enabled")):
        return None
    panel = load_router_joint_training_panel(str(paired.get("panel_path", "")))
    expected = str(paired.get("panel_sha256", "")).strip().lower()
    if str(panel.get("checksum", "")) != expected:
        raise ValueError(
            "Dynamic Router Joint panel checksum mismatch: "
            f"config={expected!r}, panel={panel.get('checksum')!r}."
        )
    return panel


def _online_dynamic_router_pair(
    context: ExtensionContext,
    batch: dict[str, Any],
    modalities: tuple[str, ...],
    panel: Any,
    *,
    epoch: int,
    step: int,
    config: dict[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    from kd_sensing.engine.runtime import run_model_step
    from kd_sensing.evaluation.corruptions import CorruptionSpec, apply_inference_corruption

    if not isinstance(panel, dict) or not isinstance(panel.get("conditions"), list):
        raise ValueError("Dynamic Router pairing requires a validated Joint panel.")
    restored, _ = _restore_temporal_superset(batch, modalities, context.device)
    base_mask = restored["modality_temporal_mask"].detach().to(device="cpu", dtype=torch.bool)
    batch_size = int(base_mask.shape[0])
    loader_cfg = context.cfg.get("data", {}).get("dataloader", {})
    configured_batch = int(loader_cfg.get("train_batch_size", batch_size))
    sampling_cfg = context.cfg.get("data", {}).get("domain_balanced_sampling", {})
    samples_per_epoch = int(sampling_cfg.get("num_samples", configured_batch))
    offset = int(epoch) * samples_per_epoch + int(step) * configured_batch
    conditions = panel["conditions"]
    selected = [conditions[(offset + index) % len(conditions)] for index in range(batch_size)]
    states = torch.as_tensor([item["state_matrix"] for item in selected], dtype=torch.int64)
    severities = torch.as_tensor([item["severity_matrix"] for item in selected], dtype=torch.int64)
    drop_mask = base_mask & states.ne(STATE_DROP)
    if not bool(drop_mask.any(dim=(1, 2)).all().item()):
        raise ValueError("Dynamic Router Joint panel removed every temporal cell from a sample.")

    control_batch = _clone_router_batch(restored)
    joint_batch = _clone_router_batch(restored)
    corruption_names = {
        "image": "image_occlusion",
        "radar": "radar_noise",
        "gps": "gps_noise",
        "lidar": "lidar_sparsify",
    }
    for modality_index, modality in enumerate(modalities):
        for severity in (1, 2, 3):
            selector = states[:, :, modality_index].eq(STATE_CORRUPT) & severities[:, :, modality_index].eq(severity)
            if not bool(selector.any().item()):
                continue
            apply_inference_corruption(
                joint_batch,
                CorruptionSpec(corruption_names[modality], severity),
                seed=int(config.get("corruption_seed", 20260719)) + modality_index * 1009 + severity * 101,
                batch_index=offset,
                gps_scaler_mean=batch.get("gps_scaler_mean"),
                gps_scaler_scale=batch.get("gps_scaler_scale"),
                selector=selector,
            )
    apply_modality_temporal_mask_to_batch(control_batch, drop_mask, modalities=modalities)
    apply_modality_temporal_mask_to_batch(joint_batch, drop_mask, modalities=modalities)
    if not torch.equal(
        control_batch["modality_temporal_mask"], joint_batch["modality_temporal_mask"]
    ):
        raise RuntimeError("Dynamic Router control and Joint views lost availability alignment.")

    module_states = [(module, module.training) for module in context.primary_model.modules()]
    availability = drop_mask.any(dim=1).to(device=context.device)
    try:
        context.primary_model.eval()
        with torch.no_grad():
            control_step = run_model_step(
                context.primary_model,
                context.task,
                control_batch,
                seq_length=context.seq_length,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
                extra_model_kwargs={"missing_mask": availability, "return_router_state": True},
            )
            joint_step = run_model_step(
                context.primary_model,
                context.task,
                joint_batch,
                seq_length=context.seq_length,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
                extra_model_kwargs={"missing_mask": availability, "return_router_state": True},
            )
    finally:
        for module, training in module_states:
            module.training = training

    control_state = control_step.model_output.diagnostics.get("candidate_router_state")
    joint_state = joint_step.model_output.diagnostics.get("candidate_router_state")
    if not isinstance(control_state, dict) or not isinstance(joint_state, dict):
        raise ValueError("Dynamic Router candidate forward did not return rerouting state.")
    model = getattr(context.primary_model, "module", context.primary_model)
    with torch.no_grad():
        control_route = model.route_from_candidate_state(control_state)
    joint_route = model.route_from_candidate_state(joint_state)
    return {"control": control_route, "joint": joint_route}


def _clone_router_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def _online_superset(
    context: ExtensionContext,
    batch: dict[str, torch.Tensor],
    modalities: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    from kd_sensing.engine.runtime import run_model_step

    superset_batch, mask = _restore_temporal_superset(batch, modalities, context.device)
    module_states = [(module, module.training) for module in context.primary_model.modules()]
    try:
        context.primary_model.eval()
        with torch.no_grad():
            step = run_model_step(
                context.primary_model,
                context.task,
                superset_batch,
                seq_length=context.seq_length,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
                extra_model_kwargs={"missing_mask": mask},
            )
    finally:
        for module, training in module_states:
            module.training = training
    result = {"logits": step.logits.detach()}
    model_output = getattr(step, "model_output", None)
    diagnostics = getattr(model_output, "diagnostics", None)
    if isinstance(diagnostics, dict):
        result.update(
            {
                "router_features": diagnostics["supervised_router_reliability_features"].detach(),
                "unimodal_logits": diagnostics["unimodal_logits"].detach(),
                "missing_mask": diagnostics["missing_mask"].detach(),
            }
        )
    return result


def _online_router_quality_pair(
    context: ExtensionContext,
    batch: dict[str, Any],
    modalities: tuple[str, ...],
    clean: dict[str, torch.Tensor],
    *,
    epoch: int,
    step: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    from kd_sensing.engine.runtime import run_model_step
    from kd_sensing.evaluation.corruptions import ORACLE_GAP_CORRUPTION_GRID, apply_inference_corruption

    required = {"router_features", "unimodal_logits", "missing_mask"}
    if not required.issubset(clean):
        raise ValueError("Router quality pairing requires superset Router diagnostics.")
    spec = ORACLE_GAP_CORRUPTION_GRID[(int(epoch) + int(step)) % len(ORACLE_GAP_CORRUPTION_GRID)]
    corrupted_batch, mask = _restore_temporal_superset(batch, modalities, context.device)
    corrupted_batch = dict(corrupted_batch)
    apply_inference_corruption(
        corrupted_batch,
        spec,
        seed=int(config.get("corruption_seed", 20260719)),
        batch_index=int(epoch) * 1_000_000 + int(step),
        gps_scaler_mean=batch.get("gps_scaler_mean"),
        gps_scaler_scale=batch.get("gps_scaler_scale"),
    )
    module_states = [(module, module.training) for module in context.primary_model.modules()]
    try:
        context.primary_model.eval()
        with torch.no_grad():
            corrupted_step = run_model_step(
                context.primary_model,
                context.task,
                corrupted_batch,
                seq_length=context.seq_length,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
                extra_model_kwargs={"missing_mask": mask},
            )
    finally:
        for module, training in module_states:
            module.training = training
    corrupted = corrupted_step.model_output.diagnostics
    model = getattr(context.primary_model, "module", context.primary_model)
    router_states = [(module, module.training) for module in model.supervised_router.modules()]
    try:
        model.supervised_router.eval()
        combined_features = torch.cat(
            (clean["router_features"].detach(), corrupted["supervised_router_reliability_features"].detach()), dim=0
        )
        combined_mask = torch.cat((clean["missing_mask"], corrupted["missing_mask"].detach()), dim=0)
        combined_logits, combined_weights = model.route_from_features(combined_features, combined_mask)
    finally:
        for module, training in router_states:
            module.training = training
    count = clean["router_features"].shape[0]
    affected_name = spec.name.split("_", 1)[0]
    return {
        "clean_unimodal_logits": clean["unimodal_logits"],
        "corrupted_unimodal_logits": corrupted["unimodal_logits"].detach(),
        "available": clean["missing_mask"],
        "clean_router_logits": combined_logits[:count],
        "corrupted_router_logits": combined_logits[count:],
        "clean_router_weights": combined_weights[:count],
        "corrupted_router_weights": combined_weights[count:],
        "affected_modality_index": modalities.index(affected_name),
        "corruption_name": spec.name,
        "corruption_severity": int(spec.severity),
    }


def _paired_router_quality_loss(
    pair: Any,
    labels: torch.Tensor,
    beam_powers: torch.Tensor | None,
    *,
    temperature: float,
    utility_mode: str,
    beam_temperature: float,
    max_target_entropy: float | None,
    utility_weight: float,
    monotonic_weight: float,
    margin_scale: float,
    quality_drop_epsilon: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not isinstance(pair, dict):
        raise ValueError("Enabled router_quality_pairing requires an online clean/corrupted pair.")
    corrupt_output = {
        "logits": pair["corrupted_router_logits"],
        "router_gate_logits": pair["corrupted_router_logits"],
        "unimodal_logits": pair["corrupted_unimodal_logits"],
        "missing_mask": pair["available"],
    }
    utility_target_mode = {
        "argmax": "beam_power_soft",
        "expected": "beam_power_expected_soft",
    }.get(utility_mode)
    if utility_target_mode is None:
        raise ValueError(f"Unsupported paired Router utility mode {utility_mode!r}.")
    utility_loss, utility_diagnostics = _router_oracle_loss(
        corrupt_output,
        labels,
        circular_beam_distance=True,
        target_mode=utility_target_mode,
        temperature=temperature,
        beam_temperature=beam_temperature,
        max_target_entropy=max_target_entropy,
        beam_powers=beam_powers,
    )
    clean_utility = _unimodal_normalized_utility(
        pair["clean_unimodal_logits"], beam_powers, mode=utility_mode, beam_temperature=beam_temperature
    )
    corrupt_utility = _unimodal_normalized_utility(
        pair["corrupted_unimodal_logits"], beam_powers, mode=utility_mode, beam_temperature=beam_temperature
    )
    affected = int(pair["affected_modality_index"])
    quality_drop = (clean_utility[:, affected] - corrupt_utility[:, affected]).detach()
    active = quality_drop.gt(float(quality_drop_epsilon))
    clean_weight = pair["clean_router_weights"][:, affected].detach()
    corrupt_weight = pair["corrupted_router_weights"][:, affected]
    margin = float(margin_scale) * quality_drop
    penalties = F.relu(corrupt_weight - clean_weight + margin)
    monotonic_loss = penalties[active].mean() if bool(active.any().item()) else penalties.sum() * 0.0
    weighted_utility = float(utility_weight) * utility_loss
    weighted_monotonic = float(monotonic_weight) * monotonic_loss
    total = weighted_utility + weighted_monotonic
    active_float = active.float()
    diagnostics = {
        "loss/router_pair_utility": float(utility_loss.detach().cpu().item()),
        "loss/router_pair_utility_weighted": float(weighted_utility.detach().cpu().item()),
        "loss/router_pair_monotonic": float(monotonic_loss.detach().cpu().item()),
        "loss/router_pair_monotonic_weighted": float(weighted_monotonic.detach().cpu().item()),
        "router_pair_quality_drop_mean": float(quality_drop.mean().cpu().item()),
        "router_pair_quality_drop_positive_ratio": float(quality_drop.gt(0).float().mean().cpu().item()),
        "router_pair_active_ratio": float(active_float.mean().cpu().item()),
        "router_pair_violation_ratio": float(((corrupt_weight.detach() > clean_weight) & active).float().mean().cpu().item()),
        "router_pair_clean_affected_weight": float(clean_weight.mean().cpu().item()),
        "router_pair_corrupted_affected_weight": float(corrupt_weight.detach().mean().cpu().item()),
        "router_pair_corruption_severity": float(pair["corruption_severity"]),
        "router_pair_target_entropy": float(utility_diagnostics.get("router_oracle_target_entropy", 0.0)),
        "router_pair_target_informative_ratio": float(
            utility_diagnostics.get("router_oracle_informative_ratio", 1.0)
        ),
    }
    return total, diagnostics


def _restore_temporal_superset(
    batch: dict[str, Any],
    modalities: tuple[str, ...],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    payload = batch.get(TEMPORAL_SUPERSET_PAYLOAD_KEY)
    if not isinstance(payload, dict):
        raise ValueError("Superset KL requires temporal_missing.preserve_unmasked_for_superset=true.")
    inputs = payload.get("inputs")
    base_mask = payload.get("base_mask")
    payload_modalities = tuple(payload.get("modalities", ()))
    if not isinstance(inputs, dict) or not torch.is_tensor(base_mask) or payload_modalities != modalities:
        raise ValueError("Invalid temporal superset payload.")
    base_mask = base_mask.to(device=device, dtype=torch.bool)
    student_mask = batch.get("modality_temporal_mask")
    if not torch.is_tensor(student_mask) or tuple(student_mask.shape) != tuple(base_mask.shape):
        raise ValueError("Temporal superset and student masks must have matching [B,T,M] shapes.")
    student_mask = student_mask.to(device=device, dtype=torch.bool)
    if bool((student_mask & ~base_mask).any().item()):
        raise ValueError("Temporal student mask must be a subset of the preserved superset mask.")
    if not bool(student_mask.any(dim=(1, 2)).all().item()) or not bool(base_mask.any(dim=(1, 2)).all().item()):
        raise ValueError("Temporal student and superset masks must retain one cell per sample.")
    restored = dict(batch)
    restored.update(inputs)
    restored["modality_temporal_mask"] = base_mask
    restored["temporal_mask"] = base_mask.any(dim=2)
    restored["available_modalities"] = base_mask.any(dim=1)
    return restored, base_mask.any(dim=1)


def _as_prediction_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits.unsqueeze(1)
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B,C] or [B,T,C], got {tuple(logits.shape)}.")
    return logits


def _as_prediction_labels(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=torch.long)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.ndim != 2 or labels.shape[0] != logits.shape[0] or labels.shape[1] < logits.shape[1]:
        raise ValueError(f"labels must cover logits shape {tuple(logits.shape[:2])}, got {tuple(labels.shape)}.")
    return labels[:, : logits.shape[1]]


def _beam_supervised_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def _external_missing_mask(
    batch: dict[str, Any],
    labels: torch.Tensor,
    modalities: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    available = batch.get("available_modalities")
    expected = (int(labels.shape[0]), len(modalities))
    if not torch.is_tensor(available):
        raise ValueError("external missing-mask mode requires batch.available_modalities.")
    if available.dtype != torch.bool or tuple(available.shape) != expected:
        raise ValueError(f"batch.available_modalities must be a bool tensor with shape {expected}.")
    mask = available.to(device=device).clone()
    if not bool(mask.any(dim=1).all().item()):
        raise ValueError("external missing-mask mode requires at least one available modality per sample.")
    return mask


def _confidence_gated_temperature_kl(
    student_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student = _as_prediction_logits(student_logits)
    reference = _as_prediction_logits(reference_logits).to(device=student.device, dtype=student.dtype).detach()
    labels = _as_prediction_labels(labels, student)
    if reference.shape != student.shape:
        raise ValueError(f"Superset logits shape {tuple(reference.shape)} must match student logits {tuple(student.shape)}.")
    confidence = torch.softmax(reference, dim=-1).amax(dim=-1)
    correct = reference.argmax(dim=-1).eq(labels)
    gate = confidence * correct.to(dtype=confidence.dtype)
    per_sample = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(reference / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1).mean(dim=1) * temperature**2
    weighted = (per_sample * gate.mean(dim=1)).sum() / gate.mean(dim=1).sum().clamp_min(1e-6)
    return weighted, per_sample.mean(), gate.mean(dim=1)


def _router_oracle_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    circular_beam_distance: bool,
    target_mode: str,
    temperature: float,
    beam_temperature: float = 1.0,
    max_target_entropy: float | None = None,
    beam_powers: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output.get("router_gate_logits")
    unimodal = output.get("unimodal_logits")
    available = output.get("missing_mask")
    if not torch.is_tensor(logits) or not torch.is_tensor(unimodal) or not torch.is_tensor(available):
        zero = output["logits"].sum() * 0.0
        return zero, {"router_oracle_active_ratio": 0.0, "loss/router_oracle": 0.0}
    if logits.ndim != 2 or unimodal.ndim != 3 or tuple(logits.shape) != tuple(unimodal.shape[:2]):
        raise ValueError("supervised_router outputs must be [B,M] gate logits and [B,M,C] unimodal logits.")
    available = available.to(device=logits.device, dtype=torch.bool)
    if tuple(available.shape) != tuple(logits.shape):
        raise ValueError("missing_mask must match supervised_router gate logits.")
    target = labels[:, 0].to(device=logits.device, dtype=torch.long)
    base_active = target.ne(-100) & available.sum(dim=1).gt(1)
    if not bool(base_active.any().item()):
        zero = logits.sum() * 0.0
        return zero, {"router_oracle_active_ratio": 0.0, "loss/router_oracle": 0.0}
    targets, tied = _router_oracle_targets(
        unimodal,
        target,
        available,
        circular_beam_distance=circular_beam_distance,
        target_mode=target_mode,
        temperature=temperature,
        beam_temperature=beam_temperature,
        beam_powers=beam_powers,
    )
    target_entropy = -(targets * targets.clamp_min(torch.finfo(targets.dtype).tiny).log()).sum(dim=1)
    if max_target_entropy is None:
        informative = torch.ones_like(base_active)
    else:
        if float(max_target_entropy) <= 0.0:
            raise ValueError("Router oracle max target entropy must be positive when set.")
        informative = target_entropy.le(float(max_target_entropy))
    active = base_active & informative
    informative_ratio = float(informative[base_active].float().mean().detach().cpu().item())
    if not bool(active.any().item()):
        zero = logits.sum() * 0.0
        return zero, {
            "loss/router_oracle": 0.0,
            "router_oracle_active_ratio": 0.0,
            "router_oracle_informative_ratio": informative_ratio,
            "router_oracle_target_entropy": float(target_entropy[base_active].mean().detach().cpu().item()),
        }
    masked = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
    if target_mode.startswith("hard_"):
        oracle = targets.argmax(dim=1)
        loss = F.cross_entropy(masked[active], oracle[active])
    else:
        loss = -(targets[active] * F.log_softmax(masked[active], dim=1)).sum(dim=1).mean()
    accuracy = masked[active].argmax(dim=1).eq(targets[active].argmax(dim=1)).float().mean()
    return loss, {
        "loss/router_oracle": float(loss.detach().cpu().item()),
        "router_oracle_accuracy": float(accuracy.detach().cpu().item()),
        "router_oracle_active_ratio": float(active.float().mean().detach().cpu().item()),
        "router_oracle_informative_ratio": informative_ratio,
        "router_oracle_tie_ratio": float(tied[active].float().mean().detach().cpu().item()),
        "router_oracle_target_entropy": float(target_entropy[base_active].mean().detach().cpu().item()),
        "router_oracle_first_modality_target_mass": float(targets[active, 0].mean().detach().cpu().item()),
    }


def _router_oracle_targets(
    unimodal: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
    *,
    circular_beam_distance: bool,
    target_mode: str,
    temperature: float,
    beam_temperature: float = 1.0,
    beam_powers: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_mode not in {
        "hard_first",
        "hard_confidence_tie",
        "soft_uniform_tie",
        "soft_confidence_tie",
        "distance_soft",
        "distance_confidence_soft",
        "beam_power_soft",
        "beam_power_expected_soft",
    }:
        raise ValueError(f"Unsupported router oracle target mode {target_mode!r}.")
    if float(temperature) <= 0.0:
        raise ValueError("router oracle temperature must be positive.")

    if target_mode in {"beam_power_soft", "beam_power_expected_soft"}:
        utilities = _unimodal_normalized_utility(
            unimodal,
            beam_powers,
            mode="expected" if target_mode == "beam_power_expected_soft" else "argmax",
            beam_temperature=beam_temperature,
        )
        scores = utilities / float(temperature)
        targets = torch.softmax(scores.masked_fill(~available, torch.finfo(scores.dtype).min), dim=1)
        maximum = utilities.masked_fill(~available, -1.0).amax(dim=1, keepdim=True)
        tied = (available & utilities.isclose(maximum, atol=1e-7, rtol=1e-6)).sum(dim=1).gt(1)
        return targets, tied

    safe_target = target.clamp_min(0)
    predicted = unimodal.detach().argmax(dim=-1)
    distance = (predicted - safe_target.unsqueeze(1)).abs()
    if circular_beam_distance:
        distance = torch.minimum(distance, int(unimodal.shape[-1]) - distance)
    unavailable_distance = int(unimodal.shape[-1]) + 1
    masked_distance = distance.masked_fill(~available, unavailable_distance)
    minimum = masked_distance.min(dim=1, keepdim=True).values
    ties = available & masked_distance.eq(minimum)
    tied = ties.sum(dim=1).gt(1)

    true_probability = torch.softmax(unimodal.detach(), dim=-1).gather(
        2,
        safe_target[:, None, None].expand(-1, unimodal.shape[1], 1),
    ).squeeze(-1)
    if target_mode == "hard_first":
        index = masked_distance.argmin(dim=1)
        return F.one_hot(index, num_classes=unimodal.shape[1]).to(dtype=unimodal.dtype), tied
    if target_mode == "hard_confidence_tie":
        index = true_probability.masked_fill(~ties, -1.0).argmax(dim=1)
        return F.one_hot(index, num_classes=unimodal.shape[1]).to(dtype=unimodal.dtype), tied
    if target_mode == "soft_uniform_tie":
        weights = ties.to(dtype=unimodal.dtype)
    elif target_mode == "soft_confidence_tie":
        weights = true_probability * ties.to(dtype=true_probability.dtype)
    else:
        scores = -masked_distance.to(dtype=unimodal.dtype) / float(temperature)
        if target_mode == "distance_confidence_soft":
            scores = scores + true_probability.clamp_min(torch.finfo(true_probability.dtype).tiny).log()
        return torch.softmax(scores.masked_fill(~available, torch.finfo(scores.dtype).min), dim=1), tied
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny), tied


def _unimodal_normalized_utility(
    unimodal: torch.Tensor,
    beam_powers: torch.Tensor | None,
    *,
    mode: str = "argmax",
    beam_temperature: float = 1.0,
) -> torch.Tensor:
    if not torch.is_tensor(beam_powers):
        raise ValueError("beam_power_soft Router targets require batch future_beam_power.")
    powers = beam_powers.to(device=unimodal.device, dtype=unimodal.dtype).detach()
    if powers.ndim != 2 or powers.shape != (unimodal.shape[0], unimodal.shape[2]):
        raise ValueError("future_beam_power must have shape [B,C] matching unimodal logits.")
    if not bool(torch.isfinite(powers).all()) or bool((powers < 0).any()):
        raise ValueError("future_beam_power must contain finite non-negative values.")
    normalized = powers / powers.amax(dim=1, keepdim=True).clamp_min(torch.finfo(powers.dtype).tiny)
    detached = unimodal.detach()
    if mode == "argmax":
        return normalized.gather(1, detached.argmax(dim=-1))
    if mode != "expected":
        raise ValueError(f"Unsupported normalized utility mode {mode!r}.")
    if float(beam_temperature) <= 0.0:
        raise ValueError("beam utility temperature must be positive.")
    probabilities = torch.softmax(detached / float(beam_temperature), dim=-1)
    return (probabilities * normalized.unsqueeze(1)).sum(dim=-1)


def _loss_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_beam: torch.Tensor,
) -> dict[str, float]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = flat_labels.ne(-100)
    if not bool(valid.any().item()):
        top1 = top5 = 0.0
    else:
        active_logits = flat_logits[valid]
        active_labels = flat_labels[valid]
        top1 = float(active_logits.argmax(dim=-1).eq(active_labels).float().mean().detach().cpu().item())
        top5 = float(
            active_logits.topk(min(5, active_logits.shape[-1]), dim=-1).indices.eq(active_labels.unsqueeze(-1)).any(dim=-1).float().mean().detach().cpu().item()
        )
    return {
        "loss_beam": float(loss_beam.detach().cpu().item()),
        "ce_loss": float(loss_beam.detach().cpu().item()),
        "accuracy/top1": top1,
        "accuracy/top5": top5,
    }


__all__ = ["UMaskBeamJEPATrainingExtension", "u_mask_beam_jepa_config", "u_mask_beam_jepa_loss"]
