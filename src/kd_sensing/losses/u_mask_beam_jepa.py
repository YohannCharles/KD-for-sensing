from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F

from kd_sensing.data.missing_mask import sample_missing_mask
from kd_sensing.data.temporal_missing_contract import TEMPORAL_SUPERSET_PAYLOAD_KEY
from kd_sensing.engine.evaluation_pass_runtime import sample_ids_from_batch
from kd_sensing.engine.training_extensions import (
    BaseLossResult,
    BatchState,
    ExtensionContext,
    ForwardControls,
    TrainingExtension,
)
from kd_sensing.losses.bcacl import bcacl_auxiliary_loss
from kd_sensing.losses.beam_prototype_alignment import prototype_alignment_loss_per_sample
from kd_sensing.losses.cmsbl import (
    NUM_NON_EMPTY_MASKS,
    accumulate_mask_losses,
    all_reduce_mask_statistics,
    capacity_gap_weights,
    effective_auxiliary_weights,
    fusion_mask_ids,
    hard_mask_weights,
    load_capacity_reference,
    mask_name,
    update_mask_loss_ema,
    update_metric_ema,
)
from kd_sensing.losses.modality_alignment_contrastive import amber_cma_analogue_loss
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.losses.u_mask_beam_jepa_prototype import add_prototype_alignment_losses
from kd_sensing.utils.cmsbl_diagnostics import write_cmsbl_epoch_diagnostics


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
    router_oracle_weight: float = 0.0,
    router_oracle_target_mode: str = "hard_first",
    router_oracle_temperature: float = 1.0,
    router_oracle_beam_temperature: float = 1.0,
    circular_beam_distance: bool = True,
    beam_powers: torch.Tensor | None = None,
    fusion_sample_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    del router_oracle_beam_temperature, circular_beam_distance, beam_powers
    if use_beam_prototype_alignment and use_amber_cma_analogue:
        raise ValueError("BPA and the AMBER CMA analogue are mutually exclusive.")
    if str(router_oracle_target_mode).strip().lower() != "hard_first":
        raise ValueError("Current T2 supports router_oracle_target_mode=hard_first only.")
    if min(float(amber_cma_temperature), float(superset_temperature), float(router_oracle_temperature)) <= 0:
        raise ValueError("loss temperatures must be positive.")
    if min(float(lambda_amber_cma), float(lambda_superset_consistency), float(router_oracle_weight)) < 0:
        raise ValueError("loss weights must be non-negative.")

    logits = _as_prediction_logits(output["logits"])
    targets = _as_prediction_labels(labels, logits)
    per_sample_beam = _beam_supervised_loss_per_sample(logits, targets)
    loss_beam = (
        per_sample_beam.mean()
        if fusion_sample_weights is None
        else _weighted_sample_mean(per_sample_beam, fusion_sample_weights)
    )
    loss = loss_beam
    per_sample_restoration: torch.Tensor | None = None
    if fusion_sample_weights is not None and use_beam_prototype_alignment and prototype_bank is not None:
        per_sample_restoration, prototype_diagnostics = prototype_alignment_loss_per_sample(
            prototype_bank,
            targets,
            fused_features=output["output_features"],
            modality_features=output.get("modality_features"),
            mask=output.get("missing_mask"),
            beam_label_sigma=beam_label_sigma,
            circular=prototype_target_circular,
            topology_id=prototype_topology_id,
            topology_permutation=prototype_topology_permutation,
            lambda_proto=lambda_proto,
            lambda_modality_proto=lambda_modality_proto,
        )
        loss = loss + _weighted_sample_mean(per_sample_restoration, fusion_sample_weights)
    else:
        loss, prototype_diagnostics = add_prototype_alignment_losses(
            loss,
            output,
            targets,
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
    loss_cma = zero
    diagnostics = dict(prototype_diagnostics)
    if use_amber_cma_analogue:
        loss_cma, cma_diagnostics = amber_cma_analogue_loss(
            output["output_features"],
            output["modality_features"],
            output["missing_mask"],
            sample_ids,
            temperature=float(amber_cma_temperature),
        )
        weighted_cma = float(lambda_amber_cma) * loss_cma
        loss = loss + weighted_cma
        diagnostics.update(cma_diagnostics)
        diagnostics["loss/amber_cma_weighted"] = _scalar(weighted_cma)

    loss_superset = zero
    if use_superset_confidence_gated_kl:
        if superset_output is None or not torch.is_tensor(superset_output.get("logits")):
            raise ValueError("Enabled superset KL requires a same-model superset output.")
        loss_superset, raw_kl, gate = _confidence_gated_temperature_kl(
            logits,
            superset_output["logits"].detach(),
            targets,
            temperature=float(superset_temperature),
        )
        loss = loss + float(lambda_superset_consistency) * loss_superset
        diagnostics.update(
            {
                "loss/superset_consistency": _scalar(loss_superset),
                "superset_consistency/raw_kl": _scalar(raw_kl),
                "superset_consistency/gate_mean": _scalar(gate.mean()),
                "superset_consistency/gate_active_ratio": _scalar(gate.gt(0).float().mean()),
            }
        )

    loss_router = zero
    if float(router_oracle_weight) > 0:
        loss_router, router_diagnostics = _router_oracle_loss(
            output,
            targets,
            temperature=float(router_oracle_temperature),
        )
        loss = loss + float(router_oracle_weight) * loss_router
    else:
        router_diagnostics = {
            "router_oracle_active_ratio": 0.0,
            "router_oracle_tie_ratio": 0.0,
            "router_oracle_enabled": 0.0,
            "router_oracle_disabled": 1.0,
        }
    diagnostics.update(router_diagnostics)
    diagnostics["loss/router_oracle"] = _scalar(loss_router)
    diagnostics["loss/router_oracle_weighted"] = _scalar(float(router_oracle_weight) * loss_router)
    diagnostics.update(_loss_diagnostics(logits, targets, loss_beam))

    per_sample_fusion = per_sample_beam
    if per_sample_restoration is not None:
        per_sample_fusion = per_sample_fusion + per_sample_restoration
    if fusion_sample_weights is not None:
        diagnostics["loss/fusion_raw_unweighted"] = _scalar(per_sample_fusion.mean())
        diagnostics["loss/fusion_weighted"] = _scalar(
            _weighted_sample_mean(per_sample_fusion, fusion_sample_weights)
        )
    return {
        "loss": loss,
        "loss_beam": loss_beam,
        "loss_amber_cma": loss_cma,
        "loss_superset": loss_superset,
        "loss_router_oracle": loss_router,
        "per_sample_fusion_loss": per_sample_fusion,
        "diagnostics": diagnostics,
    }


class UMaskBeamJEPATrainingExtension(TrainingExtension):
    name = "u_mask_beam_jepa"
    state_schema_version = 2

    def setup(self, context: ExtensionContext) -> dict[str, Any]:
        state: dict[str, Any] = {
            "config": u_mask_beam_jepa_config(context.cfg),
            "online_superset": None,
            "device": context.device,
        }
        _initialize_cmsbl_state(context, state)
        _reset_epoch_state(state)
        return state

    def state_dict(self, state: Any) -> dict[str, Any]:
        config = state.get("config", {})
        cmsbl = config.get("cmsbl", {})
        if not cmsbl.get("enabled"):
            return {}
        return {
            "cmsbl": {
                "capacity_identity": state["cmsbl_capacity_identity"],
                "capacity_reference": state["cmsbl_capacity_reference"].detach().cpu(),
                "metric_ema": state["cmsbl_metric_ema"].detach().cpu(),
                "metric_initialized": state["cmsbl_metric_initialized"].detach().cpu(),
                "mask_loss_ema": state["cmsbl_mask_loss_ema"].detach().cpu(),
                "mask_initialized": state["cmsbl_mask_initialized"].detach().cpu(),
                "mask_counts": state["cmsbl_mask_counts"].detach().cpu(),
            }
        }

    def load_state_dict(self, state: Any, payload: Mapping[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("u_mask_beam_jepa extension state must be a mapping.")
        cmsbl = state.get("config", {}).get("cmsbl", {})
        if not cmsbl.get("enabled"):
            if payload:
                raise ValueError("Disabled CMSBL cannot restore CMSBL state.")
            return
        saved = payload.get("cmsbl")
        if not isinstance(saved, Mapping):
            raise ValueError("CMSBL checkpoint state is missing.")
        if saved.get("capacity_identity") != state["cmsbl_capacity_identity"]:
            raise ValueError("CMSBL capacity reference identity changed across resume.")
        shapes = {
            "cmsbl_capacity_reference": ("capacity_reference", torch.float32, len(cmsbl["modalities"])),
            "cmsbl_metric_ema": ("metric_ema", torch.float32, len(cmsbl["modalities"])),
            "cmsbl_metric_initialized": ("metric_initialized", torch.bool, len(cmsbl["modalities"])),
            "cmsbl_mask_loss_ema": ("mask_loss_ema", torch.float32, NUM_NON_EMPTY_MASKS),
            "cmsbl_mask_initialized": ("mask_initialized", torch.bool, NUM_NON_EMPTY_MASKS),
            "cmsbl_mask_counts": ("mask_counts", torch.long, NUM_NON_EMPTY_MASKS),
        }
        for state_key, (payload_key, dtype, size) in shapes.items():
            value = torch.as_tensor(saved.get(payload_key), dtype=dtype).reshape(-1)
            if value.shape != (size,):
                raise ValueError(f"Invalid CMSBL checkpoint tensor {payload_key!r}.")
            state[state_key] = value.cpu()
        _reset_epoch_state(state)

    def before_epoch(self, context: ExtensionContext, state: Any, *, epoch: int) -> None:
        del context, epoch
        _reset_epoch_state(state)

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
        del step
        config = state.get("config", {})
        if not config.get("enabled"):
            return ForwardControls()
        modalities = tuple(getattr(context.primary_model, "modalities", ()))
        if not modalities:
            raise ValueError("u_mask_beam_jepa requires primary_model.modalities.")
        mask_config = config["missing_mask"]
        if mask_config["mode"] == "external":
            mask = _external_missing_mask(batch, labels, modalities, context.device)
        else:
            mask = sample_missing_mask(
                int(labels.shape[0]),
                len(modalities),
                mask_config["p_missing"],
                always_available_indices=mask_config.get("always_available_indices"),
                ensure_at_least_one=mask_config["ensure_at_least_one"],
                device=context.device,
            )
        superset = config["superset_consistency"]
        state["online_superset"] = (
            _online_superset(context, batch, modalities)
            if superset["enabled"] and superset["confidence_gated_kl"]
            else None
        )
        kwargs: dict[str, Any] = {"missing_mask": mask}
        if config.get("bcacl", {}).get("enabled"):
            kwargs.update(_bcacl_forward_kwargs(context, batch, modalities, fusion_mask=mask))
        state["epoch_number"] = int(epoch) + 1
        return ForwardControls(model_kwargs=kwargs)

    def compute_base_loss(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> BaseLossResult | None:
        config = state.get("config", {})
        if not config.get("enabled"):
            return None
        output = {
            "logits": batch_state.primary_logits,
            "input_features": batch_state.primary_output.input_features,
            "output_features": batch_state.primary_output.output_features,
            **batch_state.primary_output.diagnostics,
        }
        cmsbl = config["cmsbl"]
        sample_weights = None
        mask_ids = None
        if cmsbl.get("enabled") and cmsbl["hard_mask"]["enabled"]:
            mask_ids = fusion_mask_ids(output["missing_mask"], tuple(cmsbl["modalities"]))
            table, difficulty = hard_mask_weights(
                state["cmsbl_mask_loss_ema"],
                state["cmsbl_mask_counts"],
                state["cmsbl_mask_initialized"],
                epoch_number=batch_state.epoch + 1,
                config=cmsbl["hard_mask"],
            )
            sample_weights = table.to(device=batch_state.primary_logits.device)[mask_ids - 1]
            state["cmsbl_epoch_mask_weights"] = table.detach().cpu()
            state["cmsbl_epoch_mask_difficulty"] = difficulty.detach().cpu()

        superset = config["superset_consistency"]
        topology = config["prototype_topology"]
        result = u_mask_beam_jepa_loss(
            output,
            batch_state.labels,
            prototype_bank=getattr(context.primary_model, "prototype_bank", None),
            use_beam_prototype_alignment=config["use_beam_prototype_alignment"],
            lambda_proto=config["lambda_proto"],
            lambda_modality_proto=config["lambda_modality_proto"],
            beam_label_sigma=config["beam_label_sigma"],
            prototype_target_circular=config["prototype_target_circular"],
            prototype_topology_id=topology["id"] if topology["id"] != "not_applicable" else None,
            prototype_topology_permutation=topology["permutation"],
            use_amber_cma_analogue=config["use_amber_cma_analogue"],
            lambda_amber_cma=config["lambda_amber_cma"],
            amber_cma_temperature=config["amber_cma_temperature"],
            sample_ids=sample_ids_from_batch(batch_state.batch) if config["use_amber_cma_analogue"] else None,
            superset_output=state.get("online_superset"),
            use_superset_confidence_gated_kl=superset["enabled"] and superset["confidence_gated_kl"],
            lambda_superset_consistency=superset["kl_weight"],
            superset_temperature=superset["temperature"],
            router_oracle_weight=config["router_oracle_weight"],
            router_oracle_target_mode=config["router_oracle_target_mode"],
            router_oracle_temperature=config["router_oracle_temperature"],
            fusion_sample_weights=sample_weights,
        )
        total = result["loss"]
        diagnostics = dict(result["diagnostics"])
        if config.get("bcacl", {}).get("enabled"):
            auxiliary = _bcacl_loss(state, batch_state, output)
            total = total + auxiliary["loss"]
            diagnostics.update(_bcacl_diagnostics(auxiliary))
            _accumulate_capacity_statistics(state, auxiliary)
        if mask_ids is not None:
            raw = result["per_sample_fusion_loss"]
            accumulate_mask_losses(
                state["cmsbl_epoch_mask_loss_sums"],
                state["cmsbl_epoch_mask_counts"],
                raw,
                mask_ids,
            )
        task_loss = result["loss_beam"]
        return BaseLossResult(
            total_loss=total,
            task_loss=task_loss,
            auxiliary_loss=total - task_loss,
            diagnostics=diagnostics,
        )

    def after_epoch(
        self,
        context: ExtensionContext,
        state: Any,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        cmsbl = state.get("config", {}).get("cmsbl", {})
        if not cmsbl.get("enabled"):
            return {}
        epoch_number = int(epoch) + 1
        _update_capacity_state(state, cmsbl, epoch_number=epoch_number)
        _update_mask_state(state, cmsbl, epoch_number=epoch_number)
        metrics, modality_rows, mask_rows = _cmsbl_diagnostics(state, cmsbl, epoch_number=epoch_number)
        if cmsbl["diagnostics"]["enabled"]:
            dataset = _dataset_name(context.cfg)
            state["cmsbl_diagnostic_path"] = write_cmsbl_epoch_diagnostics(
                context.run_dir,
                epoch=epoch_number,
                dataset=dataset,
                modalities=tuple(cmsbl["modalities"]),
                capacity_identity=state["cmsbl_capacity_identity"],
                modality_rows=modality_rows,
                mask_rows=mask_rows,
                losses={
                    key: value / max(state["cmsbl_loss_batches"], 1)
                    for key, value in state["cmsbl_loss_sums"].items()
                },
                metrics=metrics,
            )
        return metrics


def _bcacl_loss(
    state: dict[str, Any],
    batch_state: BatchState,
    output: dict[str, torch.Tensor],
) -> dict[str, Any]:
    config = state["config"]
    bcacl = config["bcacl"]
    cmsbl = config["cmsbl"]
    private_weight = 1.0
    shared_weight = float(bcacl["lambda_shared"])
    private_modality = shared_modality = None
    if cmsbl.get("enabled"):
        scheduled_private, scheduled_shared = effective_auxiliary_weights(cmsbl, batch_state.epoch + 1)
        private_weight *= scheduled_private
        shared_weight *= scheduled_shared
        state["cmsbl_epoch_aux_weights"] = torch.tensor([private_weight, shared_weight])
        capacity = cmsbl["capacity_gap"]
        if capacity["enabled"]:
            modality_weights, gaps = capacity_gap_weights(
                state["cmsbl_capacity_reference"],
                state["cmsbl_metric_ema"],
                state["cmsbl_metric_initialized"],
                epoch_number=batch_state.epoch + 1,
                config=capacity,
            )
            state["cmsbl_epoch_modality_weights"] = modality_weights.detach().cpu()
            state["cmsbl_epoch_capacity_gaps"] = gaps.detach().cpu()
            if "private" in capacity["apply_to"]:
                private_modality = modality_weights
            if "shared" in capacity["apply_to"]:
                shared_modality = modality_weights
    result = bcacl_auxiliary_loss(
        features=output["bcacl_features"],
        private_logits=output["bcacl_private_logits"],
        shared_logits=output["bcacl_shared_logits"],
        labels=batch_state.labels,
        observed_mask=output["bcacl_observed_mask"],
        lambda_private=private_weight,
        lambda_shared=shared_weight,
        private_modality_weights=private_modality,
        shared_modality_weights=shared_modality,
    )
    if cmsbl.get("enabled"):
        state["cmsbl_loss_sums"]["private_raw"] += _scalar(result["loss_private"])
        state["cmsbl_loss_sums"]["private_weighted"] += _scalar(result["loss_private_weighted"])
        state["cmsbl_loss_sums"]["shared_raw"] += _scalar(result["loss_shared"])
        state["cmsbl_loss_sums"]["shared_weighted"] += _scalar(result["loss_shared_weighted"])
        state["cmsbl_loss_batches"] += 1
    return result


def _bcacl_diagnostics(result: dict[str, Any]) -> dict[str, float]:
    return {
        "loss/bcacl_private": _scalar(result["loss_private"]),
        "loss/bcacl_shared": _scalar(result["loss_shared"]),
        "loss/bcacl_total": _scalar(result["loss"]),
    }


def _accumulate_capacity_statistics(state: dict[str, Any], result: dict[str, Any]) -> None:
    cmsbl = state["config"]["cmsbl"]
    if not cmsbl.get("enabled") or not cmsbl["capacity_gap"]["enabled"]:
        return
    state["cmsbl_epoch_modality_correct"] += result["private_correct"].detach().to(
        device=state["cmsbl_epoch_modality_correct"].device,
        dtype=torch.float64,
    )
    state["cmsbl_epoch_modality_count"] += result["observed_counts"].detach().to(
        device=state["cmsbl_epoch_modality_count"].device,
        dtype=torch.long,
    )


def _initialize_cmsbl_state(context: ExtensionContext, state: dict[str, Any]) -> None:
    cmsbl = state["config"]["cmsbl"]
    if not cmsbl.get("enabled"):
        return
    modalities = tuple(cmsbl["modalities"])
    capacity = cmsbl["capacity_gap"]
    if capacity["enabled"]:
        reference, identity = load_capacity_reference(
            cmsbl["capacity_reference"],
            dataset=_dataset_name(context.cfg),
            modalities=modalities,
        )
    else:
        reference = torch.zeros(len(modalities), dtype=torch.float32)
        identity = {"enabled": False, "dataset": _dataset_name(context.cfg), "modalities": list(modalities)}
    state.update(
        {
            "cmsbl_capacity_identity": identity,
            "cmsbl_capacity_reference": reference.cpu(),
            "cmsbl_metric_ema": torch.zeros(len(modalities), dtype=torch.float32),
            "cmsbl_metric_initialized": torch.zeros(len(modalities), dtype=torch.bool),
            "cmsbl_mask_loss_ema": torch.zeros(NUM_NON_EMPTY_MASKS, dtype=torch.float32),
            "cmsbl_mask_initialized": torch.zeros(NUM_NON_EMPTY_MASKS, dtype=torch.bool),
            "cmsbl_mask_counts": torch.zeros(NUM_NON_EMPTY_MASKS, dtype=torch.long),
        }
    )


def _reset_epoch_state(state: dict[str, Any]) -> None:
    cmsbl = state.get("config", {}).get("cmsbl", {})
    state["online_superset"] = None
    if not cmsbl.get("enabled"):
        return
    modality_count = len(cmsbl["modalities"])
    device = torch.device(state.get("device", "cpu"))
    state.update(
        {
            "cmsbl_epoch_modality_correct": torch.zeros(modality_count, device=device, dtype=torch.float64),
            "cmsbl_epoch_modality_count": torch.zeros(modality_count, device=device, dtype=torch.long),
            "cmsbl_epoch_mask_loss_sums": torch.zeros(NUM_NON_EMPTY_MASKS, device=device, dtype=torch.float64),
            "cmsbl_epoch_mask_counts": torch.zeros(NUM_NON_EMPTY_MASKS, device=device, dtype=torch.long),
            "cmsbl_epoch_aux_weights": torch.tensor([1.0, 1.0]),
            "cmsbl_epoch_modality_weights": torch.ones(modality_count),
            "cmsbl_epoch_capacity_gaps": torch.zeros(modality_count),
            "cmsbl_epoch_mask_weights": torch.ones(NUM_NON_EMPTY_MASKS),
            "cmsbl_epoch_mask_difficulty": torch.ones(NUM_NON_EMPTY_MASKS),
            "cmsbl_current_metric": torch.zeros(modality_count),
            "cmsbl_loss_sums": {
                "private_raw": 0.0,
                "private_weighted": 0.0,
                "shared_raw": 0.0,
                "shared_weighted": 0.0,
            },
            "cmsbl_loss_batches": 0,
        }
    )


def _update_capacity_state(state: dict[str, Any], cmsbl: dict[str, Any], *, epoch_number: int) -> None:
    capacity = cmsbl["capacity_gap"]
    if not capacity["enabled"]:
        return
    correct = state["cmsbl_epoch_modality_correct"]
    counts = state["cmsbl_epoch_modality_count"]
    _all_reduce_pair(correct, counts)
    current = (correct / counts.clamp_min(1).to(correct)).float().cpu()
    valid = counts.gt(0).cpu()
    state["cmsbl_current_metric"] = current
    if epoch_number % int(capacity["update_interval"]) == 0:
        ema, initialized = update_metric_ema(
            state["cmsbl_metric_ema"],
            state["cmsbl_metric_initialized"],
            current,
            valid,
            momentum=capacity["ema_momentum"],
        )
        state["cmsbl_metric_ema"] = ema.cpu()
        state["cmsbl_metric_initialized"] = initialized.cpu()


def _update_mask_state(state: dict[str, Any], cmsbl: dict[str, Any], *, epoch_number: int) -> None:
    hard = cmsbl["hard_mask"]
    if not hard["enabled"]:
        return
    sums = state["cmsbl_epoch_mask_loss_sums"]
    counts = state["cmsbl_epoch_mask_counts"]
    all_reduce_mask_statistics(sums, counts)
    if epoch_number % int(hard["update_interval"]) == 0:
        ema, initialized, cumulative = update_mask_loss_ema(
            state["cmsbl_mask_loss_ema"],
            state["cmsbl_mask_initialized"],
            state["cmsbl_mask_counts"],
            sums.detach().cpu().float(),
            counts.detach().cpu(),
            momentum=hard["ema_momentum"],
            min_count=hard["min_count"],
        )
        state["cmsbl_mask_loss_ema"] = ema.cpu()
        state["cmsbl_mask_initialized"] = initialized.cpu()
        state["cmsbl_mask_counts"] = cumulative.cpu()


def _cmsbl_diagnostics(
    state: dict[str, Any],
    cmsbl: dict[str, Any],
    *,
    epoch_number: int,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    private, shared = effective_auxiliary_weights(cmsbl, epoch_number)
    capacity_weights, gaps = capacity_gap_weights(
        state["cmsbl_capacity_reference"],
        state["cmsbl_metric_ema"],
        state["cmsbl_metric_initialized"],
        epoch_number=epoch_number,
        config=cmsbl["capacity_gap"],
    ) if cmsbl["capacity_gap"]["enabled"] else (
        torch.ones(len(cmsbl["modalities"])),
        torch.zeros(len(cmsbl["modalities"])),
    )
    mask_weights, difficulty = hard_mask_weights(
        state["cmsbl_mask_loss_ema"],
        state["cmsbl_mask_counts"],
        state["cmsbl_mask_initialized"],
        epoch_number=epoch_number,
        config=cmsbl["hard_mask"],
    ) if cmsbl["hard_mask"]["enabled"] else (
        torch.ones(NUM_NON_EMPTY_MASKS),
        torch.ones(NUM_NON_EMPTY_MASKS),
    )
    metrics = {"cmsbl/lambda_private": float(private), "cmsbl/lambda_shared": float(shared)}
    modality_rows = []
    for index, name in enumerate(cmsbl["modalities"]):
        row = {
            "name": name,
            "capacity_reference": float(state["cmsbl_capacity_reference"][index]),
            "current_metric": float(state["cmsbl_current_metric"][index]),
            "ema_metric": float(state["cmsbl_metric_ema"][index]),
            "ema_initialized": bool(state["cmsbl_metric_initialized"][index]),
            "gap": float(gaps[index]),
            "weight": float(capacity_weights[index]),
        }
        modality_rows.append(row)
        metrics[f"cmsbl/capacity/{name}/weight"] = row["weight"]
    mask_rows = []
    epoch_counts = state["cmsbl_epoch_mask_counts"].detach().cpu()
    for index in range(NUM_NON_EMPTY_MASKS):
        row = {
            "id": index + 1,
            "name": mask_name(index + 1),
            "epoch_count": int(epoch_counts[index]),
            "cumulative_count": int(state["cmsbl_mask_counts"][index]),
            "raw_loss_ema": float(state["cmsbl_mask_loss_ema"][index]),
            "initialized": bool(state["cmsbl_mask_initialized"][index]),
            "difficulty": float(difficulty[index]),
            "weight": float(mask_weights[index]),
        }
        mask_rows.append(row)
        metrics[f"cmsbl/mask/{index + 1}/count"] = float(row["cumulative_count"])
        metrics[f"cmsbl/mask/{index + 1}/loss_ema"] = row["raw_loss_ema"]
        metrics[f"cmsbl/mask/{index + 1}/weight"] = row["weight"]
    return metrics, modality_rows, mask_rows


def _bcacl_forward_kwargs(
    context: ExtensionContext,
    batch: dict[str, Any],
    modalities: tuple[str, ...],
    *,
    fusion_mask: torch.Tensor,
) -> dict[str, Any]:
    payload = batch.get(TEMPORAL_SUPERSET_PAYLOAD_KEY)
    if not isinstance(payload, Mapping):
        return {"bcacl_fusion_mask": fusion_mask}
    if tuple(payload.get("modalities", ())) != modalities:
        raise ValueError("BCACL temporal superset modalities do not match the model.")
    base_mask = torch.as_tensor(payload.get("base_mask"), device=context.device, dtype=torch.bool)
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("BCACL temporal superset payload is missing original inputs.")
    return {
        "bcacl_observed_inputs": _observed_model_inputs(
            inputs,
            modalities,
            seq_length=context.seq_length,
            device=context.device,
            non_blocking=context.non_blocking,
        ),
        "bcacl_observed_temporal_mask": base_mask,
        "bcacl_fusion_mask": fusion_mask & base_mask.any(dim=1),
    }


def _observed_model_inputs(
    values: Mapping[str, Any],
    modalities: tuple[str, ...],
    *,
    seq_length: int,
    device: torch.device,
    non_blocking: bool,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in modalities:
        if name == "radar":
            ra = _sequence(values, "radar_ra", seq_length, device, non_blocking)
            da = _sequence(values, "radar_da", seq_length, device, non_blocking)
            if ra.ndim == 4:
                ra, da = ra.unsqueeze(2), da.unsqueeze(2)
            result["radar_batch"] = torch.cat([ra, da], dim=2)
        else:
            result[f"{name}_batch"] = _sequence(values, name, seq_length, device, non_blocking)
    return result


def _sequence(
    values: Mapping[str, Any],
    key: str,
    seq_length: int,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    if key not in values:
        raise ValueError(f"BCACL original inputs are missing {key}.")
    value = torch.as_tensor(values[key]).to(device=device, non_blocking=non_blocking)
    value = value[:, -int(seq_length):]
    if value.shape[1] == int(seq_length):
        return value
    if value.shape[1] == 0:
        raise ValueError(f"BCACL original input {key} has no timesteps.")
    pad = value[:, :1].expand(-1, int(seq_length) - value.shape[1], *([-1] * (value.ndim - 2)))
    return torch.cat([pad, value], dim=1)


def _online_superset(
    context: ExtensionContext,
    batch: dict[str, Any],
    modalities: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    from kd_sensing.engine.runtime import run_model_step

    restored, mask = _restore_temporal_superset(batch, modalities, context.device)
    states = [(module, module.training) for module in context.primary_model.modules()]
    try:
        context.primary_model.eval()
        with torch.no_grad():
            step = run_model_step(
                context.primary_model,
                context.task,
                restored,
                seq_length=context.seq_length,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
                extra_model_kwargs={"missing_mask": mask},
            )
    finally:
        for module, training in states:
            module.training = training
    return {"logits": step.logits.detach()}


def _restore_temporal_superset(
    batch: dict[str, Any],
    modalities: tuple[str, ...],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    payload = batch.get(TEMPORAL_SUPERSET_PAYLOAD_KEY)
    if not isinstance(payload, Mapping):
        raise ValueError("Enabled temporal superset consistency requires preserved unmasked inputs.")
    if tuple(payload.get("modalities", ())) != modalities:
        raise ValueError("Temporal superset modalities do not match the model.")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Temporal superset payload is missing original inputs.")
    restored = dict(batch)
    restored.update(inputs)
    base_mask = torch.as_tensor(payload.get("base_mask"), dtype=torch.bool)
    restored["modality_temporal_mask"] = base_mask
    restored["temporal_mask"] = base_mask.any(dim=2)
    restored["available_modalities"] = base_mask.any(dim=1)
    mask = base_mask.any(dim=1).to(device=device)
    if not bool(mask.any(dim=1).all().item()):
        raise ValueError("Temporal superset requires at least one available modality per sample.")
    return restored, mask


def _external_missing_mask(
    batch: Mapping[str, Any],
    labels: torch.Tensor,
    modalities: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    value = batch.get("available_modalities")
    if value is None:
        temporal = batch.get("modality_temporal_mask")
        value = torch.as_tensor(temporal).any(dim=1) if temporal is not None else None
    if value is None:
        raise ValueError("external missing-mask mode requires batch.available_modalities.")
    mask = torch.as_tensor(value, device=device, dtype=torch.bool)
    expected = (int(labels.shape[0]), len(modalities))
    if tuple(mask.shape) != expected:
        raise ValueError(f"external missing mask must have shape {expected}.")
    if not bool(mask.any(dim=1).all().item()):
        raise ValueError("external missing mask requires at least one available modality per sample.")
    return mask


def _router_oracle_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    unimodal = output.get("unimodal_logits")
    router = output.get("router_gate_logits")
    available = output.get("missing_mask")
    if not all(torch.is_tensor(value) for value in (unimodal, router, available)):
        raise ValueError("Router oracle supervision requires unimodal logits, router logits, and missing mask.")
    target = labels[:, 0]
    safe = target.clamp_min(0)
    predictions = unimodal.detach().argmax(dim=-1)
    distance = (predictions - safe.unsqueeze(1)).abs()
    unavailable = ~available.to(dtype=torch.bool)
    distance = distance.masked_fill(unavailable, unimodal.shape[-1] + 1)
    minimum = distance.min(dim=1, keepdim=True).values
    ties = distance.eq(minimum) & ~unavailable
    oracle = distance.argmin(dim=1)
    valid = target.ne(-100)
    if bool(valid.any().item()):
        loss = F.cross_entropy(router[valid] / float(temperature), oracle[valid])
    else:
        loss = router.sum() * 0.0
    return loss, {
        "router_oracle_active_ratio": _scalar(valid.float().mean()),
        "router_oracle_tie_ratio": _scalar(ties.sum(dim=1).gt(1).float().mean()),
        "router_oracle_enabled": 1.0,
        "router_oracle_disabled": 0.0,
    }


def _confidence_gated_temperature_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher = _as_prediction_logits(teacher).to(device=student.device, dtype=student.dtype)
    if teacher.shape != student.shape:
        raise ValueError("Superset logits must match masked logits.")
    t = float(temperature)
    per_item = F.kl_div(
        F.log_softmax(student / t, dim=-1),
        F.softmax(teacher / t, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (t * t)
    safe = labels.clamp_min(0).unsqueeze(-1)
    teacher_true = F.softmax(teacher.detach(), dim=-1).gather(-1, safe).squeeze(-1)
    student_true = F.softmax(student.detach(), dim=-1).gather(-1, safe).squeeze(-1)
    gate = (teacher_true - student_true).clamp_min(0.0) * labels.ne(-100)
    raw = (per_item * labels.ne(-100)).sum() / labels.ne(-100).sum().clamp_min(1)
    weighted = (per_item * gate).sum() / gate.sum().clamp_min(torch.finfo(per_item.dtype).tiny)
    return weighted, raw, gate


def _as_prediction_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits.unsqueeze(1)
    if logits.ndim != 3:
        raise ValueError("Beam logits must have shape [B,C] or [B,H,C].")
    return logits


def _as_prediction_labels(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    value = labels.to(device=logits.device, dtype=torch.long)
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if value.shape != logits.shape[:2]:
        raise ValueError(f"Beam labels must have shape {tuple(logits.shape[:2])}.")
    return value


def _beam_supervised_loss_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    valid = labels.ne(-100)
    return (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _weighted_sample_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if values.reshape(-1).shape != weights.shape:
        raise ValueError("CMSBL sample weights must match per-sample losses.")
    return (values.reshape(-1) * weights).sum() / weights.sum().clamp_min(torch.finfo(values.dtype).tiny)


def _loss_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_beam: torch.Tensor,
) -> dict[str, float]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = flat_labels.ne(-100)
    if bool(valid.any().item()):
        predictions = flat_logits[valid]
        targets = flat_labels[valid]
        top1 = predictions.argmax(dim=-1).eq(targets).float().mean()
        top5 = predictions.topk(min(5, predictions.shape[-1]), dim=-1).indices.eq(
            targets.unsqueeze(-1)
        ).any(dim=-1).float().mean()
    else:
        top1 = top5 = loss_beam.detach() * 0.0
    return {
        "loss_beam": _scalar(loss_beam),
        "ce_loss": _scalar(loss_beam),
        "accuracy/top1": _scalar(top1),
        "accuracy/top5": _scalar(top5),
    }


def _dataset_name(cfg: Mapping[str, Any]) -> str:
    data = cfg.get("data", {})
    dataset = data.get("dataset", {}) if isinstance(data, Mapping) else {}
    value = dataset.get("type", "unknown") if isinstance(dataset, Mapping) else "unknown"
    return str(value).strip().lower()


def _all_reduce_pair(values: torch.Tensor, counts: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)


def _scalar(value: torch.Tensor | float | int) -> float:
    return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)


__all__ = ["UMaskBeamJEPATrainingExtension", "u_mask_beam_jepa_loss"]
