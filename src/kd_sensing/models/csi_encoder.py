from typing import Any

import torch
import torch.nn as nn

from kd_sensing.models.csi_debug import (
    _complex_tensor_stats,
    _debug_enabled,
    _hardening_drift_warning,
    _pilot_debug_values,
    _real_tensor_stats,
    _tensor_norm,
)
from kd_sensing.models.csi_estimation import PilotCSIChannelEstimator
from kd_sensing.models.csi_hardening import CSIHardening, _deep_merge_dict
from kd_sensing.models.csi_views import (
    CSIViewTokenizer,
    SymmetricViewFusion,
    _as_complex_csi,
    _constant_view_gate,
    _view_fusion_code,
    delay_view,
    frequency_view,
)
from kd_sensing.registries import ENCODERS

def _resolve_output_dim(output_dim: int | None, *fallbacks: int | None, default: int = 64) -> int:
    for candidate in (output_dim, *fallbacks):
        if candidate is not None:
            value = int(candidate)
            if value <= 0:
                raise ValueError(f"CSI output dimension must be positive, got {value}.")
            return value
    return int(default)


@ENCODERS.register("pilot_dual_view_csi")
class PilotDualViewCSIEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        d_model: int | None = None,
        feature_size: int | None = None,
        train_rms: float | None = None,
        csi_train_rms: float | None = None,
        csi_estimation: dict[str, Any] | None = None,
        pilot_estimator: dict[str, Any] | None = None,
        mode: str | None = None,
        pilot_len: int = 16,
        pilot_power: float = 1.0,
        noise_var: float | None = None,
        snr_db: float | None = None,
        est_snr_db: float | None = None,
        train_snr_min_db: float | None = None,
        train_snr_max_db: float | None = None,
        delay_taps: int | None = 32,
        view_fusion: str = "symmetric_gate",
        view_gate_warmup_epochs: int = 0,
        view_gate_warmup_mode: str = "mean",
        delay_view_warmup_epochs: int = 0,
        delay_view_warmup_mode: str = "freq_only",
        use_internal_gru: bool = True,
        csi_hardening: dict[str, Any] | bool | None = None,
        debug: bool | dict[str, Any] = False,
        hidden_channels: int = 32,
        tokenizer_hidden_channels: int | None = None,
        tokenizer: dict[str, Any] | None = None,
        temporal: dict[str, Any] | None = None,
        dropout: float = 0.1,
        return_aux: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.output_dim = _resolve_output_dim(output_dim, d_model, feature_size)
        self.view_dim = self.output_dim
        rms = float(train_rms if train_rms is not None else csi_train_rms if csi_train_rms is not None else 1.0)
        if rms <= 0.0:
            raise ValueError("train_rms must be positive for PilotDualViewCSIEncoder.")
        self.register_buffer("train_rms", torch.as_tensor(rms, dtype=torch.float32), persistent=True)
        estimation_cfg = dict(csi_estimation or {})
        if isinstance(pilot_estimator, dict):
            estimation_cfg = _deep_merge_dict(estimation_cfg, pilot_estimator)
        pilot_enabled = bool(estimation_cfg.pop("enabled", estimation_cfg.pop("enable", True)))
        estimator_kwargs = {
            "enabled": pilot_enabled,
            "mode": estimation_cfg.pop("mode", mode or "none"),
            "pilot_len": estimation_cfg.pop("pilot_len", pilot_len),
            "pilot_power": estimation_cfg.pop("pilot_power", pilot_power),
            "noise_var": estimation_cfg.pop("noise_var", noise_var),
            "snr_db": estimation_cfg.pop("snr_db", snr_db),
            "est_snr_db": estimation_cfg.pop("est_snr_db", est_snr_db),
            "train_snr_min_db": estimation_cfg.pop("train_snr_min_db", train_snr_min_db),
            "train_snr_max_db": estimation_cfg.pop("train_snr_max_db", train_snr_max_db),
        }
        if not pilot_enabled:
            estimator_kwargs["mode"] = "none"
        estimator_kwargs.update(estimation_cfg)
        self.estimator = PilotCSIChannelEstimator(**estimator_kwargs)
        self.delay_taps = None if delay_taps is None else int(delay_taps)
        if self.delay_taps is not None and self.delay_taps <= 0:
            raise ValueError("delay_taps must be positive when provided.")
        self.view_fusion = str(view_fusion or "symmetric_gate").lower()
        if self.view_fusion not in {"mean", "concat", "symmetric_gate", "freq_only"}:
            raise ValueError("view_fusion must be one of mean, concat, symmetric_gate, freq_only.")
        self.view_gate_warmup_epochs = max(int(view_gate_warmup_epochs), 0)
        self.view_gate_warmup_mode = str(view_gate_warmup_mode or "mean").lower()
        if self.view_gate_warmup_mode != "mean":
            raise ValueError("view_gate_warmup_mode currently supports only 'mean'.")
        self.delay_view_warmup_epochs = max(int(delay_view_warmup_epochs), 0)
        self.delay_view_warmup_mode = str(delay_view_warmup_mode or "freq_only").lower()
        if self.delay_view_warmup_mode != "freq_only":
            raise ValueError("delay_view_warmup_mode currently supports only 'freq_only'.")
        self.current_epoch = 0
        self.use_internal_gru = bool(use_internal_gru)
        self.csi_hardening = CSIHardening(csi_hardening)
        tokenizer_cfg = dict(tokenizer or {})
        hidden = int(tokenizer_cfg.get("hidden_channels", tokenizer_hidden_channels or hidden_channels))
        tokenizer_dropout = float(tokenizer_cfg.get("dropout", dropout))
        use_second_conv = bool(tokenizer_cfg.get("use_second_conv", True))
        self.frequency_tokenizer = CSIViewTokenizer(
            self.view_dim,
            hidden_channels=hidden,
            dropout=tokenizer_dropout,
            use_second_conv=use_second_conv,
        )
        self.delay_tokenizer = CSIViewTokenizer(
            self.view_dim,
            hidden_channels=hidden,
            dropout=tokenizer_dropout,
            use_second_conv=use_second_conv,
        )
        if self.view_fusion == "concat":
            self.concat_projection = nn.Linear(self.view_dim * 2, self.output_dim)
        elif self.view_fusion == "symmetric_gate":
            self.symmetric_fusion = SymmetricViewFusion(self.view_dim)
        temporal_cfg = dict(temporal or {})
        temporal_layers = int(temporal_cfg.get("num_layers", 1))
        temporal_dropout = float(temporal_cfg.get("dropout", dropout)) if temporal_layers > 1 else 0.0
        self.temporal = (
            nn.GRU(
                input_size=self.output_dim,
                hidden_size=self.output_dim,
                num_layers=temporal_layers,
                dropout=temporal_dropout,
                batch_first=True,
            )
            if self.use_internal_gru
            else None
        )
        self.return_aux = bool(return_aux)
        self.last_aux: dict[str, torch.Tensor] = {}
        self.debug_enabled = _debug_enabled(debug)
        self.debug_config = dict(debug) if isinstance(debug, dict) else {}
        self._debug_batch_source = "unknown"
        self._debug_recorded_sources: set[str] = set()
        self.debug_records: list[dict[str, Any]] = []

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def set_debug_enabled(self, enabled: bool = True) -> None:
        self.debug_enabled = bool(enabled)

    def set_debug_batch_source(self, source: str) -> None:
        self._debug_batch_source = str(source or "unknown")

    def consume_debug_records(self) -> list[dict[str, Any]]:
        records = list(self.debug_records)
        self.debug_records.clear()
        return records

    def forward(self, csi_batch: torch.Tensor, *, return_aux: bool | None = None):
        csi = _as_complex_csi(csi_batch)
        csi = csi / self.train_rms.to(device=csi.device, dtype=csi.real.dtype)
        want_aux = self.return_aux if return_aux is None else bool(return_aux)
        hardening_aux: dict[str, torch.Tensor] = {}
        normalized_csi = csi
        if self.csi_hardening.enabled:
            csi, hardening_aux = self.csi_hardening(csi, return_aux=True)
        hardened_csi = csi
        estimated, estimator_aux = self.estimator(csi, return_aux=True)
        freq_view = frequency_view(estimated)
        freq_features = self.frequency_tokenizer(freq_view)
        aux = {**hardening_aux, **estimator_aux}
        active_fusion = self._active_view_fusion()
        delay_features = None
        delay_view_tensor = None
        if active_fusion != "freq_only":
            delay_view_tensor = delay_view(estimated, delay_taps=self.delay_taps)
            delay_features = self.delay_tokenizer(delay_view_tensor)
        if active_fusion == "freq_only":
            fused = freq_features
            aux["view_gate"] = _constant_view_gate(freq_features, 1.0, 0.0)
        elif active_fusion == "mean":
            assert delay_features is not None
            fused = 0.5 * (freq_features + delay_features)
            if self.current_epoch < self.view_gate_warmup_epochs or self.view_fusion == "mean":
                aux["view_gate"] = _constant_view_gate(freq_features, 0.5, 0.5)
        elif active_fusion == "concat":
            assert delay_features is not None
            fused = self.concat_projection(torch.cat([freq_features, delay_features], dim=-1))
        else:
            assert delay_features is not None
            fused, gate_aux = self.symmetric_fusion(freq_features, delay_features, return_aux=True)
            aux.update(gate_aux)
        if active_fusion != "symmetric_gate" or active_fusion != self.view_fusion:
            aux["view_fusion_active"] = torch.as_tensor(
                _view_fusion_code(active_fusion),
                dtype=torch.long,
                device=fused.device,
            )
        if self.temporal is None:
            output = fused
            gru_output = fused
        else:
            output, _ = self.temporal(fused)
            gru_output = output
        self.last_aux = aux
        if self._should_record_debug():
            record = self._debug_record(
                normalized_csi=normalized_csi,
                hardened_csi=hardened_csi,
                estimated_csi=estimated,
                freq_view_tensor=freq_view,
                delay_view_tensor=delay_view_tensor,
                freq_features=freq_features,
                delay_features=delay_features,
                fused=fused,
                gru_output=gru_output,
                final_features=output,
                aux=aux,
            )
            self.debug_records.append(record)
            self._debug_recorded_sources.add(record["source"])
        if want_aux:
            return output, aux
        return output

    def _active_view_fusion(self) -> str:
        if self.delay_view_warmup_epochs > 0 and self.current_epoch < self.delay_view_warmup_epochs:
            return self.delay_view_warmup_mode
        if self.view_gate_warmup_epochs > 0 and self.current_epoch < self.view_gate_warmup_epochs:
            return self.view_gate_warmup_mode
        return self.view_fusion

    def _should_record_debug(self) -> bool:
        source = str(self._debug_batch_source or "unknown")
        if source == "validation":
            source = "val"
        return bool(self.debug_enabled) and source in {"train", "val", "validation"} and source not in self._debug_recorded_sources

    def _debug_record(
        self,
        *,
        normalized_csi: torch.Tensor,
        hardened_csi: torch.Tensor,
        estimated_csi: torch.Tensor,
        freq_view_tensor: torch.Tensor,
        delay_view_tensor: torch.Tensor | None,
        freq_features: torch.Tensor,
        delay_features: torch.Tensor | None,
        fused: torch.Tensor,
        gru_output: torch.Tensor,
        final_features: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        source = str(self._debug_batch_source or "unknown")
        if source == "validation":
            source = "val"
        record: dict[str, Any] = {
            "source": source,
            "epoch": int(self.current_epoch),
            "structure": {
                "use_internal_gru": bool(self.use_internal_gru),
                "view_fusion": self.view_fusion,
                "active_view_fusion": self._active_view_fusion(),
                "delay_taps": self.delay_taps,
                "d_model": int(self.output_dim),
            },
            "complex": {
                "before_hardening": _complex_tensor_stats(normalized_csi),
                "after_hardening": _complex_tensor_stats(hardened_csi),
                "after_pilot": _complex_tensor_stats(estimated_csi),
            },
            "views": {
                "freq_view": _real_tensor_stats(freq_view_tensor),
                "delay_view": _real_tensor_stats(delay_view_tensor) if delay_view_tensor is not None else None,
            },
            "feature_norms": {
                "freq_feat": _tensor_norm(freq_features),
                "delay_feat": _tensor_norm(delay_features),
                "fused_feat": _tensor_norm(fused),
                "gru_out": _tensor_norm(gru_output),
                "final_csi_feature": _tensor_norm(final_features),
            },
            "pilot": _pilot_debug_values(aux),
            "hardening": self._hardening_debug_values(normalized_csi, hardened_csi),
        }
        gate = aux.get("view_gate")
        if torch.is_tensor(gate):
            record["views"]["view_gate"] = _real_tensor_stats(gate)
        if record["feature_norms"]["fused_feat"] == 0.0 or record["feature_norms"]["final_csi_feature"] == 0.0:
            record.setdefault("warnings", []).append("nonzero CSI produced zero fused or final CSI feature norm")
        return record

    def _hardening_debug_values(self, before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
        before_stats = _complex_tensor_stats(before)
        after_stats = _complex_tensor_stats(after)
        result: dict[str, Any] = {
            "enabled": bool(self.csi_hardening.enabled),
            "shape_preserved": list(before.shape) == list(after.shape),
            "nan_count": int(after_stats["nan_count"]),
            "zero_ratio": float(after_stats["zero_ratio"]),
            "transform_identity": self.csi_hardening.transform_identity(),
        }
        drift_warning = _hardening_drift_warning(before_stats, after_stats, self.csi_hardening.config)
        if drift_warning is not None:
            result["warning"] = drift_warning
        return result



__all__ = ["PilotDualViewCSIEncoder"]
