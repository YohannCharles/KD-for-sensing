from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ADAPTER_TYPES = {
    "none",
    "adapter_v1",
    "circular_affine",
    "circular_affine_spline",
    "branch_mixture_circular",
}
ABLATIONS = {
    "backbone_only",
    "adapter_v1",
    "circular_affine",
    "circular_affine_spline",
    "branch_mixture_circular",
    "branch_mixture_circular_weighted",
    "geo_only",
    "geo_plus_backbone",
}


@dataclass(frozen=True)
class SceneAdapterV2Config:
    adapter_type: str = "circular_affine_spline"
    num_scenes: int = 4
    num_beams: int = 64
    num_bins: int = 16
    max_branches: int = 4
    min_branch_support: int = 5
    sigma: float = 2.0
    tau: float = 1.0
    smoothness_weight: float = 0.001
    scene_names: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "SceneAdapterV2Config":
        raw = dict(payload or {})
        if "type" in raw and "adapter_type" not in raw:
            raw["adapter_type"] = raw.pop("type")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        values = {key: value for key, value in raw.items() if key in allowed}
        if "scene_names" in values:
            values["scene_names"] = tuple(str(item) for item in values["scene_names"])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        adapter_type = str(self.adapter_type)
        if adapter_type not in ADAPTER_TYPES:
            raise ValueError(f"model.adapter.type must be one of {sorted(ADAPTER_TYPES)}, got {adapter_type}.")
        if int(self.num_scenes) <= 0:
            raise ValueError(f"model.adapter.num_scenes must be positive, got {self.num_scenes}.")
        if int(self.num_beams) <= 0:
            raise ValueError(f"model.adapter.num_beams must be positive, got {self.num_beams}.")
        if int(self.num_bins) <= 0:
            raise ValueError(f"model.adapter.num_bins must be positive, got {self.num_bins}.")
        if int(self.max_branches) <= 0:
            raise ValueError(f"model.adapter.max_branches must be positive, got {self.max_branches}.")


class MMWTownGpsV2Backbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_beams: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_beams = int(num_beams)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_beams),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[-1]) != self.input_dim:
            raise ValueError(
                "MMW Town GPS v2 features must have shape [B, model.input_dim]; "
                f"got {tuple(features.shape)} with model.input_dim={self.input_dim}."
            )
        return self.net(features)


class SceneAdapterV2(nn.Module):
    def __init__(self, cfg: SceneAdapterV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        scene_shape = (int(cfg.num_scenes),)
        branch_shape = (int(cfg.num_scenes), int(cfg.max_branches))
        parameter_shape = branch_shape if cfg.adapter_type == "branch_mixture_circular" else scene_shape
        self.psi_degrees = nn.Parameter(torch.zeros(parameter_shape, dtype=torch.float32))
        self.delta_beams = nn.Parameter(torch.zeros(parameter_shape, dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.zeros(parameter_shape, dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.full(parameter_shape, float(cfg.sigma), dtype=torch.float32).log())
        self.log_tau = nn.Parameter(torch.full(parameter_shape, float(cfg.tau), dtype=torch.float32).log())
        self.flip_logit = nn.Parameter(torch.full(parameter_shape, -6.0, dtype=torch.float32))
        residual_shape = (*parameter_shape, int(cfg.num_bins))
        self.periodic_residual_bins = nn.Parameter(torch.zeros(residual_shape, dtype=torch.float32))
        self.register_buffer("branch_support_counts", torch.zeros(branch_shape, dtype=torch.long), persistent=False)
        self.scene_mapping_metadata = {
            "scene_names": list(cfg.scene_names) if cfg.scene_names else [str(idx) for idx in range(int(cfg.num_scenes))],
            "num_scenes": int(cfg.num_scenes),
            "adapter_type": str(cfg.adapter_type),
        }

    def forward(
        self,
        theta_degrees: torch.Tensor,
        scene_id: torch.Tensor,
        *,
        branch_id: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        theta = torch.as_tensor(theta_degrees, device=self.psi_degrees.device, dtype=torch.float32).reshape(-1)
        scene = torch.as_tensor(scene_id, device=theta.device, dtype=torch.long).reshape(-1)
        if int(theta.shape[0]) != int(scene.shape[0]):
            raise ValueError("theta_degrees and scene_id must have matching first dimension.")
        self._validate_scene_id(scene)
        branch = self._effective_branch_id(scene, branch_id, batch_size=int(scene.shape[0]))
        params = self._select_params(scene, branch)
        forward_center = self._center_from_theta(theta, params, reverse=False)
        reverse_center = self._center_from_theta(theta, params, reverse=True)
        if self.cfg.adapter_type in {"circular_affine_spline", "branch_mixture_circular"}:
            residual = self._spline_residual(theta, scene, branch)
            forward_center = forward_center + residual
            reverse_center = reverse_center + residual
        forward_logits = _circular_gaussian_logits(
            forward_center,
            num_beams=int(self.cfg.num_beams),
            sigma=params["sigma"],
            tau=params["tau"],
        )
        reverse_logits = _circular_gaussian_logits(
            reverse_center,
            num_beams=int(self.cfg.num_beams),
            sigma=params["sigma"],
            tau=params["tau"],
        )
        forward_prob = F.softmax(forward_logits, dim=-1)
        reverse_prob = F.softmax(reverse_logits, dim=-1)
        alpha = torch.sigmoid(params["flip_logit"]).unsqueeze(-1)
        probability = (1.0 - alpha) * forward_prob + alpha * reverse_prob
        geo_logits = torch.log(probability.clamp_min(1e-12))
        return {
            "geo_logits": geo_logits,
            "adapter_diagnostics": {
                "adapter_type": self.cfg.adapter_type,
                "forward_center": forward_center.detach(),
                "reverse_center": reverse_center.detach(),
                "flip_probability": alpha.squeeze(-1).detach(),
                "branch_id": branch.detach() if branch is not None else None,
                "branch_fallback_count": int(getattr(self, "_last_branch_fallback_count", 0)),
            },
        }

    def smoothness_regularization(self) -> torch.Tensor:
        bins = self.periodic_residual_bins
        diff = bins - bins.roll(shifts=-1, dims=-1)
        return diff.pow(2).mean() * float(self.cfg.smoothness_weight)

    def _center_from_theta(self, theta: torch.Tensor, params: dict[str, torch.Tensor], *, reverse: bool) -> torch.Tensor:
        signed = -theta if reverse else theta
        scaled = ((signed + params["psi_degrees"]) % 360.0) / 360.0
        return (scaled * float(self.cfg.num_beams) * params["scale"] + params["delta_beams"]).remainder(
            float(self.cfg.num_beams)
        )

    def _select_params(self, scene: torch.Tensor, branch: torch.Tensor | None) -> dict[str, torch.Tensor]:
        if self.cfg.adapter_type == "branch_mixture_circular":
            assert branch is not None
            return {
                "psi_degrees": self.psi_degrees[scene, branch],
                "delta_beams": self.delta_beams[scene, branch],
                "scale": self.log_scale[scene, branch].exp().clamp_min(1e-6),
                "sigma": self.log_sigma[scene, branch].exp().clamp_min(1e-6),
                "tau": self.log_tau[scene, branch].exp().clamp_min(1e-6),
                "flip_logit": self.flip_logit[scene, branch],
            }
        return {
            "psi_degrees": self.psi_degrees[scene],
            "delta_beams": self.delta_beams[scene],
            "scale": self.log_scale[scene].exp().clamp_min(1e-6),
            "sigma": self.log_sigma[scene].exp().clamp_min(1e-6),
            "tau": self.log_tau[scene].exp().clamp_min(1e-6),
            "flip_logit": self.flip_logit[scene],
        }

    def _spline_residual(self, theta: torch.Tensor, scene: torch.Tensor, branch: torch.Tensor | None) -> torch.Tensor:
        table = (
            self.periodic_residual_bins[scene, branch]
            if self.cfg.adapter_type == "branch_mixture_circular" and branch is not None
            else self.periodic_residual_bins[scene]
        )
        pos = (theta.remainder(360.0) / 360.0) * int(self.cfg.num_bins)
        left = torch.floor(pos).to(torch.long).remainder(int(self.cfg.num_bins))
        right = (left + 1).remainder(int(self.cfg.num_bins))
        frac = (pos - torch.floor(pos)).to(table.dtype)
        left_value = table.gather(1, left.view(-1, 1)).squeeze(1)
        right_value = table.gather(1, right.view(-1, 1)).squeeze(1)
        return left_value * (1.0 - frac) + right_value * frac

    def _effective_branch_id(
        self,
        scene: torch.Tensor,
        branch_id: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor | None:
        if self.cfg.adapter_type != "branch_mixture_circular":
            return None
        if branch_id is None:
            branch = torch.zeros(batch_size, device=scene.device, dtype=torch.long)
        else:
            branch = torch.as_tensor(branch_id, device=scene.device, dtype=torch.long).reshape(-1)
            if int(branch.shape[0]) != batch_size:
                raise ValueError("branch_id must match theta_degrees batch size.")
            branch = branch.clamp(min=0, max=int(self.cfg.max_branches) - 1)
        counts = self.branch_support_counts[scene, branch]
        fallback = counts.lt(int(self.cfg.min_branch_support))
        self._last_branch_fallback_count = int(fallback.sum().item())
        return torch.where(fallback, torch.zeros_like(branch), branch)

    def _validate_scene_id(self, scene: torch.Tensor) -> None:
        if scene.numel() == 0:
            return
        invalid = scene.lt(0) | scene.ge(int(self.cfg.num_scenes))
        if bool(invalid.any().item()):
            bad = [int(item) for item in scene[invalid].detach().cpu().tolist()]
            raise ValueError(
                "scene_id is outside registered SceneAdapterV2 range: "
                f"scene_id={bad}, num_scenes={self.cfg.num_scenes}, "
                f"scene mapping metadata={self.scene_mapping_metadata}. "
                "Use a scene id from model.adapter.scene_names."
            )


class MMWTownGpsV2Model(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_beams: int = 64,
        adapter_cfg: SceneAdapterV2Config | None = None,
        ablation: str = "geo_plus_backbone",
        residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        if str(ablation) not in ABLATIONS:
            raise ValueError(f"model.ablation must be one of {sorted(ABLATIONS)}, got {ablation}.")
        self.ablation = str(ablation)
        self.input_dim = int(input_dim)
        self.num_beams = int(num_beams)
        self.backbone = MMWTownGpsV2Backbone(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            num_beams=int(num_beams),
        )
        self.adapter = SceneAdapterV2(
            adapter_cfg
            or SceneAdapterV2Config(
                adapter_type="none" if self.ablation == "backbone_only" else "circular_affine_spline",
                num_beams=int(num_beams),
            )
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))

    def forward(
        self,
        features: torch.Tensor,
        *,
        theta_degrees: torch.Tensor,
        scene_id: torch.Tensor,
        branch_id: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        residual_logits = self.backbone(features)
        adapter_out = self.adapter(theta_degrees, scene_id, branch_id=branch_id)
        geo_logits = adapter_out["geo_logits"]
        if self.ablation == "backbone_only":
            logits = residual_logits
        elif self.ablation == "geo_only":
            logits = geo_logits
        else:
            logits = geo_logits + self.residual_scale * residual_logits
        return {
            "logits": logits,
            "residual_logits": residual_logits,
            "geo_logits": geo_logits,
            "adapter_diagnostics": adapter_out["adapter_diagnostics"],
            "residual_scale": self.residual_scale,
        }


def _circular_gaussian_logits(
    center: torch.Tensor,
    *,
    num_beams: int,
    sigma: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    classes = torch.arange(int(num_beams), device=center.device, dtype=torch.float32).view(1, -1)
    center = center.view(-1, 1)
    diff = torch.abs(classes - center)
    dist = torch.minimum(diff, torch.as_tensor(float(num_beams), device=center.device) - diff)
    sigma = sigma.view(-1, 1).clamp_min(1e-6)
    tau = tau.view(-1, 1).clamp_min(1e-6)
    return -(dist**2) / (2.0 * sigma * sigma * tau)


__all__ = [
    "ABLATIONS",
    "ADAPTER_TYPES",
    "MMWTownGpsV2Backbone",
    "MMWTownGpsV2Model",
    "SceneAdapterV2",
    "SceneAdapterV2Config",
]
