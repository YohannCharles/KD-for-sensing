# 模型任务上下文

`pcpf_temporal_risk_fusion` 是唯一 active research mainline：四个独立 encoder、唯一共享 Temporal Transformer、唯一 `BeamPrototypeBank`、四项拓扑风险和解析概率融合。它不得污染 U0 state dict，也不得读取 CSI/channel/future metadata。

`u_mask_beam_jepa`、`modular_sequence` 和 DeepSense6G T2 分别保留 U0、AMBER-Full/RMBP-MM 与独立数据路线。历史 sparse CSI 只保留 PCPF 直接使用的固定 simulator、codebook、cache、encoder 与 dataset sidecar。

先读 PCPF active spec 与 `openspec/specs/u0-mainline/spec.md`。最小验证：`conda run -n kd_mm_beam pytest tests/test_pcpf_temporal_transformer.py tests/test_pcpf_risk.py tests/test_pcpf_fusion.py tests/test_pcpf_loss.py tests/test_pcpf_stage_freezing.py tests/test_component_registry.py -q`。
