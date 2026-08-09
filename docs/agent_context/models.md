# 模型任务上下文

`pcpf_temporal_risk_fusion` 为 sensing-guided local beam probing 提供冻结预测与不确定性：四个独立 encoder、唯一共享 Temporal Transformer、唯一 `BeamPrototypeBank` 和四项拓扑风险。`fused_probability[64]` 无参数派生 MAP、circular mean、beam variance/spread 与 entropy；K=7 evaluator 提供固定 Local、posterior-mass Adaptive Local 和 Posterior Top-K。Direct Router、CUAF、nested/R0--R7 已删除。模型不得污染 U0 state dict，也不得读取未来 channel、beam power 或标签信息。

`u_mask_beam_jepa`、`modular_sequence` 和 DeepSense6G T2 分别保留 U0、AMBER-Full/RMBP-MM 与独立数据路线。历史 sparse CSI 只保留 PCPF 直接使用的固定 simulator、codebook、cache、encoder 与 dataset sidecar。

先读 current PCPF spec、当前 probing change 与 `openspec/specs/u0-mainline/spec.md`。最小验证：`conda run -n kd_mm_beam pytest tests/test_beam_posterior.py tests/test_beam_probe_diagnostic.py tests/test_pcpf_temporal_transformer.py tests/test_pcpf_risk.py tests/test_pcpf_fusion.py tests/test_pcpf_loss.py tests/test_pcpf_stage_freezing.py tests/test_component_registry.py -q`。
