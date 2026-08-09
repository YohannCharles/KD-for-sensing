# 模型任务上下文

`four_modal_topology_predictor` 是 probing 主线唯一模型：`image/radar/gps/lidar` 四个 encoder、唯一共享 Temporal Transformer、唯一 `BeamPrototypeBank`，一次训练完成。`fused_probability[64]` 无参数派生 MAP、circular mean、beam variance/spread 与 entropy；模型不含 CSI、risk head、learned fusion、stage transfer 或 train-only probing likelihood。

`u_mask_beam_jepa`、`modular_sequence` 和 DeepSense6G T2 分别保留 U0、AMBER-Full/RMBP-MM 与独立数据路线。

先读 `openspec/specs/four-modal-topology-predictor/spec.md`、当前 probing change 与 `openspec/specs/u0-mainline/spec.md`。最小验证：`conda run -n kd_mm_beam pytest tests/test_four_modal_topology_predictor.py tests/test_beam_posterior.py tests/test_beam_probe_diagnostic.py tests/test_component_registry.py -q`。
