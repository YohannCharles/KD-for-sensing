# 模型任务上下文

`four_modal_topology_predictor` 是 probing 主线唯一模型：`image/radar/gps/lidar` 四个 encoder、唯一共享 Temporal Transformer、唯一 `BeamPrototypeBank`，一次训练完成。active change 保留 masked mean/static reliability 诊断，并新增标准 `masked_feature_mlp` backbone：单模态与融合特征查询同一Bank，availability显式进入融合MLP。模型不含 CSI、risk head、显式样本级模态gate、stage transfer 或 train-only probing likelihood。

`u_mask_beam_jepa`、`modular_sequence` 和 DeepSense6G T2 分别保留 U0、AMBER-Full/RMBP-MM 与独立数据路线。

先读 `openspec/specs/four-modal-topology-predictor/spec.md`、当前 probing change 与 `openspec/specs/u0-mainline/spec.md`。最小验证：`conda run -n kd_mm_beam pytest tests/test_four_modal_topology_predictor.py tests/test_beam_posterior.py tests/test_beam_probe_diagnostic.py tests/test_component_registry.py -q`。
