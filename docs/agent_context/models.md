# 模型任务上下文

只维护 `u_mask_beam_jepa` 的 T2/S1 路径，以及 `modular_sequence` 的 AMBER-Full、RMBP-MM baseline。四模态固定为 image、radar、gps、lidar；BPA、CMA 和 same-model consistency 仅在 active T2 protocol 中保留。

先读 `openspec/specs/t2-baseline-surface/spec.md`、`openspec/specs/u-mask-beam-jepa/spec.md` 和 active T2 change。不要新增任何不属于这四方法的模型或兼容分支。

最小验证：`conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py tests/test_s1_temporal_superset_training.py -q`。
