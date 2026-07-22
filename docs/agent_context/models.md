# 模型任务上下文

维护 `u_mask_beam_jepa` 的 T2/S1、BCACL U2/CMSBL 路径，以及 `modular_sequence` 的 AMBER-Full、RMBP-MM baseline。四模态固定为 image、radar、gps、lidar；CMSBL 只改变训练 objective，不改变 T2 推理。

先读 `openspec/specs/t2-baseline-surface/spec.md`、`openspec/specs/u-mask-beam-jepa/spec.md` 和 active CMSBL change。不要恢复 PCER、PGCD、动态 Router、BCACL U3--U5 或其他兼容分支。

最小验证：`conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py tests/test_bcacl.py tests/test_cmsbl.py tests/test_s1_temporal_superset_training.py -q`。
