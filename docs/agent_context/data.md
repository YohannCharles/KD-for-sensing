# 数据任务上下文

MMW 只能通过 `mmw_clean_inner_development_v1` 或 `mmw_trajectory_disjoint_v1` 中一个精确绑定且审计通过的协议运行。clean-inner 审计 train/validation；trajectory-disjoint 进一步审计 train/validation/test 的 trajectory group、场景执行、依赖帧与四模态资源。未授权 test 不得构建，归一化、CSI codebook 和 prototype 统计只从 train 拟合。

DeepSense6G 保持 Scene31--34、四模态和 64 类 future-beam 契约，不使用 MMW protocol。`dataset/`、`outputs/`、cache、日志和 checkpoint 均为本地边界。

先读 `openspec/specs/clean-data-integrity/spec.md` 与 `mmw-trajectory-disjoint-protocol/spec.md`。最小验证：`conda run -n kd_mm_beam pytest tests/test_clean_inner_protocol.py tests/test_mmw_trajectory_split.py tests/test_train_only_normalization.py tests/test_deepsense6g_dataset.py -q`。
