# 数据任务上下文

MMW 只能通过精确绑定且审计通过的 `mmw_id_stratified_block_v1` 运行。trajectory key 为 `(scene_id,cav_id)`，三种天气按 `(scene_id,cav_id,base_frame_index)` 绑定，以 128-base-frame 连续 block 做 70/15/15 分配。当前 seed 0 为 90/19/19 blocks、31,602/6,723/6,855 windows，三个 split 均覆盖 5 个场景和 16 条轨迹。loader 默认只构建 train/validation；归一化与可拟合统计只从 train 拟合，sparse-CSI split bundle 只扫描 train/validation。

DeepSense6G 保持 Scene31--34、四模态和 64 类 future-beam 契约，不使用 MMW protocol。`dataset/`、`outputs/`、cache、日志和 checkpoint 均为本地边界。

先读 `openspec/specs/clean-data-integrity/spec.md` 与 `openspec/specs/mmw-id-stratified-block-protocol/spec.md`。最小验证：`conda run -n kd_mm_beam pytest tests/test_mmw_trajectory_split.py tests/test_train_only_normalization.py tests/test_training_final_test_policy.py tests/test_deepsense6g_dataset.py -q`。
