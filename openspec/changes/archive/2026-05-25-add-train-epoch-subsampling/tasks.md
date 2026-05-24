## 1. 配置与校验

- [x] 1.1 在默认配置中新增 `training.epoch_subsampling`，默认 `enabled=false`，包含 `fraction`、`num_samples`、`seed`、`rotate_each_epoch` 和 `shuffle` 字段。
- [x] 1.2 实现子采样配置解析与校验，确保 `fraction` 和 `num_samples` 二选一、比例和数量非法时抛出包含 `training.epoch_subsampling` 的清晰错误。
- [x] 1.3 确认现有 `data.dataset.portion` 语义不变，并在代码注释或文档中区分 dataset 缩小与 epoch 子采样。

## 2. Sampler 与 DataLoader 接入

- [x] 2.1 新增可设置 epoch 的 train 子采样 sampler，支持无放回抽样、固定子集、按 epoch 轮换和 `experiment.seed` 默认 seed。
- [x] 2.2 将 sampler 接入 train DataLoader 构建；启用 sampler 时不得同时传入 `shuffle=True`，test/validation DataLoader 必须保持现有完整 split 行为。
- [x] 2.3 为 sampler 提供轻量 metadata helper，返回完整 train 样本数、有效 epoch 样本数、策略、seed、轮换状态和是否退化为 full epoch。

## 3. 训练循环与运行产物

- [x] 3.1 在训练循环每个 epoch 开始前设置 train sampler 的 epoch，保证 resume 后同一绝对 epoch 生成相同样本选择。
- [x] 3.2 在 epoch 日志中记录当前 epoch 是否启用子采样、有效 train 样本数和 sampler epoch。
- [x] 3.3 扩展 `dataloaders_run_metadata`、`throughput_run_metadata` 或等价运行产物，使最终配置/运行元数据记录子采样策略和 full-epoch 退化结果。
- [x] 3.4 确认 checkpoint、early stopping、TensorBoard、`training_outputs.npz` 和验证指标结构保持兼容。

## 4. 测试覆盖

- [x] 4.1 添加配置校验测试，覆盖默认禁用、按比例、按数量、非法比例、非法数量和同时设置两种限制。
- [x] 4.2 添加 sampler 单元测试，覆盖固定子集、epoch 轮换、无放回抽样、seed 可复现和 resume 绝对 epoch 一致性。
- [x] 4.3 添加 DataLoader 构建测试，确认 train split 使用子采样 sampler，test split 不受影响，`len(train_loader)` 随有效样本数和 `drop_last` 正确变化。
- [x] 4.4 添加短训练 smoke test，使用 `conda run -n kd_mm_beam pytest <target> -q` 验证子采样训练能完成 forward、backward、validation、checkpoint 和日志写出。

## 5. 文档与验证

- [x] 5.1 更新 `docs/training_throughput.md` 或 README，给出快速调试示例命令，例如通过 `-o training.epoch_subsampling.enabled=true` 和 `fraction`/`num_samples` 缩短 epoch。
- [x] 5.2 文档说明子采样主要减少 epoch training step，不会缩小 validation/test split，也不会替代 `data.dataset.portion`。
- [x] 5.3 运行相关测试：`conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q`，必要时补充更窄目标测试。
- [x] 5.4 运行训练入口帮助检查：`conda run -n kd_mm_beam kd-sensing-train --help`。
- [x] 5.5 运行 OpenSpec 校验：`openspec validate add-train-epoch-subsampling --strict` 和 `openspec status --change add-train-epoch-subsampling`。
