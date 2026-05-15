## Why

当前 TensorBoard 中 beam 指标仍写入通用 `accuracy/*` 分组，而 occlusion 和 position 已经分别写入 `occlusion/*`、`position/*` 分组。用户比较 `beam`、`occlusion`、`position` 单任务 run 时，`accuracy/val` 会把非 beam 任务的占位或诊断性 accuracy 混到同一张卡片里，导致 beam 预测精度图难以解释。

这不是需要新增 beam 模型分支的问题，而是日志指标命名空间不完整。beam 应和 occlusion、position 一样拥有稳定的一等任务指标 tag。

## What Changes

- 为 beam 预测新增 canonical TensorBoard 指标命名空间，例如 `beam/accuracy_train`、`beam/accuracy_val`、`beam/val_atop3`、`beam/val_atop5` 和 `beam/val_adba`。
- 训练日志和指标别名支持 objective-specific beam tag，early stopping 配置中使用 `beam/*` 或旧的 `accuracy/*`/`dba/*` 名称时都能解析到同一内部指标。
- TensorBoard 写入逻辑只在 beam objective 活跃时写入 `beam/*` scalar：`experiment.objective: beam` 写 beam 单任务指标，`multitask` 写 beam 分任务指标；`occlusion` 和 `position` 单任务不得把诊断性 beam accuracy 写进 `beam/*`。
- 旧 `accuracy/*` 和 `dba/val_adba` TensorBoard tag 改为迁移兼容路径：实现阶段通过配置开关保留或恢复，但新推荐图表和测试以 `beam/*` 为准，默认不再依赖通用 accuracy 分组。
- 更新测试和实验说明，明确 `accuracy/*` 是历史通用 tag，不再作为 objective-aware 实验比较的首选入口。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `experiment-workflow`: 训练流程的 TensorBoard 与日志指标必须提供 objective-specific beam 命名空间，并避免通用 accuracy 分组混淆 beam、occlusion 和 position 实验。

## Impact

- 影响 `src/kd_sensing/engine/trainer.py` 的 TensorBoard scalar 写入、metric alias 和相关 history/epoch log 处理。
- 影响 `tests/test_training_io_workflow.py` 等训练日志与 TensorBoard regression tests。
- 影响 README 或训练说明中关于 objective-aware TensorBoard tag 的解释。
- 不改变模型结构、dataset target、loss 计算或已有 checkpoint 格式。
