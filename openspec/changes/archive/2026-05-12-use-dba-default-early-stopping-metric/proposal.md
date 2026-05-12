## Why

当前训练与实验比较已经将 DBA/ADBA 作为论文核心指标之一，但默认早停指标仍容易沿用 Top-1 验证准确率相关假设，导致默认 checkpoint 与停止时机未必服务于 beam 距离质量。将默认早停指标统一切换为 DBA，可以让默认训练流程更贴近项目当前的评估目标。

## What Changes

- 将所有默认早停监控指标改为验证 DBA/ADBA，而不是 `top1_val_acc` 或等价 Top-1 验证准确率别名。
- 在训练配置默认值、canonical 配置生成和现有默认 YAML 配置中统一早停指标名称。
- 训练循环按配置解析早停指标，默认以 DBA 越大越好进行 improvement 判断，并继续允许显式配置覆盖为其他指标。
- 保存和恢复 early stopping 相关 checkpoint metadata 时记录实际监控指标，避免历史 Top-1 字段继续被默认早停路径依赖。
- 更新测试与文档，覆盖默认 DBA 早停、显式 Top-1 覆盖和无 DBA 指标时的清晰错误。

## Capabilities

### New Capabilities

### Modified Capabilities
- `experiment-workflow`: 默认训练工作流的 early stopping 监控指标改为 DBA/ADBA，并要求训练产物记录实际 early stopping 指标。

## Impact

- 影响训练默认配置：`src/kd_sensing/config/defaults.py`、`src/kd_sensing/config/canonical.py` 和 `configs/**/*.yaml` 中的 training early stopping 字段。
- 影响训练循环：`src/kd_sensing/engine/trainer.py` 的 improvement 判断、patience 计数、checkpoint metadata 和恢复逻辑。
- 影响测试：配置默认值测试、训练 I/O workflow 测试，以及覆盖 metric alias 和早停方向的新增单元测试。
- 影响文档：README 中默认 early stopping 指标和 DBA/ADBA 指标说明需要同步。
