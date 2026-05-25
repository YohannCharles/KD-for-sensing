## Why

Raymobtime s008 的现有结果显示：单任务 LiDAR/CIL 在 beam、LOS、link 上都较强，但 selection multitask 模型显著提升 LOS/link 的同时牺牲 beam top1，这很像多任务模态失衡或任务冲突。现在需要先用可复现实验排除 early stopping、loss 尺度、seed、checkpoint 选择和诊断产物缺失等代码/参数混杂，再决定是否把 s009 作为跨场景验证。

## What Changes

- 新增 Raymobtime s008 模态失衡确认实验方案，覆盖多 seed、loss 权重消融、任务组合消融、checkpoint 选择消融和诊断产物补齐。
- 明确“确认模态失衡”的判定标准：单任务优势、融合负迁移、任务间优化冲突、gate/drop/gradient 证据必须相互支持。
- 将 s009 放入第二阶段外部验证：只有当 s008 诊断矩阵能稳定排除训练/参数问题后，才复刻最小矩阵到 s009。
- 产物统一写入 ignored 输出目录，避免把训练日志、checkpoint、cache 或临时报告纳入源码变更。
- 不引入新的训练入口，不绕过当前 `src/kd_sensing` 包结构和既有 Raymobtime s008 当前快照语义。

## Capabilities

### New Capabilities

- `raymobtime-modality-imbalance-diagnosis`: 定义 Raymobtime s008 多任务模态失衡诊断的实验矩阵、判定标准、输出产物和 s009 外部验证门槛。

### Modified Capabilities

- 无。

## Impact

- 影响实验配置、运行计划、结果汇总和分析报告；主要复用现有 `scripts/train.py`、评估入口和 Raymobtime s008 模态失衡分析能力。
- 不改变 Raymobtime s008 dataset、模型输出、正式 metrics 契约或已有训练 CLI。
- 训练和分析命令必须通过 `conda run -n kd_mm_beam ...` 执行。
- 运行产物位于 `outputs/raymobtime_s008/...`、`logs/...` 或等价 ignored 目录；不提交本地数据、checkpoint、cache 或 TensorBoard 日志。
