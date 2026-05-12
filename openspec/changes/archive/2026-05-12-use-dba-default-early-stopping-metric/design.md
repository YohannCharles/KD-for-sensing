## Context

训练循环已经在每个验证 epoch 聚合 `val_acc`、`val_atop3`、`val_atop5` 和 `val_adba`，TensorBoard 也会写入 `dba/val_adba`。当前默认配置只有 `use_early_stopping`、`patience` 和 `min_delta` 等控制项，训练循环的 improvement、patience 计数和 checkpoint metadata 仍带有验证 loss 或 Top-1 相关假设；这会让默认停止时机与项目更关注的 DBA/ADBA 指标不一致。

## Goals / Non-Goals

**Goals:**
- 所有默认训练配置、canonical 配置生成和配置模板都使用 DBA/ADBA 作为默认 early stopping 监控指标。
- 训练循环支持按配置解析监控指标，并以 `val_adba` 默认执行越大越好的 improvement 判断。
- checkpoint metadata、恢复训练和日志中记录实际 early stopping 指标、方向和最佳值，便于复现实验。
- 保留显式覆盖能力，使需要 Top-1 或 loss 早停的实验可以通过配置 opt in。

**Non-Goals:**
- 不改变 DBA 的计算公式、Top-3 beam 使用方式或验证阶段 per-slot 指标聚合方式。
- 不强制重命名现有 `best_top1.pth` 或历史运行产物。
- 不重新定义 teacher registry 的 prior 计算权重。

## Decisions

1. 使用 `training.early_stopping_metric` 和 `training.early_stopping_mode` 作为显式配置字段。
   - 默认值为 `early_stopping_metric: val_adba` 和 `early_stopping_mode: max`，与现有扁平 `training` 配置风格一致。
   - 备选方案是新增嵌套 `training.early_stopping.metric`，但会让所有 YAML 和命令行覆盖路径产生更大迁移成本。

2. 训练循环从 epoch 标量表中解析 early stopping metric。
   - `val_adba` 直接来自现有 `_aggregate_validation_metrics()`，不重复实现 DBA。
   - 保留别名兼容：`dba`、`val_dba`、`val_adba` 解析到验证 ADBA；`top1_val_acc`、`val_acc` 解析到验证 Top-1；`val_loss` 解析到验证 loss。
   - 默认配置不得再写入 `top1_val_acc` 或等价 Top-1 别名。

3. 由 metric direction 控制 improvement 判断。
   - `early_stopping_mode: max` 使用 `current > best + min_delta`，用于 DBA 和准确率。
   - `early_stopping_mode: min` 使用 `current < best - min_delta`，用于 loss。
   - 如果 mode 未显式提供，训练循环可以根据 metric alias 推断；配置默认仍显式写 `max`，降低阅读歧义。

4. checkpoint metadata 记录通用 early stopping 状态。
   - `last.pth` 中新增或统一保存 `early_stopping_metric`、`early_stopping_mode`、`best_early_stopping_value`、`best_early_stopping_epoch` 和 `epochs_without_improvement`。
   - 恢复训练优先读取通用字段；缺少通用字段的历史 checkpoint 继续从 `best_val_loss`、`best_val_top1` 或现有字段做兼容恢复。
   - 可以继续保存 `best.pth` 作为默认最佳 checkpoint，但其语义应来自 configured early stopping metric；如果额外保存 `best_dba.pth`，不得破坏历史 `best_top1.pth` 的显式用途。

## Risks / Trade-offs

- [Risk] 历史实验默认早停语义与新实验不同，横向比较可能混淆。→ Mitigation：在 README 和运行 metadata 中明确记录 early stopping metric，并在结果表中区分旧 run。
- [Risk] 某些验证路径没有产出 DBA，默认 DBA 早停会失败。→ Mitigation：训练路径缺少 `val_adba` 时抛出清晰错误，提示配置可显式切回 `val_loss` 或补齐 DBA 指标。
- [Risk] `min_delta` 对 DBA 的数值尺度与 loss 不同。→ Mitigation：保持默认 `0.0001` 以最小化配置变动，但在文档中说明其含义已对应 DBA 增益阈值。
- [Risk] teacher registry 或分析工具仍偏好 `best_top1.pth`。→ Mitigation：本 change 只改变 early stopping 默认指标；依赖 Top-1 checkpoint 的分析配置必须显式请求 `best_top1`，不能隐式代表默认早停。
