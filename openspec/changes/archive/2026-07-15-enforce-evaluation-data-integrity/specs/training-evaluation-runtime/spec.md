## ADDED Requirements

### Requirement: Model selection 与 final test 隔离
训练 runtime MUST 只使用独立 validation split 执行 epoch validation、scheduler、checkpoint 选择和 early stopping。系统 MUST NOT 在 validation 缺失时回退到 test。

#### Scenario: Early stopping 缺少 validation
- **WHEN** resolved config 启用 early stopping 或基于验证指标的 best-checkpoint 选择，但 dataloader 没有独立 validation
- **THEN** 训练 MUST 在第一个 optimizer step 前失败
- **AND** 错误 MUST 提示提供独立 validation 或改用显式 fixed-epoch/no-selection

#### Scenario: Fixed epoch 无 validation
- **WHEN** resolved config 显式使用 fixed epoch、关闭 model selection 且没有 validation
- **THEN** trainer MUST 跳过逐轮 validation、验证 scheduler 和 best-checkpoint 选择
- **AND** final evaluation MUST 使用 `last.pth` 或显式指定 checkpoint
- **AND** test loader MUST 不在训练循环中被迭代

#### Scenario: 独立 validation 正常选模
- **WHEN** dataloader 提供独立 validation 且 resolved config 启用 model selection
- **THEN** trainer MUST 只用 validation metrics 更新 best checkpoint 和 early stopping
- **AND** final test MUST 仅由显式 final evaluation 消费

### Requirement: Validation loss 按有效观测加权
共享 evaluation pass MUST 按每个 batch 的有效 sample 或 token 数聚合 loss，MUST NOT 计算未加权的 batch mean 平均值。

#### Scenario: 最后一个 batch 较小
- **WHEN** validation dataset 的最后一个 batch 小于其它 batch
- **THEN** reported validation loss MUST 等于所有有效观测 loss 总和除以有效观测总数
- **AND** 它 MUST 不因 batch 分组方式不同而改变

#### Scenario: 任务具有有效 token mask
- **WHEN** objective loss 只对部分 token、target 或样本有效
- **THEN** evaluation pass MUST 使用该 objective 的有效计数作为分母
- **AND** 零有效计数 MUST 被清晰拒绝或报告为 unavailable
