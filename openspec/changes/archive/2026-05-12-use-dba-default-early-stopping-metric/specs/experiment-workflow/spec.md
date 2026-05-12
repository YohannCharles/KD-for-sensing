## ADDED Requirements

### Requirement: 默认 early stopping 指标使用 DBA
训练工作流 MUST 默认使用验证 DBA/ADBA 作为 early stopping 监控指标。所有默认训练配置、canonical 配置生成结果和配置模板 MUST 将默认 early stopping 指标设置为 `val_adba` 或等价 DBA 别名，并 MUST 使用越大越好的比较方向。默认配置 MUST NOT 使用 `top1_val_acc`、`val_acc` 或其它 Top-1 验证准确率别名作为 early stopping 指标。

#### Scenario: 默认配置记录 DBA early stopping
- **WHEN** 用户使用默认 image、radar、GPS、LiDAR、mmWave 或 fusion 训练配置启动训练
- **THEN** 系统 MUST 在解析后的最终配置中记录 early stopping 监控指标为 `val_adba` 或等价 DBA 别名
- **AND** 系统 MUST 将 early stopping 比较方向记录为越大越好
- **AND** 系统 MUST 不把 `top1_val_acc` 或等价 Top-1 验证准确率别名作为默认 early stopping 指标

#### Scenario: canonical 配置默认使用 DBA
- **WHEN** 开发者生成或读取 canonical 训练配置
- **THEN** canonical 配置 MUST 默认包含 DBA/ADBA early stopping 指标
- **AND** canonical 配置 MUST 不把 Top-1 验证准确率作为默认 early stopping 指标

#### Scenario: 显式覆盖 early stopping 指标
- **WHEN** 用户在训练配置或命令行覆盖中显式设置 early stopping 指标为 Top-1、loss 或其它受支持指标
- **THEN** 系统 MUST 使用用户显式指定的指标和比较方向
- **AND** 该覆盖 MUST 不改变项目默认配置继续使用 DBA/ADBA 的要求

### Requirement: 训练循环按配置指标执行 early stopping
训练循环 MUST 从每个 epoch 的验证标量中解析配置的 early stopping 指标，并基于该指标更新最佳值、patience 计数和默认最佳 checkpoint。DBA/ADBA 和准确率类指标 MUST 按越大越好判断 improvement；loss 类指标 MUST 按越小越好判断 improvement。

#### Scenario: DBA improvement 重置 patience
- **WHEN** early stopping 指标为 `val_adba` 且当前 epoch 的 `val_adba` 相比历史最佳值提升超过 `training.min_delta`
- **THEN** 系统 MUST 更新最佳 early stopping 值和最佳 epoch
- **AND** 系统 MUST 将 `epochs_without_improvement` 重置为 0
- **AND** 系统 MUST 保存默认最佳 checkpoint

#### Scenario: DBA 未提升累计 patience
- **WHEN** early stopping 指标为 `val_adba` 且当前 epoch 的 `val_adba` 未提升超过 `training.min_delta`
- **THEN** 系统 MUST 累加 `epochs_without_improvement`
- **AND** 当 `training.use_early_stopping` 启用且累计值达到 `training.patience` 时，系统 MUST 停止训练

#### Scenario: 缺失 DBA 指标时报错
- **WHEN** 默认 early stopping 指标为 DBA/ADBA 但验证结果没有产出可解析的 DBA/ADBA 标量
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出缺失的 early stopping 指标，并提示用户补齐 DBA 指标或显式配置其它受支持指标

### Requirement: early stopping metadata 可复现
训练产物 MUST 记录实际使用的 early stopping 指标、比较方向、最佳值、最佳 epoch 和未提升 epoch 计数。恢复训练 MUST 优先使用这些通用 metadata 继续 early stopping 状态；历史 checkpoint 缺少通用 metadata 时，系统 MUST 使用兼容路径恢复已有 loss 或 Top-1 相关状态。

#### Scenario: checkpoint 记录 early stopping 状态
- **WHEN** 训练完成至少一个 epoch 并保存 `last.pth`
- **THEN** checkpoint metadata MUST 包含实际 early stopping 指标、比较方向、最佳值、最佳 epoch 和 `epochs_without_improvement`
- **AND** 运行日志或最终配置 MUST 能追溯本次训练使用的 early stopping 指标

#### Scenario: 恢复 DBA early stopping 状态
- **WHEN** 用户从包含通用 early stopping metadata 的 checkpoint 恢复训练
- **THEN** 系统 MUST 恢复 DBA/ADBA 的最佳值、最佳 epoch 和 `epochs_without_improvement`
- **AND** 后续 early stopping 判断 MUST 延续恢复前的指标和比较方向

#### Scenario: 兼容历史 checkpoint
- **WHEN** 用户从缺少通用 early stopping metadata 的历史 checkpoint 恢复训练
- **THEN** 系统 MUST 尽可能从历史 `best_val_loss`、`best_val_top1` 或等价字段恢复状态
- **AND** 系统 MUST 不因缺少新 metadata 而拒绝恢复历史 checkpoint
