## MODIFIED Requirements

### Requirement: 恢复训练
训练入口 MUST 让 `training.resume` fail-closed 生效，并在任何可变 run artifact 写出、normalization 拟合/覆盖、模型/optimizer 构建或 optimizer step 之前完成恢复路径与 resume role schema 预检。current schema 恢复 MUST 加载 student/primary 模型、optimizer、scheduler、已完成 epoch、early-stopping/selection state 和版本化 `runtime_state`；历史 schema MUST 只通过显式 legacy migration 分支恢复并记录非等价 provenance。恢复训练 MUST 继续使用统一输出目录、checkpoint 保存和 early stopping 语义。

#### Scenario: 从 last checkpoint 恢复
- **WHEN** 用户设置 `training.resume: true` 且 `output.run_name` 指向已有运行目录
- **THEN** 系统 MUST 从该运行目录的 `checkpoints/last.pth` 加载 checkpoint
- **AND** 后续训练 MUST 从 checkpoint 中记录的下一轮 epoch 开始
- **AND** optimizer、scheduler、early-stopping/selection state 和 current schema `runtime_state` MUST 被恢复

#### Scenario: 从显式路径跨 run 恢复
- **WHEN** 用户将 `training.resume` 设置为另一 run 中的 checkpoint 文件路径
- **THEN** 系统 MUST 记录源 checkpoint、源 run、目标 run 和是否跨 run
- **AND** 系统 MUST 只允许目标 output identity 等 allowlist 字段变化
- **AND** 源 run 的 checkpoint 和 artifact MUST 不被覆盖或移动

#### Scenario: 恢复路径不存在
- **WHEN** 用户启用 resume 但目标 checkpoint 不存在
- **THEN** 系统 MUST 在创建/覆盖 `run_status.json`、resolved/final config、normalization artifact、模型或 optimizer 前抛出明确错误
- **AND** 错误信息 MUST 包含尝试恢复的 checkpoint 路径
- **AND** 系统 MUST 不退化为从头训练

#### Scenario: Resume role 缺少核心字段
- **WHEN** resume checkpoint 缺少 `optimizer`、`scheduler` 或 `epoch` 任一字段，或者当前运行启用了 scheduler 但 checkpoint 的 scheduler state 为 `null`
- **THEN** 系统 MUST 在第一个 optimizer step 前拒绝恢复
- **AND** 错误 MUST 包含 checkpoint 路径、`resume` role 和缺失或不兼容字段
- **AND** `training.start_epoch` MUST 不得替代 resume role 缺失的 `epoch`

#### Scenario: Current schema 缺少运行时状态
- **WHEN** checkpoint 声明 current schema version 但 `runtime_state` 缺少 RNG、DataLoader generator、GradScaler、extension state、history 或 epoch logs 任一必需部分
- **THEN** 系统 MUST 拒绝 exact resume
- **AND** 系统 MUST 不把 current schema 自动降级为 legacy checkpoint

#### Scenario: 不可变恢复契约一致
- **WHEN** 当前配置、实际 split 和训练 normalization artifact 与 checkpoint 中的 canonical fingerprint 一致，且差异只位于封闭 allowlist
- **THEN** 系统 MUST 允许恢复
- **AND** allowlist MUST 只覆盖 resume 字段、合法增加的总 epoch、目标 output identity、进度/TensorBoard、日志频率和显式 timing profile 等不改变训练语义的字段
- **AND** checkpoint load provenance MUST 记录三个 fingerprint 与所有已放行差异

#### Scenario: 不可变恢复契约不一致
- **WHEN** model/objective/loss、optimizer/scheduler、AMP、seed、DataLoader/sampler、split sample identity 或 normalization fingerprint 与 checkpoint 不一致
- **THEN** 系统 MUST 在覆盖源/目标 normalization 或 resolved/final config 前拒绝恢复
- **AND** 错误 MUST 提供字段路径、checkpoint 值、当前值和 fingerprint，而不能只报告 hash 不同
- **AND** 用户 MUST 不能通过自定义通配 allowlist 绕过该检查

#### Scenario: 零剩余 epoch 的恢复
- **WHEN** resume checkpoint 的 next epoch 不小于配置的 `training.epochs`
- **THEN** 系统 MUST 执行零个训练 epoch，并从恢复的 selection catalog 或 resume 文件解析实际 final-test checkpoint
- **AND** 系统 MUST 不假设目标 run 新生成了 `best.pth` 或 `last.pth`
- **AND** 无可验证候选时系统 MUST 清晰失败

#### Scenario: 历史 checkpoint 显式迁移
- **WHEN** 历史 checkpoint 没有 current schema version 但包含完整模型、optimizer、scheduler 和 epoch 核心状态
- **THEN** 系统 MUST 通过独立 legacy migration 分支恢复可用状态
- **AND** checkpoint load provenance MUST 记录 legacy 版本、迁移 warning 和 `trajectory_equivalence: false`
- **AND** 只有该分支 MAY 在缺少 `best_val_loss` 时把历史 `test_loss` 解释为旧 validation-loss alias

