## MODIFIED Requirements

### Requirement: 恢复训练

训练入口 MUST 在任何可变 run artifact 写出、normalization 拟合/覆盖、模型/optimizer 构建或 optimizer step 前完成 resume path、current schema 与不可变 fingerprint 预检。current checkpoint MUST 加载 primary model、optimizer、scheduler、epoch、selection state 和版本化 `runtime_state`；不符合 current schema 的 checkpoint MUST 拒绝，不提供 legacy migration 或 alias。

#### Scenario: 从 current last checkpoint 恢复

- **WHEN** 用户设置 `training.resume: true` 且 `output.run_name` 指向已有 current run
- **THEN** 系统 MUST 从该运行目录的 `checkpoints/last.pth` 加载并验证 checkpoint
- **AND** 后续训练 MUST 从 checkpoint 中记录的下一轮 epoch 开始

#### Scenario: 恢复输入无效

- **WHEN** checkpoint 路径不存在、缺少 current schema、缺少核心 role/runtime state 或 fingerprint 不一致
- **THEN** 系统 MUST 在写入目标 run artifact 前报告明确错误
- **AND** 系统 MUST 不退化为 fresh run、迁移分支或兼容 alias
