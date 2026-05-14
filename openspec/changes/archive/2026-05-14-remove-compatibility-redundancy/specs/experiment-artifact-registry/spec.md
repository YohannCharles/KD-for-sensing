## MODIFIED Requirements

### Requirement: checkpoint 解析优先级
KD teacher 和评估权重解析 MUST 支持从最佳 checkpoint 归档目录加载匹配 checkpoint。显式传入的绝对路径或评估入口 `--weights` MUST 保持最高优先级；未显式指定时，系统 MUST 查找归档目录中的匹配 checkpoint。归档目录缺失或无匹配时，系统 MUST 抛出清晰错误，且 MUST 不再回退到 legacy `paths.weights_dir / distillation.teacher_model_name` 或旧评估配置路径。

#### Scenario: KD teacher 从归档目录加载
- **WHEN** 用户运行 KD 配置且未显式覆盖 teacher checkpoint 为绝对路径
- **THEN** 系统 MUST 根据配置推导对应 teacher baseline slug
- **AND** 如果归档目录存在该 slug 的最高验证 Top-1 checkpoint，系统 MUST 加载该 checkpoint 作为 frozen teacher
- **AND** `checkpoint_loads` MUST 记录最终加载路径和来源为 registry

#### Scenario: 显式权重路径覆盖 registry
- **WHEN** 用户通过评估入口 `--weights` 或配置中的绝对路径显式指定 checkpoint
- **THEN** 系统 MUST 加载该显式路径
- **AND** 系统 MUST 不用归档目录中的候选替换该显式路径

#### Scenario: registry 缺失时报错
- **WHEN** 归档目录没有匹配当前配置的 checkpoint
- **THEN** 系统 MUST 抛出包含 registry 候选和显式 checkpoint 配置方式的清晰错误
- **AND** 系统 MUST 不尝试 legacy 权重目录 fallback

## REMOVED Requirements

### Requirement: 场景化 legacy 权重路径解析
**Reason**: legacy 权重目录 fallback 会掩盖场景、run name 和模型结构不匹配，并继续维护历史目录约定。
**Migration**: 使用场景隔离的最佳 checkpoint registry，或显式配置 teacher/evaluation checkpoint 绝对路径。

#### Scenario: Scenario 9 KD fallback
- **WHEN** Scenario 9 KD 配置没有可用 registry checkpoint
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 系统 MUST 不从 `outputs/scene9/<teacher_run_name>/checkpoints` 自动解析 fallback

#### Scenario: 不同场景同名 teacher 不冲突
- **WHEN** Scenario 9 和 Scenario 32 都存在同名 teacher run
- **THEN** 系统 MUST 只使用当前场景 registry 或显式路径
- **AND** 系统 MUST 不通过 legacy fallback 目录推断 teacher
