## ADDED Requirements

### Requirement: 场景隔离的最佳 checkpoint registry
最佳 checkpoint registry MUST 按 DeepSense6G 场景隔离。默认 registry 目录 MUST 位于当前场景输出分组下，例如 `outputs/scene9/best_checkpoints/` 和 `outputs/scene32/best_checkpoints/`。

#### Scenario: Scenario 9 registry 写入 scene9
- **WHEN** 用户运行 Scenario 9 teacher no-KD 训练并产生新的最高验证 Top-1 checkpoint
- **THEN** 系统 MUST 将归档 checkpoint 写入 `outputs/scene9/best_checkpoints/`
- **AND** metadata sidecar MUST 记录 `scene_id: 9` 和 `scene_slug: scene9`

#### Scenario: Scenario 32 registry 不复用 scene9
- **WHEN** 用户运行 Scenario 32 KD 配置且未显式指定绝对 teacher checkpoint
- **THEN** 系统 MUST 优先查找 `outputs/scene32/best_checkpoints/`
- **AND** 系统不得默认加载 `outputs/scene9/best_checkpoints/` 中同 slug 的 checkpoint

#### Scenario: 显式绝对 checkpoint 仍最高优先级
- **WHEN** 用户通过绝对路径显式指定 teacher checkpoint 或评估权重
- **THEN** 系统 MUST 使用该显式路径
- **AND** 场景 registry 不得替换该路径

### Requirement: 场景化 legacy 权重路径解析
KD teacher fallback 解析 MUST 支持场景化 legacy 权重目录。canonical KD 配置的 `paths.weights_dir` MUST 指向当前场景下的 teacher 运行目录，Scenario 9 当前训练结果 MUST 使用 `outputs/scene9/<teacher_run_name>/checkpoints`。

#### Scenario: Scenario 9 KD fallback
- **WHEN** Scenario 9 KD 配置没有可用 registry checkpoint
- **THEN** 系统 MUST 尝试从 `outputs/scene9/<teacher_run_name>/checkpoints` 解析 teacher 权重
- **AND** 错误信息 MUST 同时包含尝试过的 registry 和 fallback 候选

#### Scenario: 不同场景同名 teacher 不冲突
- **WHEN** Scenario 9 和 Scenario 32 都存在同名 teacher run
- **THEN** Scenario 32 KD 配置 MUST 只把 Scenario 32 的 teacher 运行目录作为默认 fallback
- **AND** Scenario 9 的同名 teacher 运行目录不得被默认使用
