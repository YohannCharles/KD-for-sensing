## MODIFIED Requirements

### Requirement: 场景隔离的最佳 checkpoint registry
最佳 checkpoint registry MUST 按 DeepSense6G 场景隔离。默认 registry 目录 MUST 位于当前场景输出分组下，例如 `outputs/scene9/best_checkpoints/`、`outputs/scene31/best_checkpoints/` 和 `outputs/scene32/best_checkpoints/`。

#### Scenario: Scenario 9 registry 写入 scene9
- **WHEN** 用户运行 Scenario 9 teacher no-KD 训练并产生新的最高验证 Top-1 checkpoint
- **THEN** 系统 MUST 将归档 checkpoint 写入 `outputs/scene9/best_checkpoints/`
- **AND** metadata sidecar MUST 记录 `scene_id: 9` 和 `scene_slug: scene9`

#### Scenario: 默认 Scenario 31 registry 不复用其它场景
- **WHEN** 用户运行默认 Scenario 31 KD 配置且未显式指定绝对 teacher checkpoint
- **THEN** 系统 MUST 优先查找 `outputs/scene31/best_checkpoints/`
- **AND** 系统不得默认加载 `outputs/scene9/best_checkpoints/` 或 `outputs/scene32/best_checkpoints/` 中同 slug 的 checkpoint

#### Scenario: Scenario 32 registry 不复用 scene31
- **WHEN** 用户运行显式 Scenario 32 KD 配置且未显式指定绝对 teacher checkpoint
- **THEN** 系统 MUST 优先查找 `outputs/scene32/best_checkpoints/`
- **AND** 系统不得默认加载 `outputs/scene31/best_checkpoints/` 中同 slug 的 checkpoint

#### Scenario: 显式绝对 checkpoint 仍最高优先级
- **WHEN** 用户通过绝对路径显式指定 teacher checkpoint 或评估权重
- **THEN** 系统 MUST 使用该显式路径
- **AND** 场景 registry 不得替换该路径

### Requirement: Teacher reliability registry artifact
实验产物体系 MUST 支持 teacher reliability registry。该 registry MUST 按场景隔离，引用 teacher checkpoint 和指标来源，并能被 Stage 2/3 配置稳定解析。

#### Scenario: 默认 Scene31 teacher registry 写入 scene31 输出组
- **WHEN** 用户为默认 Scenario 31 构建 teacher reliability registry
- **THEN** 默认输出路径 MUST 位于 `outputs/scene31/`
- **AND** registry MUST 记录 `scene_id: 31` 或等价 scene metadata

#### Scenario: Scene32 teacher registry 写入 scene32 输出组
- **WHEN** 用户为显式 Scenario 32 构建 teacher reliability registry
- **THEN** 默认输出路径 MUST 位于 `outputs/scene32/`
- **AND** registry MUST 记录 `scene_id: 32` 或等价 scene metadata

#### Scenario: registry 引用 checkpoint metadata
- **WHEN** teacher checkpoint 有 checkpoint registry sidecar metadata
- **THEN** teacher reliability registry MUST 记录 checkpoint 路径
- **AND** registry MUST 保留可追溯到源 run_dir、epoch 和验证 Top-1 的 metadata 或引用

#### Scenario: Stage 2 解析 registry 路径
- **WHEN** Stage 2 配置提供相对 teacher registry 路径
- **THEN** 系统 MUST 按项目根目录解析该路径
- **AND** 如果文件不存在，错误信息 MUST 包含解析后的绝对路径
