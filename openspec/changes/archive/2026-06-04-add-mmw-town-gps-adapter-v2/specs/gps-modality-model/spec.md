## ADDED Requirements

### Requirement: MMW Town GPS v2 model registration
系统 MUST 提供显式 opt-in 的 MMW Town GPS v2 模型构建能力。该能力 MUST 与既有 `gps_teacher`、`gps_student` 序列模型分离，并 MUST 通过模型注册或 runner 内部构建入口支持 GPS MLP backbone、SceneAdapterV2 和 residual logits 组合。

#### Scenario: 既有 GPS teacher/student 默认不变
- **WHEN** 用户运行现有 GPS-only `gps_teacher` 或 `gps_student` 配置
- **THEN** 系统 MUST 继续接收 `[B, T, 3]` GPS-Rel-Polar 输入
- **AND** 系统 MUST NOT 要求 MMW Town GPS v2 feature、scene_id、theta 或 branch_id

#### Scenario: 构建 MMW Town GPS v2 模型
- **WHEN** 用户通过 v2 配置选择 MMW Town GPS v2 model
- **THEN** 系统 MUST 构建轻量 MLP GPS backbone
- **AND** 系统 MUST 按配置构建 SceneAdapterV2 或 v1 baseline adapter
- **AND** forward 输出 MUST 至少包含 `logits`、`residual_logits`、可用的 `geo_logits` 和 adapter diagnostics

### Requirement: MMW Town GPS v2 feature validation
MMW Town GPS v2 模型 MUST 校验输入 feature 维度、scene id、num_beams 和 adapter 类型。非法配置 MUST 抛出包含字段名和可执行修复提示的错误。

#### Scenario: feature 维度不匹配
- **WHEN** v2 模型接收的 GPS feature 最后一维不等于配置声明的 input_dim
- **THEN** 系统 MUST 抛出配置或运行时错误
- **AND** 错误信息 MUST 包含实际维度、期望维度和 `model.input_dim`

#### Scenario: scene id 超出范围
- **WHEN** v2 adapter forward 收到超出已注册 scene 数的 scene_id
- **THEN** 系统 MUST 拒绝 forward
- **AND** 错误信息 MUST 包含 scene_id、num_scenes 和当前 scene mapping metadata
