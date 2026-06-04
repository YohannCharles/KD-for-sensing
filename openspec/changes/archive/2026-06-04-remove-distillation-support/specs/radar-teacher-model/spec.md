## MODIFIED Requirements

### Requirement: RadarTeacher 蒸馏角色配置
系统 MUST 不再支持在 radar-only KD 配置中将 `radar_teacher` 作为 frozen teacher。Radar 强模型只能作为 supervised primary model、评估模型或可被显式权重评估的 checkpoint 来源。

#### Scenario: 构建 radar strong 模型
- **WHEN** 配置指定 radar strong primary model
- **THEN** 系统 MUST 通过模型注册表构建 `RadarModalityNet`
- **AND** 训练流程 MUST 将其作为被优化的 primary model

#### Scenario: 旧 radar KD 模型配置被拒绝
- **WHEN** 配置同时指定 frozen radar teacher、radar student 和 `logits_kd` 或 `rkd`
- **THEN** 配置加载 MUST 失败
- **AND** 系统 MUST 不执行 teacher/student forward

