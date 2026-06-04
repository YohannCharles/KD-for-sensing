## MODIFIED Requirements

### Requirement: CSI-only 配置可加载
系统 MUST 提供 CSI-only supervised 配置，使用户能构建 CSI-only primary model 并运行训练或评估。该配置 MUST 不使用 no-KD 或 distillation 命名。

#### Scenario: CSI-only supervised 配置可加载
- **WHEN** 用户加载 CSI-only supervised 配置
- **THEN** 配置 MUST 构建 CSI-only primary model
- **AND** 配置 MUST 不包含 `distillation.type`
- **AND** 配置 MUST 不要求 teacher checkpoint

