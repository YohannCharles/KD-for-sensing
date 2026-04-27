## MODIFIED Requirements

### Requirement: 可扩展模型和模态
新增 teacher、student、backbone、head、radar 或 fusion 模型时，开发者 MUST 能通过新增模块和注册名称扩展系统，而不需要复制训练脚本或修改训练循环主体。

#### Scenario: 新增 image-only student
- **WHEN** 开发者实现并注册一个新的 image-only student 模型
- **THEN** 用户 MUST 能在配置中选择该模型，并复用现有 image-only 训练流程

#### Scenario: 新增多模态 fusion 模型
- **WHEN** 开发者实现并注册一个新的 image+radar fusion 模型
- **THEN** 用户 MUST 能在配置中选择该模型，并复用现有 fusion 训练流程

#### Scenario: 新增 radar-only teacher
- **WHEN** 开发者实现并注册一个新的 radar-only teacher 模型
- **THEN** 用户 MUST 能在配置中选择该模型，并复用 radar-only 训练和评估流程
- **AND** 模型 MUST 保持统一的 `(pred, features, output_features)` 输出约定
