## ADDED Requirements

### Requirement: Auxiliary heads 可作为 primary objective
CLS-token Transformer fusion MUST 支持将 `occlusion_head` 和 `position_head` 作为 primary objective 输出使用。配置为 `occlusion` 或 `position` objective 时，模型 MUST 启用对应 head，并保持主 beam logits 输出兼容。

#### Scenario: occlusion primary output
- **WHEN** `experiment.objective` 为 `occlusion` 且模型类型为 `cls_token_transformer_fusion`
- **THEN** 模型配置 MUST 启用 `auxiliary_heads.occlusion`
- **AND** forward 输出 MUST 包含形状为 `[B, H]` 的 `occlusion_logits`

#### Scenario: position primary output
- **WHEN** `experiment.objective` 为 `position` 且模型类型为 `cls_token_transformer_fusion`
- **THEN** 模型配置 MUST 启用 `auxiliary_heads.position`
- **AND** forward 输出 MUST 包含形状为 `[B, H, 2]` 的 `position`

#### Scenario: multitask primary outputs
- **WHEN** `experiment.objective` 为 `multitask` 且模型类型为 `cls_token_transformer_fusion`
- **THEN** 模型配置 MUST 启用 beam、occlusion 和 position 所需输出
- **AND** forward 输出 MUST 同时提供 beam logits、`occlusion_logits` 和 `position`

### Requirement: Objective head 校验
配置校验 MUST 在训练开始前确认当前 objective 所需的模型 head 可用。模型不支持当前 objective 时，系统 MUST 拒绝配置并给出可执行的修复提示。

#### Scenario: occlusion head 未启用
- **WHEN** 配置设置 `experiment.objective: occlusion` 但 `model.student.auxiliary_heads.occlusion` 未启用
- **THEN** 系统 MUST 拒绝加载配置
- **AND** 错误信息 MUST 提示启用 `model.student.auxiliary_heads.occlusion=true`

#### Scenario: position head 未启用
- **WHEN** 配置设置 `experiment.objective: position` 但 `model.student.auxiliary_heads.position` 未启用
- **THEN** 系统 MUST 拒绝加载配置
- **AND** 错误信息 MUST 提示启用 `model.student.auxiliary_heads.position=true`
