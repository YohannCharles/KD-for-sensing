## ADDED Requirements

### Requirement: GPS 模型可导出 coarse anchor
GPS 模型系统 MUST 支持显式 opt-in 的 coarse anchor export profile。启用该 profile 时，GPS encoder 或 GPS-only 模型 MUST 能输出 coarse anchor 字段；未启用时现有 GPS teacher/student 契约 MUST 保持兼容。

#### Scenario: GPS teacher/student 默认契约不变
- **WHEN** 用户运行现有 GPS teacher 或 GPS student no-KD 配置且未启用 coarse anchor export
- **THEN** 模型 MUST 继续输出既有 beam logits、input features 和 output features
- **AND** 系统 MUST NOT 要求 coarse label、coarse loss 或 GPS anchor metadata

#### Scenario: 启用 GPS coarse anchor export
- **WHEN** 用户配置 GPS 模型 `coarse_anchor.enabled=true`
- **THEN** 模型或训练 wrapper MUST 输出 `coarse_logits`、`center_beam`、`confidence` 和可选 `beam_scores`
- **AND** 输出形状 MUST 满足 `gps-coarse-anchor-prediction` 能力定义的 anchor 契约
- **AND** run metadata MUST 记录 anchor source 为 `gps_neural_coarse` 或等价配置值

#### Scenario: GPS coarse head 参数可配置
- **WHEN** 用户构建 GPS coarse anchor 模型
- **THEN** 配置 MUST 支持 `group_size`、`num_classes`、coarse head hidden size、dropout 和 loss weights
- **AND** 系统 MUST 校验 `num_classes` 能被 `group_size` 整除
- **AND** 非法配置 MUST 抛出包含 `num_classes` 和 `group_size` 的清晰错误
