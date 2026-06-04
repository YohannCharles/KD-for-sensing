## ADDED Requirements

### Requirement: HiST-Beam 可显式消费 GPS coarse anchor
HiST-Beam 系统 MUST 支持显式 opt-in 的 GPS coarse anchor 条件输入。启用该能力后，模型、loss、评估和 prediction artifact MUST 能消费并记录 GPS anchor；未启用时现有 HiST-Beam 默认行为 MUST 保持不变。

#### Scenario: 默认不启用 GPS anchor 条件输入
- **WHEN** 用户运行未设置 `hist_beam.gps_anchor.enabled=true` 的 HiST-Beam 配置
- **THEN** batch preparation MUST NOT 要求 GPS coarse anchor 字段
- **AND** 模型 forward MUST 保持现有输入契约
- **AND** run metadata MUST NOT 声称模型使用了 GPS anchor

#### Scenario: 显式启用 GPS anchor 条件输入
- **WHEN** 用户设置 `hist_beam.gps_anchor.enabled=true`
- **THEN** batch preparation MUST 向模型提供 `coarse_logits`、`center_beam`、`confidence` 和 `residual_anchor_beam` 或等价 GPS anchor 字段
- **AND** 模型 forward MUST 将 GPS anchor 用作 coarse/fine 或 residual 分支的条件输入
- **AND** run metadata MUST 记录 `uses_gps_coarse_anchor=true`

#### Scenario: 缺失 GPS anchor 字段清晰失败
- **WHEN** HiST-Beam 配置启用 GPS anchor 条件输入但 batch 中缺少必需 anchor 字段
- **THEN** 系统 MUST 抛出包含缺失字段名和配置路径的清晰错误
- **AND** 系统 MUST NOT 静默回退到普通 HiST-Beam 输入语义

#### Scenario: prediction artifact 记录 anchor
- **WHEN** 启用 GPS anchor 的 HiST-Beam evaluation 完成
- **THEN** predictions artifact MUST 包含 true beam、predicted beam、coarse true/pred、GPS anchor center beam、anchor coarse top-k 和 anchor confidence
- **AND** summary MUST 能比较 model prediction 相对 GPS anchor 的 residual 改善
