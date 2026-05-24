## ADDED Requirements

### Requirement: Artifact schema 拆分兼容
训练和评估相关模块拆分后，用户可见 artifact schema MUST 保持兼容。`final_config.yaml`、`resolved_config.yaml`、`train_log.json`、`training_outputs.npz`、`metrics.json`、checkpoint sidecar、teacher metrics 和 TensorBoard tag 的关键字段、路径和含义 MUST 不因内部模块移动而改变。

#### Scenario: 训练 artifact 字段保持
- **WHEN** 训练流程内部 writer、objective metadata 或 runtime metadata helper 被拆分
- **THEN** `final_config.yaml`、`train_log.json` 和 `metrics.json` 中既有公开字段 MUST 保持可用
- **AND** focused tests MUST 覆盖关键字段 presence

#### Scenario: objective metadata 拆分后兼容
- **WHEN** objective metadata 表、alias、history fields 或 TensorBoard schema 被迁移到窄模块
- **THEN** 训练、验证和评估 MUST 继续解析同一组 objective、metric alias、metric mode 和 history fields
- **AND** 现有 objective tests MUST 保持通过
