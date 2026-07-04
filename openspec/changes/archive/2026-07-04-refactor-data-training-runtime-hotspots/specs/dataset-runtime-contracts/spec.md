## ADDED Requirements

### Requirement: Dataset contract 规则必须进入窄 helper
新的 DeepSense6G 或 MMW dataset contract 规则 MUST 实现在 GPS contract、cache path、column、target provider、scaler setup、resource reader 或 family adapter 等窄 helper 中，而不是继续扩大 dataset orchestration class。

#### Scenario: 新增 column 或 target 规则
- **WHEN** a change adds a required column, beam target source, GPS feature mode or cache path rule
- **THEN** 实现 MUST 位于对应 dataset contract/helper 模块
- **AND** synthetic focused tests MUST 覆盖该规则且不读取真实 `dataset/`
