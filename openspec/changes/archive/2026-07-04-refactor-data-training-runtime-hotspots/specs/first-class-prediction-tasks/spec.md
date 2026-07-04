## ADDED Requirements

### Requirement: Prediction objective 输出必须在 runtime 拆分后保持一致
Prediction objective metadata, target preparation, loss payloads, history fields and TensorBoard scalar schema MUST remain compatible when batch or evaluation helpers are moved.

#### Scenario: objective metadata 仍是单一来源
- **WHEN** runtime helpers prepare prediction targets or objective-specific outputs
- **THEN** 它们 MUST 从 `kd_sensing.engine.objectives.metadata` 消费 metadata
- **AND** they MUST NOT recreate a second objective registry or old facade import path
