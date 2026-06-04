## REMOVED Requirements

### Requirement: mmWave KD 兼容性
**Reason**: mmWave-only KD 配置随项目 KD 支持删除。
**Migration**: 使用 mmWave strong 或 lightweight supervised baseline。

#### Scenario: mmWave logits KD
- **WHEN** 用户运行旧 mmWave logits KD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不加载 frozen mmWave teacher

#### Scenario: mmWave RKD
- **WHEN** 用户运行旧 mmWave RKD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不计算关系蒸馏损失

