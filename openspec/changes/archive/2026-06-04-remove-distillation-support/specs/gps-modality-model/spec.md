## REMOVED Requirements

### Requirement: GPS KD 兼容性
**Reason**: GPS-only KD 配置随项目 KD 支持删除。
**Migration**: 使用 GPS strong 或 lightweight supervised baseline。

#### Scenario: GPS logits KD
- **WHEN** 用户运行旧 GPS logits KD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不加载 frozen GPS teacher

#### Scenario: GPS RKD
- **WHEN** 用户运行旧 GPS RKD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不计算关系蒸馏损失

