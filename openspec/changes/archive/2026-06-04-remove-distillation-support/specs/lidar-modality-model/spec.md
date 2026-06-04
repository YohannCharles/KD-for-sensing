## REMOVED Requirements

### Requirement: LiDAR KD 兼容性
**Reason**: LiDAR-only KD 配置随项目 KD 支持删除。
**Migration**: 使用 LiDAR strong 或 lightweight supervised baseline。

#### Scenario: LiDAR logits KD
- **WHEN** 用户运行旧 LiDAR logits KD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不加载 frozen LiDAR teacher

#### Scenario: LiDAR RKD
- **WHEN** 用户运行旧 LiDAR RKD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不计算关系蒸馏损失

