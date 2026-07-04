## ADDED Requirements

### Requirement: U-MaskBeamJEPA loss extension 必须按训练职责拆分
U-MaskBeamJEPA training extension MUST 将 missing-pattern DRO、BTAPA/prototype target construction、loss config normalization、missing-mask metadata 和 epoch logging 拆分到窄 helper，并保持 hook 行为。

#### Scenario: MP-DRO 日志兼容
- **WHEN** MP-DRO helper is moved or refactored
- **THEN** `mpdro_group_log.csv`, epoch metadata, pattern weights and warning behavior MUST remain compatible
- **AND** tests MUST 覆盖 enabled 和 disabled extension 路径
