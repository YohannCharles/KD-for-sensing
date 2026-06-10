## ADDED Requirements

### Requirement: MMW Town standalone Top8 candidate manifest 已退役
MMW Town standalone GPS Top8 candidate manifest CLI/config 不再属于当前支持能力。系统 MUST 不再提供独立 MMW Top8 manifest CLI 或配置；BGAM 内部候选 manifest 支撑代码 MAY 保留。

#### Scenario: MMW Top8 manifest 入口不存在
- **WHEN** 开发者检查安装入口和配置
- **THEN** 项目 MUST 不声明 `kd-sensing-prepare-mmw-town-top8-candidate-manifest`
- **AND** 项目 MUST 不保留 `configs/mmw_town_top8_selector.yaml` 作为当前配置

## REMOVED Requirements

### Requirement: MMW Town GPS Top8 candidate manifest workflow
**Reason**: standalone MMW Top8 candidate manifest workflow 已退役。
**Migration**: MMW GPS v2 自身 workflow 和 BGAM 内部候选 manifest 支撑代码保留。
