## MODIFIED Requirements

### Requirement: MMW Town standalone Top8 candidate manifest 已退役
MMW Town standalone GPS Top8 candidate manifest CLI/config 不属于当前支持能力。系统 MUST 不再提供独立 MMW Top8 manifest CLI、配置或 BGAM 内部候选 manifest 支撑代码；BGAM 退役后，MMW GPS v2 保留 workflow MUST 不依赖 BGAM/Top8 candidate manifest 作为中间产物。

#### Scenario: MMW Top8 manifest 和 BGAM candidate 入口不存在
- **WHEN** 开发者检查安装入口和配置
- **THEN** 项目 MUST 不声明 `kd-sensing-prepare-mmw-town-top8-candidate-manifest`
- **AND** 项目 MUST 不声明 `kd-sensing-prepare-mmw-town-gps-lidar-bgam-manifest`
- **AND** 项目 MUST 不保留 `configs/mmw_town_top8_selector.yaml` 或 `configs/mmw_town_gps_lidar_bgam.yaml` 作为当前配置
