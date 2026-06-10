# gps-coarse-anchor-prediction Specification

## Purpose
定义 GPS coarse anchor prediction 的输入边界、输出契约、校准来源和评估产物，确保几何或轻量 neural anchor 只使用可观测 GPS/pose 信息，并能作为 HiST-Beam residual/fusion 工作流的可审计粗粒度 beam 先验。
## Requirements
### Requirement: GPS coarse anchor prediction 已退役
GPS coarse anchor、neural coarse head、residual preview、GPS prior fallback 和 coarse-anchor 历史 pseudo label 导出不再属于当前支持能力。系统 MUST 不再提供 `kd-sensing-gps-coarse-anchor`、coarse anchor 配置、engine 或下游 residual 消费契约。

#### Scenario: coarse anchor 入口不存在
- **WHEN** 开发者检查安装入口和配置目录
- **THEN** 项目 MUST 不声明 `kd-sensing-gps-coarse-anchor`
- **AND** 项目 MUST 不保留 `configs/gps/*coarse*` 作为当前可运行配置

