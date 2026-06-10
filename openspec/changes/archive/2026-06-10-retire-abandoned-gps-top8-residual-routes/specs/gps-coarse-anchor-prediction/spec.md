## ADDED Requirements

### Requirement: GPS coarse anchor prediction 已退役
GPS coarse anchor、neural coarse head、residual preview、GPS prior fallback 和 coarse-anchor 历史 pseudo label 导出不再属于当前支持能力。系统 MUST 不再提供 `kd-sensing-gps-coarse-anchor`、coarse anchor 配置、engine 或下游 residual 消费契约。

#### Scenario: coarse anchor 入口不存在
- **WHEN** 开发者检查安装入口和配置目录
- **THEN** 项目 MUST 不声明 `kd-sensing-gps-coarse-anchor`
- **AND** 项目 MUST 不保留 `configs/gps/*coarse*` 作为当前可运行配置

## REMOVED Requirements

### Requirement: GPS coarse anchor 输入边界
**Reason**: GPS coarse anchor profile 已退役。
**Migration**: 使用保留的 GPS-Rel-Polar 或 GPS v2 workflow；不提供 coarse anchor 迁移。

### Requirement: GPS coarse anchor 输出契约
**Reason**: coarse anchor 输出不再作为当前 artifact 契约。
**Migration**: 无兼容迁移。

### Requirement: BeamBench-style 几何 anchor
**Reason**: geometry calibrated anchor 已退役。
**Migration**: 无兼容迁移。

### Requirement: GPS neural coarse anchor
**Reason**: neural coarse head 已退役。
**Migration**: 使用普通 GPS-only supervised 模型。

### Requirement: 跨场景 GPS anchor 评估
**Reason**: anchor evaluation profile 已退役。
**Migration**: 使用 GPS v2 或普通 evaluation metrics。

### Requirement: Residual anchor 预览
**Reason**: residual preview 服务已放弃的差值学习路线。
**Migration**: 无兼容迁移。

### Requirement: DeepSense6G GPS v2 prior artifact export
**Reason**: 该 requirement 位于 coarse/residual downstream 契约中；downstream residual/Top8 selector 已退役。
**Migration**: GPS v2 自身诊断或 BGAM 需要的 logits artifact 由对应保留 capability 定义。

### Requirement: DeepSense6G GPS prior fallback
**Reason**: fallback Gaussian prior 服务 residual/camera residual workflow，已退役。
**Migration**: 无兼容迁移。

### Requirement: 历史 GPS coarse/pseudo label 序列导出
**Reason**: coarse-anchor pseudo-history 导出服务 residual，已退役。
**Migration**: 无兼容迁移。
