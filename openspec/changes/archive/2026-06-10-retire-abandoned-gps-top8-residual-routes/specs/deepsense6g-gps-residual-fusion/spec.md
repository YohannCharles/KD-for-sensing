## ADDED Requirements

### Requirement: DeepSense6G GPS residual fusion 已退役
DeepSense6G GPS v2 prior anchored residual correction 不再属于当前支持能力。系统 MUST 不再提供 residual input inspection、manifest、GPSAnchoredResidualFusion、residual loss、training protocol、reranker、plotter、comparison CLI 或默认配置。

#### Scenario: residual workflow 不可运行
- **WHEN** 开发者检查 console scripts、配置和包内模块
- **THEN** 项目 MUST 不声明 DeepSense6G residual 相关 `kd-sensing-*` 命令
- **AND** 项目 MUST 不保留 `configs/deepsense6g_residual_fusion.yaml` 作为当前配置
- **AND** `src/kd_sensing` MUST 不保留该 workflow 专属实现模块

## REMOVED Requirements

### Requirement: DeepSense6G GPS v2 prior anchored residual workflow
**Reason**: GPS 粗预测后学习 residual/delta 的路线实验精度不足，已退役。
**Migration**: 无兼容迁移；使用保留的 GPS v2、supervised/adaptation 或其它当前主线。

### Requirement: Residual input inspection
**Reason**: residual workflow 已退役。
**Migration**: 无兼容迁移。

### Requirement: Residual manifest
**Reason**: residual manifest 只服务退役 workflow。
**Migration**: 无兼容迁移。

### Requirement: GPSAnchoredResidualFusion model
**Reason**: GPS anchored residual model 已退役。
**Migration**: 无兼容迁移。

### Requirement: Residual fusion losses
**Reason**: residual/gate/good-anchor loss 只服务退役 workflow。
**Migration**: 无兼容迁移。

### Requirement: Residual training protocols
**Reason**: residual training protocol 已退役。
**Migration**: 无兼容迁移。

### Requirement: Residual ablation matrix
**Reason**: residual ablation matrix 已退役。
**Migration**: 无兼容迁移。

### Requirement: GPS anchored top-K reranker
**Reason**: GPS anchored reranker 属于失败路线。
**Migration**: 无兼容迁移。

### Requirement: Residual outputs and comparison report
**Reason**: residual 输出与报告随 workflow 退役。
**Migration**: 无兼容迁移。

### Requirement: Residual visualization
**Reason**: residual visualization CLI 随 workflow 退役。
**Migration**: 无兼容迁移。
