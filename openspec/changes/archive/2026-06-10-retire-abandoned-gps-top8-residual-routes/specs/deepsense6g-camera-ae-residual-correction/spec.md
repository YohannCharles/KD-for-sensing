## ADDED Requirements

### Requirement: DeepSense6G camera residual 已退役
DeepSense6G camera-assisted residual correction 不再属于当前支持能力。系统 MUST 不再提供 camera residual manifest、camera residual model/loss/training、candidate reranker、plotter、comparison CLI 或默认 residual 配置。

#### Scenario: camera residual 入口不存在
- **WHEN** 开发者检查安装入口和配置
- **THEN** 项目 MUST 不声明 `kd-sensing-run-deepsense6g-camera-residual`、plot、compare 或 manifest 入口
- **AND** 项目 MUST 不保留 `configs/deepsense6g_camera_residual.yaml` 作为当前配置

## REMOVED Requirements

### Requirement: DeepSense6G camera residual 默认工作流
**Reason**: camera 预测 residual/delta 的路线已退役。
**Migration**: 无兼容迁移。

### Requirement: Camera residual manifest
**Reason**: manifest 只服务 camera residual workflow。
**Migration**: 无兼容迁移。

### Requirement: Camera AE pretraining
**Reason**: 该 AE pretraining 在本 capability 中服务 camera residual；若未来作为独立能力需要保留，必须另立非 residual capability。
**Migration**: 无兼容迁移。

### Requirement: AE feature extraction
**Reason**: AE feature extraction 在本 capability 中服务 camera residual。
**Migration**: 无兼容迁移。

### Requirement: CameraGPSResidualFusion 模型
**Reason**: camera GPS residual 模型已退役。
**Migration**: 无兼容迁移。

### Requirement: Camera residual loss
**Reason**: camera residual loss 已退役。
**Migration**: 无兼容迁移。

### Requirement: Camera residual training protocols and ablations
**Reason**: camera residual training protocol 已退役。
**Migration**: 无兼容迁移。

### Requirement: Beam candidate attention reranker
**Reason**: candidate reranker 属于退役 Top8/residual 路线。
**Migration**: 无兼容迁移。

### Requirement: Camera residual outputs and visualization
**Reason**: camera residual 输出和可视化随 workflow 退役。
**Migration**: 无兼容迁移。
