# deepsense6g-gps-residual-fusion Specification

## Purpose
定义 DeepSense6G GPS v2 prior anchored residual fusion 工作流的配置、manifest、模型、loss、训练协议、ablation、输出和 comparison 契约，确保 optional modalities 只在 GPS prior 之上做 residual correction 或 candidate rerank，并保证 target query 只用于最终评价。
## Requirements
### Requirement: DeepSense6G GPS residual fusion 已退役
DeepSense6G GPS v2 prior anchored residual correction 不再属于当前支持能力。系统 MUST 不再提供 residual input inspection、manifest、GPSAnchoredResidualFusion、residual loss、training protocol、reranker、plotter、comparison CLI 或默认配置。

#### Scenario: residual workflow 不可运行
- **WHEN** 开发者检查 console scripts、配置和包内模块
- **THEN** 项目 MUST 不声明 DeepSense6G residual 相关 `kd-sensing-*` 命令
- **AND** 项目 MUST 不保留 `configs/deepsense6g_residual_fusion.yaml` 作为当前配置
- **AND** `src/kd_sensing` MUST 不保留该 workflow 专属实现模块

