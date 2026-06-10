## ADDED Requirements

### Requirement: geometry residual label 路线已退役
专门用于把绝对 beam 拆成 geometry coarse beam 与 residual/delta class 的 label-space 路线不再属于当前支持能力。系统 MUST 不再要求 dataset 暴露 `beam_geo`、`beam_residual`、`residual_class`、GPS local delta class 或 geometry residual target provider。

#### Scenario: 默认数据样本不含 geometry residual 契约
- **WHEN** 用户运行当前保留训练或评估配置
- **THEN** dataset sample MUST 不要求 geometry residual 字段
- **AND** 项目 MUST 不保留专门服务 residual/delta 路线的 target provider 文档或测试

## REMOVED Requirements

### Requirement: geometry coarse beam 构造
**Reason**: geometry coarse beam 服务退役 residual label 路线。
**Migration**: 无兼容迁移。

### Requirement: circular residual beam label
**Reason**: residual beam label 路线已退役。
**Migration**: 通用 circular distance/metrics 可继续由其它 specs 约束。

### Requirement: clipped residual class
**Reason**: clipped residual class 服务退役 delta learning。
**Migration**: 无兼容迁移。

### Requirement: dataset sample 暴露 geometry-residual 字段
**Reason**: geometry residual sample contract 已退役。
**Migration**: 使用当前保留 dataset 的 absolute label contract。

### Requirement: geometry sector 诊断字段
**Reason**: geometry sector 在本 capability 中服务 residual diagnostics。
**Migration**: 若未来需要非 residual sector diagnostics，需另行提出。

### Requirement: GPS anchored signed circular residual utilities
**Reason**: GPS anchored residual utilities 服务退役 residual/camera residual workflow。
**Migration**: 通用 circular distance 保留；GPS anchored residual helper 不提供兼容。

### Requirement: GPS-prior local residual delta class
**Reason**: local residual delta class 是已放弃差值学习路线核心。
**Migration**: 无兼容迁移。
