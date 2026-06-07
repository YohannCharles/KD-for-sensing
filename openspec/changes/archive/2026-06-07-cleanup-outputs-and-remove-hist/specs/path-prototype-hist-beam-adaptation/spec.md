## ADDED Requirements

### Requirement: P3/HiST path prototype 已退役
P3/HiST-Beam path prototype adaptation MUST 从当前支持面退役。系统 MUST 不再提供 P3-HiST 模型 forward、path prototype target adaptation、P3 smoke 配置或 P3 inference artifact。

#### Scenario: P3 Hist 配置不可运行
- **WHEN** 用户引用 P3/HiST path prototype 配置或 variant
- **THEN** 系统 MUST 报告该入口已退役或配置不存在
- **AND** 系统 MUST 不构建 HiST-Beam path prototype 模型

## REMOVED Requirements

### Requirement: Path-level 数据巡检与字段映射
**Reason**: P3/HiST path prototype workflow 已退役。
**Migration**: 当前主线如需 path diagnostics，必须用非 Hist spec 重新定义。

### Requirement: Path-level dataset 输出与输入边界
**Reason**: P3/HiST dataset contract 已退役。
**Migration**: 无兼容迁移。

### Requirement: PathFeatureBuilder 物理 descriptor
**Reason**: P3/HiST path feature builder 已退役。
**Migration**: 无兼容迁移。

### Requirement: PathSemanticLabelBuilder
**Reason**: P3/HiST path semantic label builder 已退役。
**Migration**: 无兼容迁移。

### Requirement: P3-HiST-Beam 模型 forward
**Reason**: P3-HiST model 已退役。
**Migration**: 无兼容迁移。

### Requirement: Source path loss
**Reason**: P3 source path loss 已退役。
**Migration**: 无兼容迁移。

### Requirement: Source path prototype artifact
**Reason**: P3 source path prototype artifact 已退役。
**Migration**: 无兼容迁移。

### Requirement: Path prototype 不依赖旧解耦 source
**Reason**: P3 path prototype 整体退役，该约束不再需要。
**Migration**: 无兼容迁移。

### Requirement: Path prototype target adaptation
**Reason**: P3 target adaptation 已退役。
**Migration**: 无兼容迁移。

### Requirement: P3 inference 与诊断指标
**Reason**: P3 inference diagnostics 已退役。
**Migration**: 无兼容迁移。

### Requirement: P3 配置、变体与 smoke tests
**Reason**: P3 configs and smoke tests 已退役。
**Migration**: 无兼容迁移。
