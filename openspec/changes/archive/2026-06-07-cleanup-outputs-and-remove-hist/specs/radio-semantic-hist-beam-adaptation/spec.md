## ADDED Requirements

### Requirement: Radio-semantic Hist 已退役
Radio-semantic HiST-Beam adaptation MUST 从当前支持面退役。系统 MUST 不再提供 radio-semantic HiST model output、radio prototype artifact、radio-conditioned beam head、target adaptation 或 variant matrix。

#### Scenario: Radio-semantic Hist variant 不可构建
- **WHEN** 用户选择 radio-semantic HiST variant
- **THEN** 系统 MUST 报告该入口已退役或注册名不存在
- **AND** 系统 MUST 不构建 radio-conditioned HiST beam head

## REMOVED Requirements

### Requirement: Radio-semantic label 构造
**Reason**: Radio-semantic HiST workflow 已退役。
**Migration**: 非 Hist radio diagnostics 需另行定义。

### Requirement: Radio-semantic dataset contract
**Reason**: Radio-semantic HiST dataset contract 已退役。
**Migration**: 无兼容迁移。

### Requirement: Radio-semantic HiST-Beam 模型输出与融合推理
**Reason**: Radio-semantic HiST model 已退役。
**Migration**: 无兼容迁移。

### Requirement: Source radio prototype artifact
**Reason**: Radio prototype artifact 已退役。
**Migration**: 无兼容迁移。

### Requirement: Radio-semantic target adaptation
**Reason**: Radio-semantic target adaptation 已退役。
**Migration**: 无兼容迁移。

### Requirement: Radio-semantic loss 与防泄漏
**Reason**: Radio-semantic HiST loss 已退役。
**Migration**: 无兼容迁移。

### Requirement: Radio-semantic 评估指标
**Reason**: Radio-semantic HiST evaluation 已退役。
**Migration**: 无兼容迁移。

### Requirement: Radio-semantic variant matrix
**Reason**: Radio-semantic HiST matrix 已退役。
**Migration**: 无兼容迁移。

### Requirement: Radio prototype 不依赖旧解耦 baseline
**Reason**: Radio-semantic Hist 整体退役，该局部约束不再需要。
**Migration**: 无兼容迁移。
