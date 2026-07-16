# openspec-document-health Specification

## Purpose

保持 current 文档与 MMW T2/baseline 支持面一致，防止已删除路线通过 README、维护索引、claim 文档或 current spec 重新被描述为可用能力。

## Requirements

### Requirement: current 文档只描述 retained surface

README、导航、维护索引、实验协议、claim registry 与 current specs MUST 只把 T2、S1、AMBER-Full、RMBP-MM 及其 MMW runtime 描述为 current。

#### Scenario: 文档清理后检查

- **WHEN** source/config/CLI 删除一个非 retained route
- **THEN** current 文档 MUST 同步移除其入口和维护要求
- **AND** 其历史用途 MUST 只保留在集中历史说明或 archive

### Requirement: 文档变更不写入运行产物

文档 health check MUST 为只读检查；它不得自动重写 README、OpenSpec、claim 文档或 local output。

#### Scenario: 运行文档验证

- **WHEN** 维护者运行文档或架构检查
- **THEN** 检查 MUST 不读取真实 dataset 或启动训练
- **AND** 不得写入 `outputs/`、logs、cache 或 checkpoint
