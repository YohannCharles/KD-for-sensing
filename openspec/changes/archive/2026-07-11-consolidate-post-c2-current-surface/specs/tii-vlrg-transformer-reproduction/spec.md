## REMOVED Requirements

### Requirement: TII baseline manifest
**Reason**: TII VLRG Transformer 外部 reproduction 没有 current config、CLI 或 claim consumer。
**Migration**: 历史 manifest 和 provenance 保留在 archive/ignored outputs。

#### Scenario: TII manifest 不再生成
- **WHEN** 用户检查 current reproduction workflows
- **THEN** 系统 MUST NOT 提供 TII baseline manifest 入口

### Requirement: TII 外部 workflow 执行边界
**Reason**: 外部 wrapper 只服务已退役的 TII reproduction。
**Migration**: 若未来需要外部 reproduction，必须用新 change 定义窄入口和产物边界。

#### Scenario: TII execute mode 不再存在
- **WHEN** current package entrypoints 被枚举
- **THEN** 它们 MUST NOT 包含 TII external-workflow wrapper

### Requirement: TII 指标适配
**Reason**: TII 专用 DBA row adapter 无 current downstream consumer。
**Migration**: 历史可比性结论保留在 archive；current metrics 由对应 evaluation owner 管理。

#### Scenario: TII row 不再进入 current ranking
- **WHEN** current strict ranking 生成
- **THEN** 它 MUST NOT 依赖 TII-specific metric adapter

### Requirement: TII 产物边界
**Reason**: 产物边界随 TII workflow 一并退役。
**Migration**: 已有本地产物继续保持 ignored，不迁移或提交。

#### Scenario: 历史 TII 产物保持本地
- **WHEN** consolidation 删除 TII workflow
- **THEN** 实现 MUST NOT 将历史 checkpoint、logs 或 predictions 移入 tracked source
