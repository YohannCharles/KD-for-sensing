## ADDED Requirements

### Requirement: History-anchor Hist 路径已退役
依赖 HiST-Beam 模型实现的 history-anchored residual beam 路径 MUST 从当前支持面退役。项目 MAY 保留与 Hist 无关的通用 history objective、GPS window baseline 或其它当前主线能力，但不得要求 Hist 模型、Hist config 或 Hist output artifact 存在。

#### Scenario: Hist history-anchor 配置不可运行
- **WHEN** 用户引用 `hist_beam.history_anchor.enabled=true` 的旧 Hist 配置
- **THEN** 系统 MUST 报告该 Hist profile 已退役或配置不存在
- **AND** 系统 MUST 不构建 HiST-Beam 模型

#### Scenario: 非 Hist history 能力不被误删
- **WHEN** 当前主线代码使用与 Hist 无关的 history objective 或 GPS window baseline
- **THEN** 该能力 MAY 保留
- **AND** 清理和源码删除 MUST 不仅凭 `history` 或 `hist` 字符串删除它

## REMOVED Requirements

### Requirement: History-anchored beam 输入契约
**Reason**: 该契约绑定旧 Hist profile，已退役。
**Migration**: 非 Hist history 输入必须由新的 workflow spec 定义。

### Requirement: Circular residual beam label
**Reason**: Hist history-anchor residual label path 已退役。
**Migration**: 当前非 Hist residual label 由对应 workflow 定义。

### Requirement: Residual logits 到绝对 beam 空间重建
**Reason**: Hist residual reconstruction path 已退役。
**Migration**: 当前非 Hist residual workflow 自行定义 reconstruction。

### Requirement: History-anchored baseline diagnostics
**Reason**: Hist history-anchor diagnostics 已退役。
**Migration**: 当前 baseline diagnostics 由保留 workflow 定义。

### Requirement: History-anchored 实验矩阵
**Reason**: Hist history-anchor quick validation 已退役。
**Migration**: 无兼容迁移。
