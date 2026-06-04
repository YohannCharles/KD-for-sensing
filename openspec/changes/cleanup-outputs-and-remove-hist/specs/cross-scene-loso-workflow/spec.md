## ADDED Requirements

### Requirement: LOSO 不再绑定 Hist 默认矩阵
当前项目 MUST 不再提供 HiST-Beam 默认 LOSO 矩阵或 `kd-sensing-hist-beam-loso` 执行入口。未来若需要跨场景矩阵，MUST 由当前保留 workflow 通过新的 spec 明确定义配置、CLI、输出和防泄漏边界。

#### Scenario: Hist LOSO 入口不可用
- **WHEN** 用户尝试运行 `kd-sensing-hist-beam-loso`
- **THEN** 系统 MUST 不把该命令作为当前支持入口
- **AND** README 和健康检查 MUST 不要求该命令存在

#### Scenario: 当前 LOSO fold 定义可被未来 workflow 复用
- **WHEN** 未来非 Hist workflow 需要 leave-one-scene-out fold
- **THEN** 新 workflow MUST 显式声明自己的 runner、配置矩阵和输出契约
- **AND** 系统 MUST 不复用已退役 Hist run plan 作为隐式默认

## REMOVED Requirements

### Requirement: LOSO 运行编排入口
**Reason**: 该入口服务 HiST-Beam 默认矩阵，已随 Hist 研究线退役。
**Migration**: 未来跨场景编排由当前保留 workflow 独立定义。

### Requirement: LOSO 结果汇总
**Reason**: HiST-Beam quick validation summary 已退役。
**Migration**: 使用当前 workflow 自身 summary。

### Requirement: LOSO execute 执行闭环
**Reason**: `kd-sensing-hist-beam-loso --execute` 已退役。
**Migration**: 使用当前保留 CLI。

### Requirement: Quick validation 最小执行矩阵
**Reason**: HiST-Beam quick validation 矩阵已退役。
**Migration**: 当前主线 quick validation 由各自 workflow spec 定义。

### Requirement: LOSO 旧解耦 baseline 退役
**Reason**: 整个 Hist LOSO baseline 集合已退役，该局部退役规则不再需要。
**Migration**: 无兼容迁移。

### Requirement: LOSO execute summary 产物
**Reason**: Hist LOSO execute summary 已退役。
**Migration**: 当前 workflow summary 自行定义。

### Requirement: LOSO execute 进度与中断可诊断
**Reason**: Hist LOSO execute runner 已退役。
**Migration**: 当前 workflow 进度和中断诊断自行定义。
