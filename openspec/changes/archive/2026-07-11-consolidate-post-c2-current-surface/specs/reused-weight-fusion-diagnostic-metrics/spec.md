## REMOVED Requirements

### Requirement: 复用权重诊断输入契约
**Reason**: 仓库没有 current runner、config、CLI 或 source consumer 实现该独立 profile。
**Migration**: 历史诊断通过 dated archive/git 查询；current evaluation 使用保留的 evaluate/U-Mask matrix owners。

#### Scenario: 独立 profile 退出
- **WHEN** current entrypoints 和 configs 被枚举
- **THEN** reused-weight diagnostic profile MUST 不存在

### Requirement: 正交融合诊断条件集
**Reason**: 默认条件依赖已退役 Scenario C/D、CxD 和 GPS-query advantage slice。
**Migration**: Current missing-modality evaluation 使用 U-Mask matrix 与受保护 difficulty owners。

#### Scenario: Retired condition preset 不再要求
- **WHEN** current diagnostics 被验证
- **THEN** validation MUST 不要求旧 CxD/GPS-query preset

### Requirement: 融合诊断指标输出
**Reason**: 对应 runner 不存在，派生 image/GPS rescue 与 fusion-interaction 指标没有 current claim consumer。
**Migration**: Current owners 只输出其正式 metric schema。

#### Scenario: Orphan metrics 不再生成
- **WHEN** current evaluation 完成
- **THEN** evaluation MUST 不被要求生成 reused-weight 专属 paired margin

### Requirement: 诊断报告和产物边界
**Reason**: 独立 diagnostics bundle 产品面整体退役。
**Migration**: 保留的一般 runtime artifact 边界继续适用于所有 current evaluation。

#### Scenario: 独立 bundle 不再存在
- **WHEN** current workflows 被列举
- **THEN** reused-weight manifest/report MUST 不作为 required artifact

### Requirement: Benchmark output matrix completeness
**Reason**: 该 matrix 只服务已退役 real-forward benchmark。
**Migration**: U-Mask eval matrix 与 current owner维护自己的 completeness contract。

#### Scenario: Retired matrix 不阻塞 claim
- **WHEN** paper export 检查 current evidence
- **THEN** 它 MUST 不要求 reused-weight planned/completed/missing matrix

### Requirement: Branch diagnostics aggregation
**Reason**: Anchor/prior/rerank schema 的 producer 已删除，普通 metrics 不应维持 orphan optional columns。
**Migration**: Current model owner可聚合其实际存在的 diagnostics，不保留本专属 capability。

#### Scenario: Rerank columns 不再要求
- **WHEN** ordinary baseline 被评估
- **THEN** report MUST 不要求 prior/rerank/candidate/fallback fields
- **AND** Top-K、DBA 和 current metrics MUST 保持
