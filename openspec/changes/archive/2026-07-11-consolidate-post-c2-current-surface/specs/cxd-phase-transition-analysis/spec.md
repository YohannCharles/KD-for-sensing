## REMOVED Requirements

### Requirement: CxD phase diagram 聚合
**Reason**: CxD benchmark/suite 已退役，phase diagram 没有 current inputs 或 claim consumer。
**Migration**: 历史 figure 从 archive/ignored artifacts 查询；未来矩阵分析需新建 change。

#### Scenario: Phase diagram 退出
- **WHEN** current analysis workflows 被枚举
- **THEN** CxD phase diagram aggregation MUST 不属于 current surface

### Requirement: Modality dominance 诊断
**Reason**: 该 dominance schema 只服务已退役 CxD/query benchmark。
**Migration**: Current modality diagnostics 继续由各 model/evaluation owner 管理。

#### Scenario: CxD dominance 不再计算
- **WHEN** current evaluation 运行
- **THEN** 系统 MUST 不要求 CxD modality dominance diagnostics

### Requirement: ResNet 与 JEPA crossing detection
**Reason**: Crossing detection 依赖已退役 CxD experiment matrix。
**Migration**: 历史 crossing 结论从 archive 查询；current comparisons 由 formal protocol 显式定义。

#### Scenario: Crossing detection 退出
- **WHEN** current run index 或 summary 聚合 results
- **THEN** 系统 MUST 不要求 ResNet/JEPA CxD crossing detection

### Requirement: Failure mode decomposition
**Reason**: CxD-specific failure buckets 没有 current runner 或 report consumer。
**Migration**: 删除专属 decomposition；保留 current generic metrics 与 diagnostics。

#### Scenario: Failure decomposition 退出
- **WHEN** current summary 被生成
- **THEN** CxD failure mode decomposition MUST 不再是 required section

### Requirement: 论文图与产物边界
**Reason**: CxD paper figure 产品面已退出 current claim pipeline。
**Migration**: 删除专属 figure contract；全局 ignored-output 与 paper-export boundaries 继续有效。

#### Scenario: CxD figure 不再导出
- **WHEN** current paper export 运行
- **THEN** 它 MUST 不要求 CxD figure artifacts
- **AND** 通用 artifact boundary MUST 保持不变

### Requirement: CxD analysis manifest schema
**Reason**: CxD runner 删除后，专属 manifest 没有 producer 或 consumer。
**Migration**: 删除 schema；current workflows 使用各自 canonical config/manifest。

#### Scenario: 旧 manifest 被拒绝
- **WHEN** current analysis 收到 CxD manifest
- **THEN** 它 MUST 不把该 manifest 作为 supported input

### Requirement: CxD runner output integration
**Reason**: CxD runner 与 dashboard/summary integration 同时退役。
**Migration**: 删除 adapter；current run index 不解析 CxD-specific outputs。

#### Scenario: CxD output 不再接入
- **WHEN** current run/evidence indexing 运行
- **THEN** 系统 MUST 不要求 CxD runner output integration

### Requirement: Dominance diagnostics ingestion
**Reason**: Ingestion 只服务已退役 dominance payload。
**Migration**: 删除专属 parser；不保留无 producer 的 compatibility path。

#### Scenario: Dominance payload 不再解析
- **WHEN** current summary 读取 artifacts
- **THEN** 它 MUST 不要求或解析 CxD dominance diagnostics

### Requirement: CxD no label shift guard
**Reason**: Guard 只约束已退役 CxD benchmark protocol。
**Migration**: Current dataset/evaluation 的 label-space integrity 继续由其 canonical owners 验证。

#### Scenario: 通用 label guard 独立保留
- **WHEN** current evaluation 校验 labels
- **THEN** 它 MUST 不依赖 CxD no-label-shift guard
- **AND** current label-space contracts MUST 保持有效

### Requirement: CxD benchmark focused tests
**Reason**: Benchmark/runtime 已删除，专属 tests 只会验证不存在的 product surface。
**Migration**: 删除 CxD tests；保留 current difficulty/evaluation focused tests。

#### Scenario: 专属 tests 退出
- **WHEN** current test inventory 被检查
- **THEN** 项目 MUST 不要求 CxD benchmark focused tests

