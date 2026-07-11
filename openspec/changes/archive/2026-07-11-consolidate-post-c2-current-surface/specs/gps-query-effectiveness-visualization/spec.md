## REMOVED Requirements

### Requirement: GPS-query 有效性证据包输入
**Reason**: GPS-query effectiveness 产品面与其 upstream pooler 均已退役。
**Migration**: 删除 evidence input schema；正式 claims 继续使用 current protocol/artifact owners。

#### Scenario: Evidence package 不再接受输入
- **WHEN** current evidence workflow 被运行
- **THEN** 它 MUST 不要求 GPS-query effectiveness input package

### Requirement: Paired ablation 有效性指标
**Reason**: Paired query ablation 没有 current run/config consumer。
**Migration**: 历史 metrics 从 archive 查询；current comparisons 使用各自 protocol 定义的 metrics。

#### Scenario: Query paired metrics 退出
- **WHEN** current summary 聚合 runs
- **THEN** 系统 MUST 不要求 GPS-query paired ablation metrics

### Requirement: GPS-query attention 热点图
**Reason**: Query attention map producer 随 pooler 删除，热点图没有 current input。
**Migration**: 删除 renderer；不为退役 attention payload 保留兼容 wrapper。

#### Scenario: Attention heatmap 不再生成
- **WHEN** current visualization workflow 运行
- **THEN** GPS-query attention heatmap MUST 不再是 required artifact

### Requirement: Query gain/regression case study
**Reason**: Case-study owner 只服务已退役 query effectiveness claim。
**Migration**: 历史案例从 archive/ignored outputs 查询；current claims 由正式 evidence owner 维护。

#### Scenario: Query case study 退出
- **WHEN** current evidence package 被导出
- **THEN** 系统 MUST 不要求 query gain/regression cases

### Requirement: Claim gate 报告
**Reason**: GPS-query claim 不再属于 current claim surface，专属 gate report 没有消费方。
**Migration**: Current claim promotion 继续由 claim registry 与 paper export gate 管理。

#### Scenario: Query gate 不阻塞 current claims
- **WHEN** current paper export 检查 claims
- **THEN** 它 MUST 不要求 GPS-query effectiveness gate report

### Requirement: 产物边界和可测试性
**Reason**: 整个 evidence/visualization product 被删除，专属 artifact/test contract 失去对象。
**Migration**: 保留全局 ignored-output 与 source artifact boundaries。

#### Scenario: 专属 artifact contract 退出
- **WHEN** project artifact boundaries 被验证
- **THEN** 系统 MUST 不要求 GPS-query visualization artifacts 或专用 tests
- **AND** 通用 output boundary MUST 保持不变

### Requirement: Attention token-read 解释边界
**Reason**: Query token-read diagnostics 删除后，其解释性边界不再有 current payload。
**Migration**: Future interpretability work 必须以新 producer 与新 OpenSpec change 定义。

#### Scenario: Token-read explanation 退出
- **WHEN** current evidence 文档被检查
- **THEN** 它 MUST 不把 GPS-query token-read explanation 声明为 current requirement

### Requirement: Attention faithfulness 诊断
**Reason**: Faithfulness diagnostics 依赖已退役 query attention 与 evidence package。
**Migration**: 删除专属 diagnostics；历史方法从 archive 查询。

#### Scenario: Faithfulness diagnostics 不再运行
- **WHEN** current diagnostics 被枚举
- **THEN** GPS-query attention faithfulness diagnostics MUST 不属于 current surface

### Requirement: Faithfulness-aware claim gate
**Reason**: 对应 GPS-query claim 与 faithfulness producer 均已退出 current surface。
**Migration**: Current claim gates 不依赖该 schema；未来恢复需重新提案。

#### Scenario: Faithfulness gate 退出
- **WHEN** current claim status 被计算或审阅
- **THEN** 系统 MUST 不要求 GPS-query faithfulness-aware gate

