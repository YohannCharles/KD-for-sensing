## REMOVED Requirements

### Requirement: GPS geometry beam prior
**Reason**: Geometry-prior baseline 已无 current config、CLI 或 claim consumer，保留实现只会维持孤立分支。
**Migration**: 删除专属 branch；MMW direct geometry 继续由 MMW owner 管理，未来恢复须新建 OpenSpec change。

#### Scenario: Geometry prior 不再构建
- **WHEN** current model config 被解析
- **THEN** `gps_geometry_prior` MUST 不再作为可构建 component
- **AND** MMW current geometry feature MUST 不受影响

### Requirement: Geometry prior logit fusion
**Reason**: Logit fusion 只服务已退役 geometry-prior baseline，且没有 current consumer。
**Migration**: Current fusion 使用保留的 modular/U-Mask owner；历史 fusion 语义从 archive 查询。

#### Scenario: Geometry logit fusion 退出
- **WHEN** current component registry 被加载
- **THEN** geometry-prior logit fusion MUST 不再注册或执行
- **AND** ordinary fusion behavior MUST 保持不变

### Requirement: Clean-first curriculum and claim gate
**Reason**: 该 curriculum 与 gate 只约束已退役 geometry-prior 实验。
**Migration**: Current claim gate 继续由正式 protocol、claim registry 与 paper export owner 管理。

#### Scenario: Geometry clean-first gate 不再要求
- **WHEN** current experiment evidence 被审查
- **THEN** 系统 MUST 不要求 geometry-prior clean-first curriculum 或专属 claim gate

### Requirement: Teacher-guided stabilization
**Reason**: Geometry 专属 teacher stabilization 没有 current route，且与保留的 supervised/U-Mask teacher owner 重叠。
**Migration**: 保留通用 current teacher guidance；不保留 geometry-specific teacher branch。

#### Scenario: Geometry teacher branch 退出
- **WHEN** current loss/runtime 被构建
- **THEN** geometry-prior teacher stabilization MUST 不再是 required path
- **AND** current U-Mask teacher behavior MUST 保持不变

### Requirement: Geometry-prior diagnostics bundle
**Reason**: 专属 baseline 删除后，其 diagnostics bundle 没有 producer 或 consumer。
**Migration**: 历史 diagnostics 从 archive 或 ignored artifacts 查询；current owners 保留各自 focused diagnostics。

#### Scenario: Geometry diagnostics 不再生成
- **WHEN** current evaluation 完成
- **THEN** 系统 MUST 不要求 geometry-prior diagnostics bundle

### Requirement: Geometry-prior claim requires real perturbation forward
**Reason**: Geometry claim 与 real-forward benchmark 同时退役，该组合 gate 不再有 current 对象。
**Migration**: Current claims 继续使用各自 formal evaluation protocol；未来恢复需重新定义 evidence contract。

#### Scenario: Retired claim gate 不阻塞 current export
- **WHEN** paper export 检查 current claim
- **THEN** 它 MUST 不要求 geometry-prior real-perturbation-forward evidence

### Requirement: Geometry prior rerank diagnostics
**Reason**: Geometry reranker 与专属 diagnostics 均无 current config consumer。
**Migration**: 删除该 diagnostics schema；保留通用 prediction/evaluation metrics。

#### Scenario: Rerank diagnostics 退出
- **WHEN** current model forward 或 evaluation 运行
- **THEN** geometry-prior rerank diagnostics MUST 不再是 output obligation

### Requirement: Clean no-regret summary
**Reason**: No-regret summary 只服务已退役 geometry/safe-rerank 组合。
**Migration**: Current summaries 继续报告其 owner 定义的普通 metrics，不生成 geometry 专属结论。

#### Scenario: No-regret summary 不再要求
- **WHEN** current summary artifact 被写出
- **THEN** 系统 MUST 不要求 geometry clean no-regret section

### Requirement: Geometry-prior fusion component configuration
**Reason**: 专属 component 已退出 current registry，配置契约没有可构建目标。
**Migration**: 删除相关 config keys/validation；旧配置按 unknown/removed route 拒绝。

#### Scenario: 旧 geometry config 被拒绝
- **WHEN** config 请求 geometry-prior fusion component
- **THEN** validation 或 component construction MUST fail clearly
- **AND** 系统 MUST 不静默映射到 current fusion

### Requirement: Geometry-prior fusion input fields
**Reason**: 专属 fusion 删除后，其 batch input fields 不再有 consumer。
**Migration**: 保留 current modality 与 reliability fields；删除 geometry-only routing obligation。

#### Scenario: Batch 不准备 geometry 专属字段
- **WHEN** current ordinary or missing-modality model 准备 batch
- **THEN** runtime MUST 不要求 geometry-prior fusion input fields

### Requirement: Geometry-prior canonical configs
**Reason**: Geometry-prior 路线已退役，不应继续以 canonical YAML 维持可运行表面。
**Migration**: 删除实体 configs 与推荐命令；历史配置从 git/archive 查询。

#### Scenario: Canonical config 退出
- **WHEN** current config inventory 被检查
- **THEN** geometry-prior canonical configs MUST 不存在或被推荐

