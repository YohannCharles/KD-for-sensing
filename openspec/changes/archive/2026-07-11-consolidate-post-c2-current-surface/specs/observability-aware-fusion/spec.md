## REMOVED Requirements

### Requirement: JEPA advantage condition
**Reason**: Scenario-D/CxD 与 GPS-query advantage 路线整体退役，current fusion 不再按 benchmark condition id 改变模型 forward。
**Migration**: 通用低图像可观测性 fallback 继续只由 reliability metadata 驱动。

#### Scenario: CxD condition 不进入 current fusion
- **WHEN** current observability-aware fusion 接收 image/GPS reliability metadata
- **THEN** fallback MUST 只由通用 validity 和 observability threshold 决定
- **AND** batch 与 model forward MUST 不装配 Scenario-D/CxD advantage condition metadata

### Requirement: Logit-level uncertainty fusion
**Reason**: 该模式只服务已退役 geometry-prior branch，当前 reliability fusion 不需要专属多分支 logit schema。
**Migration**: 保留通用 reliability weighting 和 proto-compatible mask-weighted fusion。

#### Scenario: Geometry logit fusion 不再构建
- **WHEN** current fusion registry 被枚举
- **THEN** geometry-prior logit fusion MUST 不存在
- **AND** ordinary fusion MUST 不要求 prior logits 或 branch entropy

### Requirement: Geometry-prior condition id isolation
**Reason**: Geometry-prior model branch 删除后没有 condition-id consumer。
**Migration**: 通用规则仍禁止 benchmark condition id 进入模型输入。

#### Scenario: Geometry condition guard 退出专属实现
- **WHEN** retired geometry config 被请求
- **THEN** config/registry MUST 拒绝该路线
- **AND** 项目 MUST 不保留 geometry 专属 condition guard helper

### Requirement: GPS reliability in logit fusion
**Reason**: 该 requirement 只约束已删除 geometry logit fusion。
**Migration**: Current models MAY 通过通用 reliability metadata contract 消费 GPS validity。

#### Scenario: Prior weight 不再是 current 输出
- **WHEN** current observability fusion 运行
- **THEN** diagnostics MUST 不要求 geometry prior weight 或 prior-image disagreement

### Requirement: Image observability in logit fusion
**Reason**: 该 requirement 只约束已删除 geometry logit fusion。
**Migration**: Image availability 继续由 missing-mask 和通用 reliability fusion owner处理。

#### Scenario: Geometry image branch 权重不再要求
- **WHEN** image observability 下降
- **THEN** current contract MUST 不要求已退役 geometry fusion 输出 image/prior branch weights

### Requirement: Reliability-aware predictive gate inputs
**Reason**: Predictive GPS-query++ 与其 gate 已退出 current surface。
**Migration**: JEPA mean context 和 ordinary baselines 继续使用现有 optional reliability fields。

#### Scenario: Predictive gate 不再构建
- **WHEN** config 请求 predictive GPS-query reliability gate
- **THEN** config/registry MUST 拒绝该路线
- **AND** batch runtime MUST 不保留其专属输入装配

### Requirement: Condition id isolation for predictive gates
**Reason**: Predictive gate 已删除；保留专属 isolation helper 没有 consumer。
**Migration**: Benchmark condition id 仍只能由保留的通用 evaluation/aggregation owner使用。

#### Scenario: Predictive condition guard 不再专属存在
- **WHEN** current model forward 被检查
- **THEN** predictive gate condition-id branch MUST 不存在

### Requirement: Predictive branch weight diagnostics
**Reason**: Current models 不再产生 current/predicted/GPS-residual 三分支 gate 输出。
**Migration**: Current owner 只记录其实际产生的 diagnostics。

#### Scenario: Predictive branch schema 退出
- **WHEN** diagnostics schema 被验证
- **THEN** current model MUST 不被要求输出 predictive branch weights

### Requirement: No-regret reliability gate
**Reason**: Anchor-safe residual reranker 整体退役。
**Migration**: Current U-Mask/AMR/AMBER fusion 分支保持自己的既有 gate/loss 语义。

#### Scenario: Reranker gate 不再构建
- **WHEN** config 请求 safe residual reranker
- **THEN** registry MUST 拒绝该 component
- **AND** current fusion MUST 不承担其 fallback schema

### Requirement: Anchor fallback branch diagnostics
**Reason**: Anchor/prior/residual rerank branch 已删除，没有 producer 或 claim consumer。
**Migration**: 历史 diagnostics 从 OpenSpec archive/git 查询。

#### Scenario: Rerank diagnostics 退出
- **WHEN** current evaluation 聚合 model output
- **THEN** evaluation MUST 不要求 candidate ids、prior beam、fallback reason 或 rerank delta
- **AND** ordinary model metrics MUST 保持

### Requirement: Reliability fusion Scene31 seed extension guard
**Reason**: Scene31 next-round reliability workflow 与 shared summary gate 已退役；final Scene31-34 conclusion 明确不继续该方法搜索。
**Migration**: 保留通用 proto-compatible reliability mask-weighted component，不保留旧 Scene31 seed-extension product。

#### Scenario: Scene31 reliability seed gate 退出
- **WHEN** current configs、scripts 和 docs 被枚举
- **THEN** 项目 MUST 不要求 reliability seed4/5 continuation runner 或 Scene31 summary gate
- **AND** current reliability component behavior MUST 不因删除实验编排而改变
