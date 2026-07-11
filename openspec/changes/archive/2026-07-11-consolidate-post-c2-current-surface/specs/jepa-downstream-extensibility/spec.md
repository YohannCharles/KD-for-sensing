## REMOVED Requirements

### Requirement: JEPA downstream pooler 和 adapter 可插拔
**Reason**: 当前配置只消费 mean context reuse，通用 pooler/adapter 扩展面没有 current consumer。
**Migration**: Mean reuse 归 `gps-conditioned-jepa-pretraining`；未来新 pooler 需独立 change。

#### Scenario: 扩展 registry 不再要求
- **WHEN** current JEPA downstream 构建
- **THEN** 系统 MUST 不要求通用 pooler/adapter registry

### Requirement: JEPA downstream 派生实验保持 baseline 可比
**Reason**: 派生 query 实验已退役。
**Migration**: 历史 comparability 留在 archive/claim caveat。

#### Scenario: 历史派生实验不再运行
- **WHEN** 用户请求旧 derived config
- **THEN** 系统 MUST 拒绝或报告 unknown config

### Requirement: JEPA downstream 参数组优化
**Reason**: Query/adapter 参数组随扩展面删除。
**Migration**: Mean context encoder 使用普通 encoder optimizer policy。

#### Scenario: Optimizer 不识别 query group
- **WHEN** current model 构建 optimizer
- **THEN** 不要求 downstream query/adapter 参数组

### Requirement: JEPA downstream metadata 可追踪
**Reason**: Query/adapter 专属 metadata 不再生成。
**Migration**: 保留 checkpoint、encoder type 和 mean pooling metadata。

#### Scenario: Metadata 不含 query fields
- **WHEN** current mean path 写出 metadata
- **THEN** 不要求 k_queries、condition source 或 attention diagnostics

### Requirement: JEPA temporal context fallback
**Reason**: 仅服务 predictive/query downstream。
**Migration**: Current mean path 使用现有序列输入，不维护该 fallback。

#### Scenario: Fallback config 被拒绝
- **WHEN** 配置请求 downstream temporal fallback
- **THEN** 配置 MUST 不作为 current path 接受

### Requirement: JEPA downstream 消费 observability metadata
**Reason**: 该 metadata consumer 只服务退役 downstream branch。
**Migration**: Current U-Mask/reliability owner 继续管理自己的 metadata。

#### Scenario: Mean path 不要求 observability metadata
- **WHEN** mean context encoder forward
- **THEN** 它 MUST 不要求 query observability fields

### Requirement: JEPA fallback 与 benchmark condition 对齐
**Reason**: 对应 benchmark 与 fallback 已退役。
**Migration**: 历史 condition 只留 archive。

#### Scenario: Benchmark condition 不再构建
- **WHEN** 用户请求旧 fallback benchmark
- **THEN** current workflow MUST 不提供该 suite

### Requirement: Hybrid residual query pooler
**Reason**: 无 current config consumer。
**Migration**: 使用 mean pooling；新 hybrid 需新 change。

#### Scenario: Hybrid pooler unavailable
- **WHEN** 配置请求 hybrid residual query
- **THEN** component construction MUST fail

### Requirement: Temporal predicted latent auxiliary branch
**Reason**: 仅服务退役 predictive downstream。
**Migration**: 使用 current JEPA pretraining objective 或另提 change。

#### Scenario: Auxiliary branch 不输出
- **WHEN** current mean encoder forward
- **THEN** 不要求 temporal predicted latent

### Requirement: Feature-consistency fusion diagnostics
**Reason**: 仅服务退役派生比较。
**Migration**: Current owner 保留各自 focused diagnostics。

#### Scenario: Diagnostics fields 不再要求
- **WHEN** mean downstream 运行
- **THEN** 不要求 feature-consistency fusion diagnostics

### Requirement: Predictive GPS-query++ downstream pooler
**Reason**: 无 current config、CLI 或 claim consumer。
**Migration**: 使用 mean context reuse。

#### Scenario: Predictive pooler unavailable
- **WHEN** 配置请求 Predictive GPS-query++
- **THEN** component construction MUST fail

### Requirement: Causal temporal latent predictor
**Reason**: 只属于退役 Predictive GPS-query++。
**Migration**: Future temporal predictor 必须重新定义。

#### Scenario: Predictor 不再构建
- **WHEN** current JEPA downstream 构建
- **THEN** 不要求 causal temporal predictor

### Requirement: Predictive JEPA auxiliary latent objectives
**Reason**: 只服务退役 predictive branch。
**Migration**: Current JEPA objective 保留自身 latent loss。

#### Scenario: Auxiliary objectives 不注册
- **WHEN** current loss registry 加载
- **THEN** 不要求 predictive downstream auxiliary objectives

### Requirement: GPS-query++ metadata and compatibility
**Reason**: GPS-query++ 实现与 checkpoint compatibility 一并退役。
**Migration**: 历史 checkpoint 只作为本地产物保留，不承诺 current load。

#### Scenario: 旧 checkpoint 不自动迁移
- **WHEN** 用户加载 GPS-query++ checkpoint 到 current mean path
- **THEN** 系统 MUST 不静默迁移

### Requirement: Downstream visual token source variants
**Reason**: 变体矩阵无 current config consumer。
**Migration**: Current pretraining visual encoder variants 由 pretraining spec 独立管理。

#### Scenario: Downstream token variant unavailable
- **WHEN** downstream config 请求旧 token source variant
- **THEN** current builder MUST reject it

### Requirement: K-token downstream fusion opt-in
**Reason**: K-token readout 只服务退役 query sweep。
**Migration**: Current mean path 固定输出 `[B,T,D]`。

#### Scenario: K-token opt-in 不再接受
- **WHEN** 配置启用 downstream K-token mode
- **THEN** validation MUST fail

### Requirement: Visual token diagnostics for downstream sweep
**Reason**: Downstream sweep 已退役。
**Migration**: 历史图表/metadata 留在 ignored artifacts 或 archive。

#### Scenario: Sweep diagnostics 不生成
- **WHEN** current mean path 运行
- **THEN** 不要求 token sweep diagnostics

### Requirement: GPS-query attention aggregation metadata
**Reason**: GPS-query attention path 已删除。
**Migration**: 无替代；mean pooling 无 attention map。

#### Scenario: Attention metadata 不输出
- **WHEN** current mean path 运行
- **THEN** 不要求 query/head aggregation fields

### Requirement: Opt-in per-head attention diagnostics
**Reason**: Per-head diagnostics 只服务 GPS-query。
**Migration**: 无 current 替代。

#### Scenario: Per-head flag 被拒绝
- **WHEN** config 请求 per-head query diagnostics
- **THEN** config MUST not be accepted as current

### Requirement: 现有 supervised/adaptation workflow 不变
**Reason**: 该兼容承诺依赖已退役 downstream 扩展面，current supervised behavior 由模型/训练 specs 直接覆盖。
**Migration**: 保留 mean encoder、config、training/evaluation focused tests。

#### Scenario: Current workflow 由 owner 验证
- **WHEN** current supervised config 运行
- **THEN** owner focused tests MUST 验证行为

### Requirement: JEPA downstream pooler 和 adapter 注册
**Reason**: 通用 registry 不再需要。
**Migration**: Mean pooling 内联到 context encoder owner。

#### Scenario: 旧 pooler 名称 unknown
- **WHEN** registry 收到旧 pooler 名称
- **THEN** MUST return unknown-name failure

### Requirement: JEPA downstream 注册保持轻量导入
**Reason**: Registry 整体删除后该导入约束失去对象。
**Migration**: Config 轻量导入边界继续由 project/component specs 覆盖。

#### Scenario: Config import 仍轻量
- **WHEN** import `kd_sensing.config`
- **THEN** 不得因 mean reuse 导入重型 runtime

### Requirement: JEPA downstream query/pooling helper 必须可独立演进
**Reason**: Query/pooling helper 被删除，不再作为扩展点维护。
**Migration**: Future query work 从新 change 和最小实现开始。

#### Scenario: Helper modules 不保留
- **WHEN** consolidation 完成
- **THEN** 只服务 query/pooling 的 helper MUST 不再存在
