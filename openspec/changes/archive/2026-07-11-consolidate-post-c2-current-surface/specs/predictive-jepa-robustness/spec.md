## REMOVED Requirements

### Requirement: Predictive Robustness 主场景
**Reason**: Predictive JEPA workflow 无 current config/runner/claim consumer，整体退出。
**Migration**: Current missing stress、JEPA pretraining 和 U-Mask matrix 使用各自 owner。
#### Scenario: 主场景退出
- **WHEN** current benchmark 被枚举
- **THEN** predictive robustness suite MUST 不存在

### Requirement: JEPA predictive hybrid fusion 模型组
**Reason**: Predictive hybrid model group 与 query branches 一并退役。
**Migration**: 保留 MMW mean-context 与 current U-Mask/AMR/AMBER models。
#### Scenario: 模型组退出
- **WHEN** registry/config 被检查
- **THEN** predictive hybrid group MUST 不可构建

### Requirement: 5 个百分点 claim 口径
**Reason**: 对应模型与 benchmark 均无 current evidence。
**Migration**: Current claim gates只使用 final C2/Scene31-34 evidence。
#### Scenario: Claim 口径退出
- **WHEN** paper claim 被审阅
- **THEN** 它 MUST 不要求 predictive 5pt margin

### Requirement: Predictive Robustness 输出产物边界
**Reason**: Workflow 产品面整体删除。
**Migration**: 一般 runtime artifact boundary 继续适用于 current workflows。
#### Scenario: 专属产物退出
- **WHEN** current outputs 被规划
- **THEN** predictive manifest/summary MUST 不作为 required artifact

### Requirement: Predictive Robustness 文档治理
**Reason**: Capability 不再是 pending/current workflow。
**Migration**: 文档只保留 retired/historical note。
#### Scenario: Current 文档行退出
- **WHEN** mainline docs 被检查
- **THEN** predictive robustness MUST 不标记 current/pending

### Requirement: 训练 profile 与完整 stress benchmark 分离
**Reason**: 专属 training/benchmark profiles 整体退役。
**Migration**: 通用 difficulty profile 继续由 modality-difficulty owner管理。
#### Scenario: Profile 产品退出
- **WHEN** current profiles 被解析
- **THEN** parser MUST 不要求 predictive stress distinction

### Requirement: GPS-query advantage slice
**Reason**: GPS-query advantage diagnostics 退役。
**Migration**: 历史 A-slice 从 archive查询。
#### Scenario: Advantage slice 退出
- **WHEN** current benchmark 被构建
- **THEN** GPS-query advantage slice MUST 不可用

### Requirement: GPS-query++ strict comparison set
**Reason**: GPS-query++ 与 matched comparison set 整体删除。
**Migration**: Current comparisons 使用 U-Mask/Scene31-34 protocol。
#### Scenario: Strict set 退出
- **WHEN** current model groups 被枚举
- **THEN** GPS-query++ comparison set MUST 不存在

### Requirement: GPS-query++ claim gate
**Reason**: 没有 current producer 或 claim consumer。
**Migration**: 人工 claim registry/paper export 使用 current evidence owners。
#### Scenario: Gate 退出
- **WHEN** claim readiness 被评估
- **THEN** GPS-query++ gate MUST 不运行

### Requirement: GPS-query++ diagnostics bundle
**Reason**: Query/gate/latent branches 已删除。
**Migration**: Current diagnostics 只记录实际 model outputs。
#### Scenario: Bundle 退出
- **WHEN** current evaluation 完成
- **THEN** 它 MUST 不要求 GPS-query++ bundle

### Requirement: Predictive robustness benchmark suite
**Reason**: 旧 shortcut benchmark runner 与 suite adapter 均退役。
**Migration**: Current difficulty/evaluation owners直接使用通用 profiles。
#### Scenario: Suite adapter 退出
- **WHEN** manifest 请求 predictive suite
- **THEN** current runner MUST 拒绝

### Requirement: Predictive regional aggregation
**Reason**: 专属 metrics schema 没有 current consumer。
**Migration**: Current owners保留自己的 aggregation。
#### Scenario: Aggregation 退出
- **WHEN** current reports 被写出
- **THEN** predictive regional fields MUST 不作为 required schema

### Requirement: Predictive claim comparability
**Reason**: Predictive claim 整体退出。
**Migration**: 一般 comparability 继续由 current protocols管理。
#### Scenario: 专属 comparability 退出
- **WHEN** current claim rows 被比较
- **THEN** 它们 MUST 不要求 predictive fields

### Requirement: Predictive stress curve suite
**Reason**: 专属 stress curves 无 current runner/config。
**Migration**: 保留 generic missing-modality stress suite。
#### Scenario: Stress curve 退出
- **WHEN** current stress suite 被枚举
- **THEN** predictive curve preset MUST 不存在

### Requirement: Predictive Robustness artifact planning 必须可独立验证
**Reason**: 已删除产品不再需要 artifact planner/tests。
**Migration**: Current owners通过各自 focused tests验证。
#### Scenario: Planner 退出
- **WHEN** current tests 被枚举
- **THEN** predictive artifact planner MUST 不被要求

### Requirement: Predictive JEPA real benchmark promotion gate
**Reason**: Benchmark 与 claim promotion 均退役。
**Migration**: Final C2/Scene31-34 gate承接 current promotion。
#### Scenario: Promotion gate 退出
- **WHEN** current evidence 被审阅
- **THEN** 它 MUST 不要求 predictive benchmark

### Requirement: Predictive GPS query explanatory visualizations 属于 diagnostics bundle
**Reason**: Query visualization 与 bundle 均删除。
**Migration**: 历史 figures 留 ignored artifacts/archive。
#### Scenario: Visualization 退出
- **WHEN** current CLI/docs 被枚举
- **THEN** predictive visualization mode MUST 不存在

### Requirement: Predictive claim 证据不得依赖被删 wrapper 路径
**Reason**: Predictive claim 本身退出，不需维护专属 wrapper migration。
**Migration**: Retired note 指向 archive/git，不提供复跑路径。
#### Scenario: Wrapper evidence 退出
- **WHEN** current claim docs 被检查
- **THEN** 它们 MUST 不引用 predictive wrapper 或 claim
