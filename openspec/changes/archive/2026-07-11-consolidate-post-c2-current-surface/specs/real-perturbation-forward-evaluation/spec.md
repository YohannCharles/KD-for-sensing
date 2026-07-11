## REMOVED Requirements

### Requirement: Real perturbation forward evaluation
**Reason**: 专属 real-perturbation runner 只服务已退役 geometry/query benchmark。
**Migration**: Current evaluation 继续通过 canonical evaluator 与 difficulty pipeline 运行。

#### Scenario: Real-forward runner 退出
- **WHEN** current evaluation entrypoints 被枚举
- **THEN** 专属 real perturbation forward runner MUST 不属于 current surface

### Requirement: Real-forward logits and diagnostics cache
**Reason**: Runner 删除后，其 logits/diagnostics cache 没有 producer 或 consumer。
**Migration**: 删除专属 cache schema；current runtime cache policy 保持不变。

#### Scenario: 专属 cache 不再读写
- **WHEN** current evaluation 运行
- **THEN** 它 MUST 不要求 real-forward logits/diagnostics cache

### Requirement: Real-forward resume and sharding
**Reason**: Resume/sharding 只服务已退役 runner。
**Migration**: Current long-running workflows 使用各自 runner 的 resume semantics，不保留通用 wrapper。

#### Scenario: Real-forward resume 退出
- **WHEN** current runtime 恢复任务
- **THEN** 它 MUST 不要求 real-forward-specific resume 或 sharding state

### Requirement: No future or target leakage in evaluation
**Reason**: 该 requirement 的专属对象是已退役 real-forward runner；通用 leakage 边界已有 current owners。
**Migration**: Dataset split、causal transform 与 evaluation contracts 继续执行 no-leakage rules。

#### Scenario: 通用 no-leakage 继续有效
- **WHEN** current evaluation 构建 inputs
- **THEN** 它 MUST 不依赖 real-forward runner 的专属 guard
- **AND** canonical no-future/no-target-leakage contracts MUST 保持有效

### Requirement: Benchmark real-forward mode
**Reason**: 依赖该 mode 的 shortcut/Scenario-D/CxD benchmarks 均已退役。
**Migration**: 删除 manifest mode；current benchmarks/evaluations 使用其 canonical modes。

#### Scenario: 旧 mode 被拒绝
- **WHEN** manifest 请求 benchmark real-forward mode
- **THEN** current parser MUST 返回 unknown/removed failure

### Requirement: Real-forward diagnostics 必须从 runner 主流程分离
**Reason**: Runner 与 diagnostics 同时删除，继续规定内部拆分没有 current 实现对象。
**Migration**: Future diagnostics 遵守通用 owner boundary；恢复 real-forward 必须重新提案。

#### Scenario: 无 speculative helper 保留
- **WHEN** consolidation 完成
- **THEN** 项目 MUST 不为 real-forward diagnostics 保留 orphan helper 或 facade
