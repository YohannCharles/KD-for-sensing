## ADDED Requirements

### Requirement: Benchmark real-forward mode
JEPA GPS shortcut benchmark MUST 支持 real-forward mode，用于真实执行 P0-P5、Scenario C/D、CxD 和 GPS advantage slice 的 model forward。

#### Scenario: real-forward manifest
- **WHEN** manifest 声明 real-forward mode
- **THEN** runner MUST 要求每个模型具备 config、weights 或 logits cache
- **AND** 对无 cache 的模型 MUST 真实执行 evaluation forward，而不是仅生成 planned/degraded metrics

#### Scenario: deterministic degradation evidence scope
- **WHEN** runner 使用 deterministic degradation model 生成 perturbation rows
- **THEN** 输出 MUST 标记 evidence scope 为 `diagnostic_estimate`
- **AND** predictive 或 geometry primary claim gate MUST 不允许该 evidence 升级 primary claim

### Requirement: Benchmark output matrix completeness
Real-forward benchmark MUST 输出 planned/completed/missing matrix。矩阵 MUST 覆盖 model、group、condition、seed、split、sample_count 和 evidence scope。

#### Scenario: 缺失 shard
- **WHEN** 某个 model-condition shard 未运行或失败
- **THEN** runner MUST 在 matrix 中标记 missing/failed
- **AND** claim gate MUST 返回 pending，除非 manifest 显式声明该 shard 不属于 claim scope

#### Scenario: 完整 strict matrix
- **WHEN** 所有 claim-scope shards 完成且 cache fingerprint 一致
- **THEN** benchmark summary MUST 标记 real-forward matrix complete
- **AND** strict comparison table MUST 可追溯每个模型的 checkpoint/config/difficulty provenance

### Requirement: Branch diagnostics aggregation
Benchmark MUST 能聚合模型输出中的 branch diagnostics，同时允许普通 baseline 缺失这些字段。

#### Scenario: opt-in 模型输出 diagnostics
- **WHEN** model output 包含 anchor logits、prior logits、rerank logits、candidate ids、branch weights 或 fallback reason
- **THEN** runner MUST 将这些字段写入 diagnostics cache 或 aggregate CSV/JSON
- **AND** 缺失字段 MUST 标记 `unavailable`

#### Scenario: 普通 baseline 缺失 diagnostics
- **WHEN** Image ResNet+GPS 或其它 baseline 不输出 rerank diagnostics
- **THEN** benchmark MUST 继续计算 metrics
- **AND** diagnostics 表中该模型对应字段 MUST 为 `unavailable`
