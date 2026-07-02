# real-perturbation-forward-evaluation Specification

## Purpose
定义 benchmark real-forward 扰动评估的真实 forward、logits/diagnostics cache、分片续跑和 leakage guard 契约，确保 P0-P5 等 condition 的指标来自模型在对应扰动 batch 上的实际输出。
## Requirements
### Requirement: Real perturbation forward evaluation
系统 MUST 支持 benchmark real-forward 模式，在每个声明的 condition 上真实应用 difficulty transform、执行模型 forward，并从实际 logits 与 hard `target_beam` 计算 Top-K、DBA 和 primary metric。

#### Scenario: 真实条件 forward
- **WHEN** benchmark manifest 声明 `evaluation.mode=real_forward` 或等价 opt-in 字段
- **THEN** runner MUST 为每个 model、condition、seed、split 构建 dataloader 并应用对应 difficulty profile
- **AND** metrics MUST 来自该 condition 下的模型 logits，而不是 deterministic degradation 估算

#### Scenario: delegated clean-only 不可升级 claim
- **WHEN** benchmark 只执行 clean delegated evaluate 且 perturbation rows 由估算生成
- **THEN** output evidence scope MUST 标记为 `delegated_clean_only`
- **AND** primary claim gate MUST 返回 `pending` 或 `unavailable`

### Requirement: Real-forward logits and diagnostics cache
Real-forward benchmark MUST 写出可复用的 logits/labels/diagnostics cache。cache MUST 记录模型、condition、seed、split、sample_count、difficulty digest、evidence scope 和 checkpoint provenance。

#### Scenario: logits cache 可复算指标
- **WHEN** real-forward evaluation 完成一个 model-condition shard
- **THEN** runner MUST 写出 logits、labels、sample ids 或等价索引
- **AND** 后续聚合 MUST 能仅从 cache 复算 Top-K、DBA 和 target-rank diagnostics

#### Scenario: diagnostics 字段缺失
- **WHEN** 模型没有输出 branch diagnostics
- **THEN** cache 或聚合表 MUST 将对应字段标记为 `unavailable`
- **AND** 系统 MUST 不生成伪造的 branch weight、entropy 或 agreement 数值

### Requirement: Real-forward resume and sharding
Real-forward benchmark MUST 支持按 model、condition 和 seed 分片运行，并能跳过已完成且 fingerprint 匹配的 cache。

#### Scenario: 分片重跑
- **WHEN** 用户只请求某个 model 或 condition 子集
- **THEN** runner MUST 只执行该子集
- **AND** 输出 manifest MUST 记录完整 planned matrix、completed shards 和 missing shards

#### Scenario: cache fingerprint mismatch
- **WHEN** checkpoint、config、difficulty digest、sample_count 或 metric profile 与已有 cache 不一致
- **THEN** runner MUST 拒绝复用 cache 或标记 stale
- **AND** claim gate MUST 不把 stale cache 当作 strict evidence

### Requirement: No future or target leakage in evaluation
Real-forward evaluation MUST 保持 target labels、beam power oracle、future frame 和 benchmark condition id 不进入模型输入。

#### Scenario: target label 只用于 metric
- **WHEN** difficulty transform 应用于 evaluation batch
- **THEN** target labels MUST 保持不变且只用于 loss/metric 或 cache labels
- **AND** 模型 forward kwargs MUST 不包含 target-only 或 oracle-only fields，除非该模型显式合法声明并且不用于 claim

#### Scenario: condition id 只用于聚合
- **WHEN** batch metadata 包含 condition、suite、c_idx、d_idx 或 advantage label
- **THEN** 这些字段 MUST 只用于输出文件名、groupby 和 report
- **AND** model/reranker/gate input tensor MUST 不消费 condition id

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
