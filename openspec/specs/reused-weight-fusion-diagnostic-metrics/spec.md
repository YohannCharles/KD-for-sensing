# reused-weight-fusion-diagnostic-metrics Specification

## Purpose
定义复用既有 checkpoint 的融合诊断评估契约，用于在 clean、图像/GPS degraded、CxD 和 hard-negative 条件下输出可比性受控的 condition metrics、paired margins 和诊断 bundle。
## Requirements
### Requirement: 复用权重诊断输入契约
系统 MUST 支持一个复用现有模型权重的融合诊断评估 profile。该 profile MUST 接收已有模型的 config 路径、weights 路径、model group、split、seed、label space、metric profile 和 checkpoint provenance，并 MUST 不触发训练、微调或 checkpoint 改写。

#### Scenario: 使用已有 config 和 weights
- **WHEN** benchmark manifest 声明 reused-weight fusion diagnostic profile
- **THEN** 系统 MUST 对每个模型读取已有 config 和 weights 执行 real-forward evaluation
- **AND** 系统 MUST 在 manifest 中记录 config、weights、checkpoint provenance、split、seed、metric profile 和 label space
- **AND** 系统 MUST 不创建新的训练 run 或覆盖已有 checkpoint

#### Scenario: 不可比模型被标记
- **WHEN** 两个模型的 split、seed、label space、metric profile、history window、scene set 或 difficulty digest 不一致
- **THEN** 系统 MUST 将相关 paired margin 标记为 not-comparable
- **AND** report MUST 不把该 paired margin 用作 claim gate

### Requirement: 正交融合诊断条件集
系统 MUST 提供默认的小型正交条件集，用于区分 clean、图像受损、GPS 受损、双模态受损和 hard-negative 场景。默认条件集 MUST 复用现有 Scenario C、Scenario D、CxD 和 GPS-query advantage slice 语义。

#### Scenario: 默认 CxD 小切片
- **WHEN** manifest 使用默认 reused-weight diagnostic condition preset
- **THEN** 系统 MUST 至少评估 `C0_sync+D0_full_image`、`C0_sync+D4_partial_occlusion`、`C0_sync+D6_burst_missing`、`C3_random_async+D0_full_image`、`C4_severe_async+D0_full_image`、`C3_random_async+D4_partial_occlusion`、`C4_severe_async+D6_burst_missing` 和 `C4_severe_async+D7_joint_worst_case`
- **AND** 每个 condition MUST 记录 GPS condition、image condition、seed、split、sample_count、difficulty digest 和 clean anchor

#### Scenario: 默认 hard-negative 切片
- **WHEN** manifest 启用 reused-weight diagnostic hard-negative preset
- **THEN** 系统 MUST 至少包含 visual ambiguity、beam-offset-constrained wrong GPS 和 visual-ambiguous wrong GPS 条件
- **AND** 每个 hard-negative condition MUST 记录 peer selection pool、beam offset constraint、fallback count 和 sample_count

### Requirement: 融合诊断指标输出
系统 MUST 为复用权重诊断输出 condition-level 指标、paired baseline margin 和融合派生指标。指标 MUST 来自对应 condition 下的真实 logits 与 hard target label。

#### Scenario: 输出 condition-level 表
- **WHEN** reused-weight diagnostic evaluation 完成
- **THEN** 系统 MUST 写出 condition-level DBA、Top-1、Top-3、Top-5、clean delta、relative drop、sample_count 和 comparability status
- **AND** 每行 MUST 包含 model、group、condition、gps_condition、image_condition、seed、split 和 difficulty digest

#### Scenario: 输出融合派生指标
- **WHEN** clean、single-modality-degraded 和 joint-degraded 条件均可比
- **THEN** 系统 MUST 计算并输出 `image_rescue`、`gps_rescue`、`fusion_interaction` 和 paired margin
- **AND** 缺少任一必要条件时，对应派生指标 MUST 标记为 unavailable 或 not-comparable

### Requirement: 诊断报告和产物边界
复用权重融合诊断 MUST 输出 machine-readable manifest、CSV/JSON summary 和可选图表登记。真实运行产物 MUST 写入 ignored `outputs/` 或 manifest 指定目录，源码变更 MUST 不提交数据、cache、checkpoint、日志或图像产物。

#### Scenario: 写出诊断 bundle
- **WHEN** reused-weight diagnostic evaluation 完成
- **THEN** 输出目录 MUST 包含 benchmark manifest、condition metrics、paired margin table、fusion diagnostic summary 和 warnings
- **AND** report MUST 明确 P0-P5 是兼容鲁棒性表，CxD/A-slice 是融合机制诊断表

#### Scenario: 运行产物不进入源码
- **WHEN** 开发者实施或验证该 change
- **THEN** 新生成的 `outputs/`、cache、日志、checkpoint、PNG、SVG 或 HTML MUST 不纳入源码变更
- **AND** 可复现入口 MUST 通过配置、manifest 或文档命令记录

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
