# jepa-visual-architecture-sweep Specification

## ADDED Requirements

### Requirement: Architecture sweep 候选矩阵
系统 MUST 提供 GPS-query JEPA visual architecture sweep 候选矩阵。矩阵 MUST 覆盖当前 patch16 baseline、patch/token 粒度、overlap tokenizer、conv stem tokenizer、局部 token mixing、CNN feature-map tokens、多尺度 tokens、frame embedding anchor、pooler/core ablation 和非 Transformer 对照，并 MUST 为每个候选声明唯一 `variant_id`。

#### Scenario: 候选矩阵包含实用架构族
- **WHEN** 开发者加载 architecture sweep manifest 或配置矩阵
- **THEN** manifest MUST 至少包含 `baseline`、`patch_granularity`、`overlap_tokenizer`、`conv_stem_tokenizer`、`local_token_mixing`、`cnn_tokens`、`multi_scale_tokens`、`frame_embedding_anchor`、`pooler_core_ablation` 和 `non_transformer_control` 架构族
- **AND** 每个候选 MUST 记录 `variant_id`、`family`、`visual_encoder.type`、`pooler.type`、`checkpoint_policy` 和 `run_tier`

#### Scenario: 候选矩阵区分 JEPA 与非 JEPA anchor
- **WHEN** 候选不复用 JEPA context encoder checkpoint
- **THEN** manifest MUST 将其标记为 `supervised_only_anchor` 或等价 checkpoint policy
- **AND** 系统 MUST 不把该候选描述为 JEPA checkpoint reuse 结果

### Requirement: Sweep strict comparability metadata
系统 MUST 为每个 architecture sweep 候选写出严格可比性 metadata。metadata MUST 记录 split、scene set、seed、history window、GPS input source window、prediction horizon、beam label space、metric profile、distance metric、normalization artifact、difficulty digest 和 output root。

#### Scenario: strict 字段完整
- **WHEN** 候选参与 strict sweep 或保留/淘汰判断
- **THEN** 该候选 metadata MUST 包含所有 strict comparability 字段
- **AND** 任一 strict 字段缺失或与 baseline 不一致时，系统 MUST 将该候选标记为不可升级 claim 或拒绝纳入 strict ranking

#### Scenario: smoke 结果不能升级 claim
- **WHEN** 候选只完成 smoke 或 lowmem 可运行性验证
- **THEN** manifest MUST 将 evidence scope 标记为 `smoke`、`lowmem` 或等价非 primary scope
- **AND** 系统 MUST 不把该结果用于最终主线保留判断

### Requirement: Sweep 诊断与选择指标
系统 MUST 为 architecture sweep 输出统一诊断和选择指标。指标 MUST 至少包含 Top-1、Top-3、Top-5、DBA、相邻 beam error 或 circular/linear beam distance summary、参数量或 trainable 参数量、token count、attention 或 branch summary 和运行 provenance。

#### Scenario: 写出统一结果表
- **WHEN** architecture sweep 评估完成
- **THEN** 系统 MUST 写出 machine-readable summary table 或 manifest
- **AND** 每行 MUST 包含 variant metadata、strict comparability 字段、主 beam metrics、compute proxy 和 diagnostics 状态

#### Scenario: GPS shortcut 诊断可用
- **WHEN** 候选使用 GPS-query、Predictive GPS-query++、GPS residual 或 reliability gate
- **THEN** diagnostics MUST 记录 attention entropy/peakiness、branch/gate weights 或等价 summary
- **AND** wrong-GPS、counterfactual GPS 或 P3/P4 条件下的指标 MUST 能与 clean/P0 指标区分

### Requirement: Sweep 输出产物边界
architecture sweep 训练、评估、checkpoint、logits、attention map、CSV、JSON 和图表 MUST 写入 ignored runtime output 目录，默认位于 `outputs/analysis/jepa_visual_architecture_sweep/` 或配置声明的 ignored output root。源码变更 MUST 只包含配置、代码、测试和 OpenSpec artifact。

#### Scenario: 运行产物不进入源码
- **WHEN** 用户运行 architecture sweep 训练或评估
- **THEN** 生成的 checkpoint、log、cache、attention map、summary figure 和 logits cache MUST 位于 ignored output 目录
- **AND** manifest 中 MUST 使用相对路径或可审计 provenance 指向这些产物

#### Scenario: 清理或重跑不影响源码
- **WHEN** 用户删除 sweep 输出目录后重新运行
- **THEN** 源码中的配置、测试和 OpenSpec artifacts MUST 仍足以重建候选矩阵
- **AND** 系统 MUST 不要求提交本地 checkpoint 或缓存
