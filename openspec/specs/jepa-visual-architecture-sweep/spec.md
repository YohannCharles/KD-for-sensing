# jepa-visual-architecture-sweep Specification

## Purpose
定义 GPS-query JEPA visual architecture sweep 的候选矩阵、严格可比性 metadata、诊断指标和输出产物边界，用于比较视觉 tokenizer、pooler/core、CNN/hybrid 先验与非 JEPA anchor 的可运行证据。
## Requirements
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

### Requirement: 当前 architecture sweep 入口优先
JEPA visual architecture sweep MUST 以 `jepa_visual_architecture_sweep` owner、`configs/diagnostics/jepa_visual_architecture_sweep_manifest.yaml` 和 `configs/fusion/experiments/jepa_image_gps/architecture_sweep_{smoke,lowmem,strict}.yaml` 作为当前推荐 sweep surface。旧 CNN/hybrid full sweep MAY 仅作为历史兼容 reader 保留，不得继续作为默认训练 runner。

#### Scenario: 模型摘要读取当前 manifest
- **WHEN** 用户通过模型架构摘要入口读取 sweep manifest
- **THEN** 系统 MUST 支持当前 `jepa_visual_architecture_sweep` manifest schema
- **AND** 系统 MUST 不要求旧 full sweep runner 存在

#### Scenario: 旧 full sweep 只读兼容
- **WHEN** 仍需读取 `cnn_hybrid_jepa_visual_prior_sweep` 历史 manifest
- **THEN** 系统 MAY 保留只读 manifest expansion 或 summary reader
- **AND** 该兼容路径 MUST 不生成训练 job、不清理 output root、不调度 GPU 任务

### Requirement: Sweep 参数摘要使用统一 schema
JEPA visual architecture sweep MUST 将候选参数和 compute metadata 映射到统一模型架构摘要 schema。summary table MUST 保留 `variant_id`、family、stage plan、checkpoint policy、token metadata、total params、trainable params、image encoder params、visual/context encoder params、compute proxy 和参数来源字段。

#### Scenario: full results 行包含统一参数字段
- **WHEN** architecture sweep 生成 full results 或 expanded manifest summary
- **THEN** 每个候选行 MUST 包含 total params、trainable params、image encoder params、visual/context encoder params、token count 和 compute proxy
- **AND** 每个候选行 MUST 记录参数来源是声明 metadata、真实 module 统计还是混合来源

#### Scenario: missing metrics 不删除参数摘要
- **WHEN** 候选训练失败、被跳过、missing metrics 或 availability 为 unavailable
- **THEN** summary MUST 仍保留该候选的参数摘要字段
- **AND** summary MUST 不因缺失指标而从 full table 中静默移除该候选

### Requirement: 极小参数量 JEPA 候选作为基准口径
JEPA visual architecture sweep MUST 将 `patch14_stage1_gps_query` 作为参数摘要基准候选之一。summary fixture 或 focused test MUST 锁定其约 0.197M total params、约 0.117M image encoder params 和约 0.088M visual/context encoder params 的当前口径，允许使用明确容差或 source-managed fixture。

#### Scenario: patch14 极小模型参数口径
- **WHEN** summary 处理 `patch14_stage1_gps_query` 候选
- **THEN** 输出 MUST 包含约 0.197M total params
- **AND** 输出 MUST 包含约 0.117M image encoder params
- **AND** 输出 MUST 包含约 0.088M visual/context encoder params

#### Scenario: patch14 与 ResNet token 候选同表比较
- **WHEN** summary 同时包含 `patch14_stage1_gps_query`、`resnet18_layer4_tokens` 和 `resnet18_layer3_layer4_tokens`
- **THEN** 三个候选 MUST 使用同一参数字段名和同一参数来源标记
- **AND** summary MUST 能按 total params、image encoder params 或 visual/context encoder params 排序

### Requirement: ResNet token 候选参数口径保持可比
JEPA visual architecture sweep MUST 保留 ResNet token 候选的参数摘要口径。`resnet18_layer4_tokens` 和 `resnet18_layer3_layer4_tokens` MUST 在 summary 中报告 total params、image encoder params 和 visual/context encoder params，以支持与 patch/overlap/hybrid 候选比较。

#### Scenario: resnet18 layer4 token 参数口径
- **WHEN** summary 处理 `resnet18_layer4_tokens` 候选
- **THEN** 输出 MUST 包含约 11.32M total params
- **AND** 输出 MUST 包含约 11.24M image encoder params
- **AND** 输出 MUST 包含约 11.21M visual/context encoder params

#### Scenario: resnet18 layer3+layer4 token 参数口径
- **WHEN** summary 处理 `resnet18_layer3_layer4_tokens` 候选
- **THEN** 输出 MUST 包含约 14.13M total params
- **AND** 输出 MUST 包含约 14.05M image encoder params
- **AND** 输出 MUST 包含约 14.02M visual/context encoder params

### Requirement: Sweep Pareto 使用统一参数字段
JEPA visual architecture sweep 的 Pareto、family best 和 Markdown summary MUST 使用统一模型架构摘要字段。系统 MUST 支持按 DBA、Top-1、trainable params、total params、image encoder params、visual/context encoder params、token count 和 compute proxy 生成候选解释。

#### Scenario: Pareto 区分极小模型和大 CNN token 模型
- **WHEN** summary 生成 params/compute Pareto
- **THEN** `patch14_stage1_gps_query` 的极小参数量优势 MUST 能在 Pareto 表中体现
- **AND** ResNet token 候选的较大视觉参数规模 MUST 能在同一表中体现

#### Scenario: Markdown summary 解释规模收益
- **WHEN** summary 生成 Markdown 报告
- **THEN** 报告 MUST 包含参数规模对照段落或表格
- **AND** 报告 MUST 避免只按最终指标排名而隐藏参数量差异

### Requirement: GPS-query token/readout paired ablation
JEPA visual architecture sweep MUST include a minimal paired ablation for GPS-query token output and token readout. The ablation MUST compare token readout candidates against matching mean, GPS-query frame, and legacy GPS-query token baselines under the same data, seed, checkpoint selection, metric profile, difficulty condition and output root.

#### Scenario: readout ablation 候选完整
- **WHEN** full 或 focused GPS-query readout sweep manifest 生成
- **THEN** manifest MUST include `pooler_mean`、`pooler_gps_query_k2_frame`、`pooler_gps_query_k2_tokens` and at least one explicit token readout candidate
- **AND** each candidate MUST record `variant_id`、family、pooler type、output mode、`k_queries`、readout type、representation core type、checkpoint policy and run tier
- **AND** manifest MUST NOT silently replace or rename existing `pooler_gps_query_k2_tokens`

#### Scenario: readout ablation 使用严格可比字段
- **WHEN** readout ablation 结果进入 strict ranking 或 claim gate
- **THEN** each row MUST include split、scene set、seed、history window、GPS input source window、prediction horizon、beam label space、metric profile、distance metric、normalization artifact、difficulty digest、checkpoint selection and output root
- **AND** any row missing these fields MUST be excluded from strict readout claim ranking

#### Scenario: readout gate 输出 paired delta
- **WHEN** summary 生成 GPS-query token/readout claim gate
- **THEN** gate MUST output paired delta versus `pooler_gps_query_k2_frame` and versus `pooler_mean`
- **AND** gate MUST include clean/P0 delta、P1-P5 mean delta、Scene31 delta、S31-S34 delta、S32-S34 delta and P3/P4 degradation-condition delta where available
- **AND** gate MUST record threshold、pass/fail status、missing evidence and caveats

#### Scenario: seed confirm 防止单 seed 误判
- **WHEN** seeds 17、23 and 42 are available for the same readout candidate
- **THEN** summary MUST report per-seed metrics and mean/std aggregation
- **AND** claim gate MUST indicate whether the readout improvement is directionally consistent across a majority of seeds

#### Scenario: query diagnostics 汇入 summary
- **WHEN** attention/query diagnostics are available for readout candidates
- **THEN** summary MUST include query diversity、attention entropy、effective patch count、readout weight summary and diagnostics availability fields
- **AND** missing diagnostics MUST be reported as `missing` or `unavailable` rather than causing the candidate row to disappear
