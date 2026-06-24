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

### Requirement: CNN/hybrid visual-prior full sweep matrix
系统 MUST 提供 CNN/hybrid JEPA visual-prior full sweep 矩阵。矩阵 MUST 在同一 manifest 中覆盖已有 controls、patch resolution、overlap tokenizer、CNN supervised tokens、CNN JEPA tokens、hybrid tokenizers、pooler/core ablation、teacher-guided stabilization、compute controls 和 seed confirm 候选，并 MUST 为每个候选声明唯一 `variant_id`。

#### Scenario: full mode 展开所有实验族
- **WHEN** 开发者以 full mode 加载或生成 CNN/hybrid visual-prior sweep
- **THEN** manifest MUST 至少包含 `existing_controls`、`patch_resolution_stage1`、`overlap_stage1`、`cnn_supervised_tokens`、`cnn_jepa_tokens`、`hybrid_tokenizers`、`pooler_core_ablation`、`teacher_guided_stabilization`、`compute_controls` 和 `seed_confirm` 实验族
- **AND** 每个候选 MUST 记录 `variant_id`、`family`、`stage_plan`、`visual_encoder.type` 或 `token_source.type`、`pooler.type`、`checkpoint_policy`、`checkpoint_selection`、`run_tier` 和 `availability`

#### Scenario: patch 和 overlap 邻域完整
- **WHEN** full mode 生成 patch/overlap 候选
- **THEN** patch resolution 候选 MUST 至少覆盖 patch size 16、14、12、10 和 8
- **AND** overlap 候选 MUST 至少覆盖 kernel/stride 12/6、14/7、16/8、20/10 和 24/12
- **AND** 每个候选 MUST 记录 image size、token grid、token count、effective stride、token budget 和位置编码策略

#### Scenario: CNN 候选声明局部先验来源
- **WHEN** full mode 生成 CNN token 候选
- **THEN** 候选 MUST 至少覆盖 ResNet18 和 ResNet34
- **AND** 候选 MUST 至少覆盖 layer3、layer4 和 layer3+layer4 token source
- **AND** 候选 MUST 记录 pretrained source、freeze policy、trainable params、total params、token grid、token count 和 supervised/JEPA stage policy

#### Scenario: hybrid 候选声明局部和全局机制
- **WHEN** full mode 生成 CNN+Transformer hybrid 候选
- **THEN** 候选 MUST 至少覆盖 conv stem、local token mixing 和 CvT-like convolutional projection 或等价机制
- **AND** 每个候选 MUST 记录 local prior mechanism、Transformer/global mixing depth、token grid、token count 和 checkpoint policy

### Requirement: Stage-aware job generation
系统 MUST 为 CNN/hybrid visual-prior full sweep 生成 stage-aware job manifest。job manifest MUST 表达 Stage 1 pretraining、supervised downstream、teacher-guided student、re-evaluation 和 summary 的依赖关系。

#### Scenario: Stage 1 候选先训练再下游
- **WHEN** 候选声明 `stage_plan: stage1_then_downstream`
- **THEN** 系统 MUST 生成 Stage 1 pretraining job
- **AND** 系统 MUST 生成 downstream job 且其 `depends_on` MUST 指向对应 Stage 1 checkpoint 或 checkpoint placeholder
- **AND** downstream metadata MUST 记录 Stage 1 config path、checkpoint path、checkpoint epoch/step 和 load policy

#### Scenario: supervised-only anchor 不伪装成 JEPA reuse
- **WHEN** 候选声明 `stage_plan: supervised_only`
- **THEN** 系统 MUST 将 `checkpoint_policy` 标记为 `supervised_only_anchor`
- **AND** summary MUST 不把该候选计入 JEPA checkpoint reuse ranking

#### Scenario: teacher-guided 候选使用当前 teacher_guidance 契约
- **WHEN** 候选声明 `stage_plan: teacher_then_student`
- **THEN** student config MUST 使用当前 `loss.teacher_guidance` 契约
- **AND** student config MUST NOT 使用 retired `distillation` key 或 retired whole-model KD route
- **AND** manifest MUST 记录 teacher variant、teacher checkpoint/logit source、temperature、weight、stop-gradient policy 和 teacher provenance

#### Scenario: checkpoint selection 同时保留 primary 与 best_top1
- **WHEN** full sweep 生成 evaluation jobs
- **THEN** 系统 MUST 为 primary checkpoint selection 生成评估记录
- **AND** 系统 MUST 为 best_top1 checkpoint selection 生成 re-evaluation 记录
- **AND** summary MUST 保留 checkpoint selection 字段，不能合并覆盖不同 selection 的指标

### Requirement: Full sweep runner safety and parallelism
系统 MUST 提供可审计的 full sweep runner 或生成的 shell 脚本。runner MUST 支持 GPU 0-3、最多 8 个项目进程并行、依赖感知调度、resume 和 output-root scoped cleanup。

#### Scenario: runner 使用项目 Python 环境
- **WHEN** runner 执行训练、评估、summary 或 config generation 命令
- **THEN** 每个项目 Python 命令 MUST 使用 `conda run -n kd_mm_beam`
- **AND** runner MUST 将 stdout/stderr 写入声明的 ignored output root 下的日志目录

#### Scenario: runner 限制 GPU 和并行数
- **WHEN** 用户使用默认 runner 设置运行 full sweep
- **THEN** runner MUST 只在 GPU 0、1、2 和 3 上分配项目训练/评估任务
- **AND** 同时运行的项目进程数 MUST NOT 超过 8
- **AND** 每个 job MUST 记录实际分配的 `CUDA_VISIBLE_DEVICES`

#### Scenario: cleanup 只删除 sweep output root
- **WHEN** 用户请求重新运行并清理旧结果
- **THEN** cleanup MUST 只删除当前 change 声明的 `outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep/` 或用户显式传入的 sweep output root
- **AND** cleanup MUST NOT 删除源码、其他实验目录、dataset、tracked weights 或系统配置文件

#### Scenario: resume 跳过已完成任务
- **WHEN** runner 发现 job 的 metrics path 和 success marker 已存在
- **THEN** runner MUST 默认跳过该 job
- **AND** runner MUST 支持 failed-only retry 或 force rerun，并在 job status table 中记录跳过原因

### Requirement: Visual-prior summary, Pareto, and claim gate
系统 MUST 为 CNN/hybrid visual-prior full sweep 输出 machine-readable summary 和 Markdown summary。summary MUST 包含全量结果、strict ranking、family best、checkpoint-selection comparison、params/compute Pareto 和 claim eligibility。

#### Scenario: summary 记录完整 metadata 和主指标
- **WHEN** full sweep summary 生成
- **THEN** 每行 MUST 包含 variant metadata、stage metadata、strict comparability fields、checkpoint policy、checkpoint selection、pretrained/freeze policy、teacher provenance、token metadata、params、trainable params、compute proxy、Top-1、Top-3、Top-5、DBA 和 beam distance 指标
- **AND** 缺失 metrics 的候选 MUST 保留为 `missing`、`failed`、`skipped` 或 `unavailable` 状态，而不是从全量表静默移除

#### Scenario: strict claim gate 过滤不可比结果
- **WHEN** summary 生成 strict ranking
- **THEN** 只有 `strict_comparable: true`、非 smoke-only、split/seed/metric/checkpoint provenance 完整且 metrics 存在的候选可以进入 strict ranking
- **AND** 不满足条件的候选 MUST 只能出现在 full table 或 diagnostics table

#### Scenario: Pareto 表区分架构收益和规模收益
- **WHEN** summary 生成 Pareto 或 compute-control 报告
- **THEN** 系统 MUST 按 DBA、Top-1、trainable params、total params、token count 和 compute proxy 生成 Pareto candidates
- **AND** 系统 MUST 标记 CNN/ImageNet/frozen/teacher-guided 候选，使用户能区分局部先验、预训练、冻结策略和模型规模贡献

#### Scenario: 全量 seed 结果可聚合
- **WHEN** 同一 `variant_id` 存在 seeds 17、23 和 42 的结果
- **THEN** summary MUST 输出 per-seed 指标和 mean/std 聚合
- **AND** strict ranking MUST 明确使用单 seed、mean 或 best selection 的排序策略

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

