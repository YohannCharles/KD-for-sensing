## ADDED Requirements

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
