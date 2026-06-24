# predictive-jepa-robustness Specification

## Purpose
Predictive Robustness 定义 Image+GPS JEPA predictive hybrid 的当前评估边界，用于检查当前图像不可观测、历史视觉仍可用或 GPS plausibly-wrong 时，预测式 JEPA 表征是否能比 Image ResNet+GPS 保持更稳的 beam prediction。该 capability 是 pending/unverified 的 current workflow：`P4_joint_predictive_recovery` 可作为训练/curriculum profile，完整 claim 仍必须来自 P0-P5 condition-level benchmark、strict comparable train-then-evaluate provenance 和 Image ResNet+GPS margin 审计。Predictive metrics 是主 claim 口径，Scenario D / CxD 结果只作为 overall sanity，不替代 P-suite provenance。
## Requirements
### Requirement: Predictive Robustness 主场景
系统 MUST 提供 Predictive Robustness benchmark suite，用于评估当前图像不可观测或 GPS 可信但错误时，JEPA 是否能利用历史视觉上下文和预测 latent 保持 beam prediction 性能。该 suite MUST 保留 P0-P5 condition-level benchmark 作为兼容鲁棒性证据，但融合机制区分 MUST 结合正交 CxD/A-slice 诊断证据；现有 C0-C4 x D0-D7 matrix MUST 继续作为 overall sanity。

#### Scenario: 解析 canonical predictive robustness suite
- **WHEN** benchmark manifest 引用 `predictive_jepa_robustness` canonical preset
- **THEN** 系统 MUST 至少支持 `P0_clean_current`、`P1_current_frame_missing_history_available`、`P2_semantic_occlusion_history_available`、`P3_plausible_wrong_gps_current_image`、`P4_joint_predictive_recovery` 和 `P5_novel_weather_history_available`
- **AND** 每个 condition MUST 记录 image/GPS operator 参数、history window、seed、split、difficulty digest 和 replay metadata
- **AND** 所有 condition MUST 保持 `target_beam`、`beam_power`、sample id 和 split metadata 不变

#### Scenario: P-suite 不单独替代融合机制诊断
- **WHEN** predictive robustness benchmark 完成
- **THEN** 输出 MUST 同时记录 P0-P5 regional metrics 和可选 CxD/A-slice diagnostic metrics
- **AND** report MUST 明确 P0-P5 mean 不得单独作为融合机制区分的主证据
- **AND** 当 reused-weight fusion diagnostic metrics 可用时，report MUST 优先使用正交 CxD/A-slice 指标解释融合行为

### Requirement: JEPA predictive hybrid fusion 模型组
系统 MUST 支持一个 JEPA predictive hybrid fusion 模型组，用于比较 Image ResNet+GPS、现有 JEPA baselines 与新增预测式 JEPA 架构。该模型组 MUST 基于模块化组件实现，并 MUST 不要求恢复旧 KD、HiST 或 residual 研究线。

#### Scenario: 模型组可由配置声明
- **WHEN** 配置声明 `model_group: jepa_predictive_hybrid` 或等价 metadata
- **THEN** 系统 MUST 能构建 JEPA context image encoder、hybrid residual query pooler、temporal predicted latent branch、feature-consistency gate 和 beam head
- **AND** 模型输出 MUST 继续包含现有 beam logits、Top-K/DBA 可评价字段和 runtime metadata
- **AND** 默认 Image ResNet+GPS、JEPA GPS-biased 和 JEPA GPS-query-pool 配置 MUST 不被静默替换

#### Scenario: Gate 不读取 condition id
- **WHEN** predictive hybrid 模型在 Predictive Robustness 或 CxD benchmark 中 forward
- **THEN** feature-consistency gate MUST NOT 直接消费 `c_idx`、`d_idx`、`predictive_condition_id` 或 condition string
- **AND** gate diagnostics MUST 说明权重来自 latent consistency、valid masks、observability score、GPS delay/reliability 或等价特征信号

### Requirement: 5 个百分点 claim 口径
系统 MUST 为 predictive robustness 主 claim 提供严格可比较的 margin-vs-ResNet 口径。只有在 split、sample_count、label space、metric profile、difficulty digest 和 seed 可比时，系统 MAY 将 `jepa_predictive_hybrid` 相对 Image ResNet+GPS 的 predictive DBA margin 标记为 P-suite claim；融合机制 claim MUST 另行满足正交 CxD/A-slice 诊断证据。

#### Scenario: 计算 predictive DBA margin
- **WHEN** benchmark 同时包含 `jepa_predictive_hybrid` 与 Image ResNet+GPS strict comparable rows
- **THEN** 系统 MUST 计算 `predictive_dba`、`predictive_top1`、`resnet_predictive_dba`、`margin_vs_resnet_dba` 和 `claim_pass_5pt`
- **AND** `claim_pass_5pt` MUST 仅在 `margin_vs_resnet_dba >= 0.05` 且 comparability status 为 strict 时为 true

#### Scenario: P-suite claim 与融合诊断 claim 分离
- **WHEN** P0-P5 margin 达到阈值但 reused-weight CxD/A-slice diagnostic metrics 缺失或 not-comparable
- **THEN** 系统 MAY 标记 P-suite robustness claim
- **AND** 系统 MUST 不将该结果表述为融合机制已被正交诊断验证

#### Scenario: smoke 不升级为真实 claim
- **WHEN** benchmark 使用 synthetic metrics、mock weights、partial model set 或 missing checkpoint
- **THEN** 系统 MUST 将 claim status 标记为 `mock/smoke`、`pending` 或 `unavailable`
- **AND** docs 和 result claims registry MUST 不记录真实性能数值

### Requirement: Predictive Robustness 输出产物边界
Predictive Robustness workflow MUST 将真实训练、评估和分析产物写入 ignored `outputs/`、`logs/` 或 manifest 指定目录。源码变更 MUST 只包含实现、测试、配置、OpenSpec 和文档账本摘要。

#### Scenario: 写出结构化产物
- **WHEN** predictive robustness benchmark 完成
- **THEN** 输出目录 MUST 包含 machine-readable manifest、condition-level metrics、regional summary、margin-vs-ResNet summary 和 warnings
- **AND** 可选图表 MUST 在 manifest 中登记，但真实 PNG/SVG/PDF 不得提交到源码

### Requirement: Predictive Robustness 文档治理
Predictive Robustness 作为 current capability 时，系统 MUST 在 spec Purpose、lifecycle inventory、主线模型目录、实验协议表和 claim 账本中明确它是 pending/unverified 的 current workflow capability，而不是已经完成真实数值 claim 的结果。

#### Scenario: current capability 但 claim 未验证
- **WHEN** 文档登记 `predictive-jepa-robustness` 为 current capability
- **THEN** 文档 MUST 明确真实 claim 仍需要 strict comparable train-then-evaluate run
- **AND** synthetic metrics、mock weights、partial model set 或 allow_missing_artifacts MUST 只能标记为 `mock/smoke`、`pending` 或 `unavailable`

#### Scenario: spec Purpose 描述真实能力边界
- **WHEN** 维护者打开 `openspec/specs/predictive-jepa-robustness/spec.md`
- **THEN** `## Purpose` MUST 说明 Predictive Robustness 用于评估 JEPA 预测表征在当前图像不可观测或 GPS plausibly-wrong 条件下的鲁棒性
- **AND** Purpose MUST 不包含待定占位符、归档 scaffold 文案或未验证数值 claim

### Requirement: 训练 profile 与完整 P-suite benchmark 分离
Predictive Robustness MUST 区分训练 difficulty profile、evaluation difficulty profile 和 P0-P5 benchmark suite。单个训练 profile 或 clean evaluation profile MUST NOT 被描述为完整 P0-P5 regional benchmark。

#### Scenario: 训练配置只启用部分 predictive condition
- **WHEN** 派生训练配置只声明 `P4_joint_predictive_recovery` 或其它单个 predictive condition
- **THEN** 文档 MUST 将其描述为训练/curriculum profile
- **AND** 文档 MUST 指向 benchmark manifest 或本地 real manifest 才能执行完整 P0-P5 regional evaluation

#### Scenario: 完整 benchmark claim 需要 P-suite provenance
- **WHEN** claim registry 准备将 Predictive Robustness 升级为真实 claim
- **THEN** provenance MUST 包含 P0-P5 condition-level metrics、strict comparability fields、difficulty digest、seed、split、sample_count 和 Image ResNet+GPS baseline
- **AND** 缺少任一 required provenance 时 claim status MUST 保持 `pending`、`unavailable` 或 `not_comparable`

### Requirement: GPS-query advantage slice
Predictive Robustness MUST support an optional GPS-query advantage slice that evaluates whether GPS-conditioned JEPA prediction helps under visual ambiguity, beam-offset-constrained wrong GPS, and combined GPS/image reliability degradation. This slice MUST supplement, not replace, the canonical P0-P5 benchmark.

#### Scenario: Advantage slice 不替代 P0-P5
- **WHEN** benchmark manifest enables GPS-query advantage slice
- **THEN** output MUST still include canonical P0-P5 condition-level metrics when a predictive robustness claim is requested
- **AND** reports MUST label advantage slice as mechanism/diagnostic evidence rather than the primary P-suite claim

#### Scenario: Advantage slice 包含关键条件
- **WHEN** GPS-query advantage slice is normalized
- **THEN** it MUST include conditions covering visual ambiguity, beam-offset-constrained wrong GPS, and at least `C3_random_async` or `C4_severe_async` combined with one of `D3_motion_blur`、`D4_partial_occlusion`、`D6_burst_missing`、`D7_joint_worst_case`
- **AND** each condition MUST record seed、split、difficulty digest、operator params、fallback count and sample count

#### Scenario: Advantage slice 输出 per-condition margin
- **WHEN** advantage slice evaluation completes with strict comparable model rows
- **THEN** output MUST include per-condition DBA/Top-K metrics and margins against `Image ResNet+GPS` and current `JEPA GPS-query k=4` or configured GPS-query baseline
- **AND** missing strict comparable rows MUST mark the slice as unavailable or not-comparable

### Requirement: GPS-query++ strict comparison set
Predictive Robustness real evaluation for this workflow MUST compare Predictive GPS-query++ against Image ResNet+GPS and a matched current GPS-query baseline under the same protocol.

#### Scenario: Strict model groups are present
- **WHEN** a real Predictive GPS-query++ benchmark manifest is used for claim-oriented evaluation
- **THEN** manifest MUST include Image ResNet+GPS, current JEPA GPS-query baseline, and Predictive GPS-query++ model groups
- **AND** model groups MUST declare config path、weights path、checkpoint provenance、metric profile、split、sample count and label space

#### Scenario: 同协议字段一致
- **WHEN** strict comparison rows are aggregated
- **THEN** rows MUST share history window、GPS input/source window、prediction horizon、scene set、seed、difficulty digest、distance metric and beam label space
- **AND** any mismatch MUST prevent claim upgrade and appear in warnings

### Requirement: GPS-query++ claim gate
Predictive GPS-query++ claim status MUST require both canonical P0-P5 evidence and advantage-slice evidence. Advantage-slice improvements MAY explain mechanism but MUST NOT alone promote a claim.

#### Scenario: Claim gate 计算
- **WHEN** P0-P5 and advantage slice metrics are available for strict comparable models
- **THEN** system MUST compute P-suite margin vs Image ResNet+GPS、advantage-slice margin vs Image ResNet+GPS、advantage-slice margin vs current GPS-query baseline and claim pass flags
- **AND** primary claim pass MUST remain based on canonical predictive robustness criteria and configured margin threshold

#### Scenario: Advantage slice 单独提升不升级 claim
- **WHEN** Predictive GPS-query++ outperforms baselines on advantage slice but not on canonical P0-P5
- **THEN** report MUST describe the result as mechanism evidence or targeted advantage
- **AND** claim status MUST remain pending, partial, unavailable or not-comparable according to provenance

### Requirement: GPS-query++ diagnostics bundle
Predictive Robustness MUST provide a diagnostics bundle for GPS-query++ evaluations that explains branch usage without treating explanations as causal proof.

#### Scenario: 输出 gate 和 latent consistency diagnostics
- **WHEN** Predictive GPS-query++ evaluation emits diagnostics
- **THEN** bundle MUST include gate weight summaries、branch availability、latent consistency summaries、fallback counts and per-condition margin tables
- **AND** diagnostics MUST be linked from a machine-readable manifest

#### Scenario: 解释性图不构成 claim
- **WHEN** report includes attention, gate, t-SNE/PCA, rank CDF or latent consistency figures
- **THEN** report MUST state that these figures are explanatory diagnostics
- **AND** numeric claim MUST still be based on strict metrics and provenance

