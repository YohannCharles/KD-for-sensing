# observability-aware-fusion Specification

## Purpose
定义 observability-aware fusion 如何消费 image/GPS reliability metadata、计算 modality weights、执行 adaptive fusion，并在 image degraded 或 missing 时通过 uncertainty gating 使用 JEPA temporal latent fallback。
## Requirements
### Requirement: Observability-aware fusion 输入契约
系统 MUST 提供 observability-aware fusion 模块，用于消费 image/GPS 表征和输入 reliability metadata。模块 MUST 接收 `z_img`、`z_gps`、`image_valid_mask`、`image_observability_score`、`gps_valid_mask`、`gps_delay_steps` 或 manifest 声明的等价字段，并 MUST 输出 fused representation 和可诊断的 modality weights。

#### Scenario: 构建 observability-aware fusion
- **WHEN** 配置声明使用 observability-aware fusion
- **THEN** 系统 MUST 构建接收 image 与 GPS latent 的 fusion 模块
- **AND** 模块 MUST 校验 image/GPS latent 的 batch/time 维兼容
- **AND** 缺失必需 reliability metadata 时 MUST 抛出清晰错误或使用配置声明的 fallback 并记录 warning

#### Scenario: 输出 modality weights
- **WHEN** observability-aware fusion forward 成功
- **THEN** 输出 MUST 包含 fused latent 或 logits 可消费的 representation
- **AND** diagnostics MUST 包含 `w_img`、`w_gps`、`image_observability_score` 和 GPS reliability summary

### Requirement: Reliability weighting 和 adaptive fusion
Observability-aware fusion MUST 根据 image observability 与 GPS reliability 计算 modality weights，并执行 adaptive fusion。默认 fusion 语义 MUST 等价于 `z_fuse = w_img * z_img + w_gps * z_gps` 或在维度不一致时使用显式 projection 后的等价加权融合。

#### Scenario: GPS async 降低 GPS 权重
- **WHEN** `gps_valid_mask` 显示 GPS missing 或 `gps_delay_steps` 较高
- **THEN** fusion MUST 降低 GPS reliability score 或 `w_gps`
- **AND** diagnostics MUST 记录降低权重的原因字段

#### Scenario: 图像缺失降低 image 权重
- **WHEN** `image_valid_mask` 为 false 或 `image_observability_score` 低于阈值
- **THEN** fusion MUST 降低 image reliability score 或 `w_img`
- **AND** physical corruption 与 missing MUST 在 diagnostics 中可区分

### Requirement: 不确定性 gating 与 JEPA fallback
Observability-aware fusion MUST 支持 uncertainty gating。当 image observability 低于阈值且 JEPA temporal prediction 可用时，系统 MUST 能选择 temporal JEPA latent prediction 作为 degraded/missing image 的 fallback，而不是强制使用 raw CNN features。

#### Scenario: 低可观测性触发 JEPA fallback
- **WHEN** `image_observability_score` 低于配置阈值且模型提供 temporal JEPA predicted latent
- **THEN** fusion MUST 使用 predicted latent 或提高其权重
- **AND** diagnostics MUST 记录 fallback 是否触发、触发阈值和使用的 latent source

#### Scenario: fallback 不可用时降级
- **WHEN** 配置启用 JEPA fallback 但当前模型不提供 temporal predicted latent
- **THEN** 系统 MUST 使用配置声明的 fallback 策略
- **AND** 系统 MUST 在 warnings 中记录 unavailable reason

### Requirement: JEPA advantage condition
系统 MUST 支持显式 JEPA advantage condition：当 GPS 为 `C3_random_async` 或 `C4_severe_async`，且 image 为 `D3_motion_blur`、`D4_partial_occlusion`、`D6_burst_missing` 或 `D7_joint_worst_case` 时，Image-JEPA+GPS 配置 MUST 能更多依赖 temporal latent prediction，而不是 raw CNN feature。

#### Scenario: 命中 JEPA advantage condition
- **WHEN** benchmark condition 为 `C3_random_async + D4_partial_occlusion`
- **THEN** Image-JEPA+GPS 模型 MUST 将 condition metadata 传入 fusion/gating
- **AND** observability-aware fusion MUST 标记 `jepa_advantage_condition=true`
- **AND** diagnostics MUST 记录 temporal latent prediction 权重或 fallback 状态

#### Scenario: clean condition 不强制 fallback
- **WHEN** benchmark condition 为 `C0_sync + D0_full_image`
- **THEN** observability-aware fusion MUST 不因 Scenario D 配置强制启用 JEPA fallback
- **AND** Image ResNet+GPS 或 standard fusion baseline MUST 仍可作为 clean ceiling 对照

### Requirement: Logit-level uncertainty fusion
Observability-aware fusion MUST 支持 opt-in 的 logit-level uncertainty/evidence fusion 模式，用于 geometry-prior 模型。该模式 MUST 使用 reliability、uncertainty 或 evidence 信号组合各 branch logits，同时 MUST 不要求普通 baseline 消费这些字段。

#### Scenario: branch uncertainty controls weights
- **WHEN** geometry-prior logit fusion receives image logits、geometry prior logits 和 branch entropy/evidence
- **THEN** fusion MAY reduce the weight of high-uncertainty or low-reliability branches
- **AND** diagnostics MUST record branch entropy/evidence、final weights and unavailable reason

#### Scenario: ordinary baseline ignores metadata
- **WHEN** Image ResNet+GPS、JEPA GPS-query k=4 或其它未 opt-in 的 baseline 在同一 benchmark batch 上运行
- **THEN** reliability、uncertainty 和 branch diagnostic fields MUST NOT be required forward inputs
- **AND** batch runtime MUST allow those models to ignore unsupported metadata

### Requirement: Geometry-prior condition id isolation
Geometry-prior reliability fusion MUST NOT consume benchmark condition identifiers as model inputs. Condition identifiers MAY only be used outside model forward for aggregation, filenames and reports.

#### Scenario: condition id 不进入 fusion input
- **WHEN** batch metadata contains `condition`、`predictive_condition_id`、`gps_condition`、`image_condition`、`c_idx` or `d_idx`
- **THEN** logit fusion input tensor MUST exclude those fields
- **AND** diagnostics MUST record `condition_id_consumed=false`

#### Scenario: condition id 用于 report 分组
- **WHEN** evaluation aggregates P0-P5 or advantage metrics
- **THEN** reports MAY use condition ids for grouping and table labels
- **AND** this grouping MUST happen outside model forward and gate/fusion input construction

### Requirement: GPS reliability in logit fusion
Geometry-prior logit fusion MUST be capable of down-weighting GPS prior when GPS reliability metadata or branch disagreement indicates likely wrong, delayed or unavailable GPS.

#### Scenario: wrong GPS 降低 prior 权重
- **WHEN** `gps_counterfactual_mask=true` or prior-image disagreement exceeds configured threshold
- **THEN** fusion MUST be capable of reducing geometry prior weight
- **AND** diagnostics MUST record the reliability signal or disagreement metric used

#### Scenario: clean high-agreement GPS 可提高 prior 权重
- **WHEN** GPS is valid, delay is low, prior entropy is low and prior-image agreement is high
- **THEN** fusion MAY increase geometry prior weight
- **AND** diagnostics MUST compare clean weight distribution against hard-condition weight distribution

### Requirement: Image observability in logit fusion
Geometry-prior logit fusion MUST be capable of down-weighting image logits when image observability is low, while still protecting clean performance.

#### Scenario: image degradation 降低 image 权重
- **WHEN** `image_valid_mask=false` or `image_observability_score` is below configured threshold
- **THEN** fusion MUST be capable of lowering image branch weight or increasing uncertainty
- **AND** diagnostics MUST distinguish missing image, occlusion, blur and burst missing where metadata is available

#### Scenario: clean condition 不强制降低 image 权重
- **WHEN** condition is clean and image observability is high
- **THEN** fusion MUST NOT force a low image weight solely because geometry prior is enabled
- **AND** clean branch weights MUST be reported as part of clean regression diagnostics

### Requirement: Reliability-aware predictive gate inputs
Observability-aware fusion MUST support opt-in predictive gates that fuse current image latent, temporal predicted latent and GPS-derived residual latent using continuous reliability signals. The gate MUST not require these signals for existing non-predictive baselines.

#### Scenario: Gate 消费连续 reliability fields
- **WHEN** Predictive GPS-query++ enables reliability-aware gate
- **THEN** gate MAY consume `image_valid_mask`、`image_observability_score`、`image_current_missing_mask`、`gps_valid_mask`、`gps_counterfactual_mask`、`gps_delay_steps` and latent consistency scores
- **AND** missing optional fields MUST either use configured fallback values or produce clear warnings

#### Scenario: 普通 baseline 不要求 reliability fields
- **WHEN** Image ResNet+GPS, mean-pooling JEPA or existing GPS-query baseline runs without predictive gate enabled
- **THEN** model forward MUST NOT require new reliability fields
- **AND** existing training/evaluation configs MUST remain runnable

### Requirement: Condition id isolation for predictive gates
Predictive gates MUST NOT directly consume benchmark condition identifiers. Condition identifiers MAY be recorded for diagnostics and aggregation only.

#### Scenario: Gate 输入不包含 condition id
- **WHEN** Predictive GPS-query++ forward receives benchmark metadata containing `condition`、`predictive_condition_id`、`gps_condition`、`image_condition`、`c_idx` or `d_idx`
- **THEN** gate input tensor MUST exclude those fields
- **AND** diagnostics MUST record `condition_id_consumed=false`

#### Scenario: Condition id 可用于 report 分组
- **WHEN** evaluation aggregates diagnostics by condition
- **THEN** reports MAY use condition ids for grouping, filenames and summary tables
- **AND** this grouping MUST occur outside model forward or gate input construction

### Requirement: Predictive branch weight diagnostics
Reliability-aware fusion MUST report how much current image, temporal predicted latent and GPS residual branches contribute to the fused representation.

#### Scenario: 输出 branch weights
- **WHEN** predictive gate forward succeeds
- **THEN** diagnostics MUST include current image weight、temporal predicted latent weight、GPS residual weight or equivalent normalized branch scores
- **AND** diagnostics MUST include batch/time aggregation suitable for per-condition reports

#### Scenario: 低 image observability 提高 predicted latent 使用
- **WHEN** image observability is low and temporal predicted latent is available
- **THEN** gate MUST be capable of increasing predicted latent branch weight according to learned or configured reliability logic
- **AND** diagnostics MUST record whether predicted branch was available and selected more strongly than in clean reference conditions

#### Scenario: wrong GPS 降低 GPS residual 使用
- **WHEN** `gps_counterfactual_mask` is true or GPS reliability score is low
- **THEN** gate MUST be capable of reducing GPS residual branch weight
- **AND** diagnostics MUST record the reliability signal that caused the reduction

### Requirement: No-regret reliability gate
Observability-aware fusion MUST 支持 no-regret reliability gate，用于 anchor-safe reranker。Gate MUST 使用连续 reliability/uncertainty 信号，而不是 benchmark condition id。

#### Scenario: gate 输入
- **WHEN** reranker gate 构建输入
- **THEN** gate MAY 使用 image observability、GPS valid mask、GPS delay、GPS counterfactual mask、anchor entropy、prior entropy 和 branch disagreement
- **AND** gate MUST NOT 使用 condition、suite、P/C/D id 或 claim label

#### Scenario: gate 输出
- **WHEN** gate forward 成功
- **THEN** diagnostics MUST 包含 gate confidence、fallback_to_anchor、fallback reason 和 residual scale
- **AND** clean/high-observability 条件下 fallback behavior MUST 可聚合统计

### Requirement: Anchor fallback branch diagnostics
Reranker-aware observability diagnostics MUST 区分 anchor branch、geometry prior branch 和 residual/rerank branch 的贡献。

#### Scenario: branch contribution
- **WHEN** reranker 改变最终 top prediction
- **THEN** diagnostics MUST 记录 changed_from_anchor=true、selected beam、anchor beam、prior beam、target rank delta 和 DBA delta
- **AND** aggregate MUST 能区分 beneficial、neutral 和 harmful changes

#### Scenario: wrong GPS 降低 prior trust
- **WHEN** gps_counterfactual_mask=true 或 prior-image disagreement 超过阈值
- **THEN** gate MUST 能降低 prior/rerank residual 权重或 fallback anchor
- **AND** diagnostics MUST 记录该 reliability signal

### Requirement: Proto-compatible reliability mask weighted fusion
The system MUST support a lightweight reliability mask weighted fusion option that is compatible with prototype prediction and randomdrop subset training. Missing modalities MUST receive zero weight and available modality weights MUST be normalized over available modalities only.

#### Scenario: missing modality receives zero weight
- **WHEN** reliability fusion receives modality features and an availability mask
- **THEN** every unavailable modality MUST have weight zero within numerical tolerance
- **AND** unavailable modality features MUST NOT contribute to the fused representation

#### Scenario: available weights normalize
- **WHEN** at least one modality is available for a sample
- **THEN** reliability weights over available modalities MUST sum to one within numerical tolerance
- **AND** the fused representation MUST be equivalent to a weighted sum over available modality features

#### Scenario: lightweight implementation boundary
- **WHEN** reliability fusion is enabled for Scene31 subset candidates
- **THEN** the implementation MUST use a small scorer such as pooled feature plus availability or learned modality reliability embeddings
- **AND** it MUST NOT introduce a complex transformer, imputation module or external dependency

#### Scenario: epoch-level reliability log
- **WHEN** training with reliability fusion completes an epoch
- **THEN** the run directory MUST contain or support writing `reliability_weights_epoch.csv`
- **AND** rows MUST include epoch, pattern, modality, mean_weight, std_weight and available_rate

### Requirement: Reliability fusion Scene31 seed extension guard
Proto-compatible reliability mask weighted fusion MUST only be expanded from seed3 to seed4/5 when the Scene31 summary gate remains positive against `proto_randomdrop_subset_es40`.

#### Scenario: seed config equivalence
- **WHEN** reliability fusion seed3, seed4 or seed5 configs are generated
- **THEN** they MUST match seed1/2 on exposure, proto model family, reliability fusion enabled state, mode `mask_weighted` and max epoch 40
- **AND** only the seed and run name MAY differ

#### Scenario: unrelated methods remain disabled
- **WHEN** reliability fusion continuation configs are generated or run
- **THEN** condBTAPA, weakKD, MPDRO, beamsoft, PatternFiLM, AMR and AMBER MUST be disabled
- **AND** no new transformer, imputation module or external dependency MAY be introduced for this continuation

#### Scenario: expand decision is summary-driven
- **WHEN** seed3 summary status is not `candidate_continue_to_seed5`
- **THEN** seed4/5 MUST remain a prepared explicit group only
- **AND** the default runner group MUST NOT run seed4/5 automatically

