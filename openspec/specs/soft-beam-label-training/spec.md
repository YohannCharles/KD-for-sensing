# soft-beam-label-training Specification

## Purpose
定义 beam soft label 的 batch 字段、source/target 域生成规则、soft-target supervised loss 消费方式，以及 hard-label 验证评估保持不变的训练契约，确保 soft target 只在允许的训练域提供额外监督而不污染 target-side 评估结论。
## Requirements
### Requirement: Beam soft target batch contract

系统 SHALL 在启用 beam soft label 时为 beam selection batch 提供 `target_beam_distribution`，该字段 MUST 与 `target_beam` 的 future horizon 对齐，并表示每个 future slot 上所有 beam class 的概率分布。

#### Scenario: batch 包含 soft distribution
- **WHEN** Dataset 样本启用 soft beam label 并存在 `num_pred=3`、`num_classes=64`
- **THEN** 样本 MUST 包含 shape 为 `[3, 64]` 的 `target_beam_distribution`
- **AND** 每个 horizon 的分布和 MUST 在数值容差内等于 1
- **AND** 样本 MUST 继续包含 shape 为 `[3]` 的 hard `target_beam`

#### Scenario: hard label 指标和评估 loss 保留
- **WHEN** 训练或验证 batch 同时包含 `target_beam` 和 `target_beam_distribution`
- **THEN** 系统 MUST 使用 `target_beam` 计算 validation/evaluation loss、top-k、DBA、split 诊断和 hard-label 评价指标

### Requirement: Soft target generation

系统 SHALL 按训练域区分 beam soft target 来源：source 域 MAY 使用 beam power/RSS profile 归一化生成 soft target distribution；target 快速适应域 MUST NOT 读取或使用 target-side power/RSS profile，只能基于 hard beam label 和码本邻接关系生成 circular Gaussian soft distribution。

#### Scenario: source 域使用 beam power/RSS oracle
- **WHEN** source-domain future beam path 指向有效的 64 维有限 beam power/RSS 向量
- **THEN** 系统 MUST 将该向量转换为非负概率分布作为对应 horizon 的 soft target
- **AND** hard `target_beam` MUST 仍等于该向量的 argmax

#### Scenario: source 域使用 Gaussian fallback
- **WHEN** source-domain future beam power/RSS 向量缺失、维度错误、全零或非有限
- **THEN** 系统 MUST 基于 hard `target_beam` 和配置的 `sigma` 生成 Gaussian soft target
- **AND** circular 模式启用时，beam 0 与最后一个 beam MUST 按环形距离相邻

#### Scenario: target 域禁止使用 beam power/RSS oracle
- **WHEN** target-domain future beam path 指向有效的 beam power/RSS 向量
- **THEN** 系统 MUST NOT 读取或使用该 target-side power/RSS 向量生成训练 soft target
- **AND** 系统 MUST 基于 hard `target_beam`、配置的 `sigma` 和 circular beam distance 生成 Gaussian soft target
- **AND** beam 0 与最后一个 beam MUST 按环形距离相邻

### Requirement: Soft target supervised loss
系统 SHALL 在 beam soft target 可用且 soft target loss 启用时，使用 soft target distribution 计算主 beam supervised loss；若 soft target 不可用，MUST 回退到 hard-label loss。该流程 MUST 不经过 distillation runtime。

#### Scenario: supervised 主 loss 使用 soft target
- **WHEN** batch 包含 `target_beam_distribution` 且 `loss.soft_targets.enabled=true`
- **THEN** supervised loss MUST 消费 soft target distribution
- **AND** `loss/beam` 和 `loss/primary` MUST 记录 soft-target supervised loss
- **AND** diagnostics MUST 不记录 `loss/distillation`

#### Scenario: validation 和 evaluation 不使用 soft target
- **WHEN** 验证 DataLoader batch 包含 `target_beam_distribution`
- **THEN** validation/evaluation loss MUST 使用 hard `target_beam`
- **AND** validation/evaluation top-k/DBA 指标 MUST 继续使用 hard `target_beam`

### Requirement: Configuration and fallback

系统 SHALL 提供配置开关控制 soft label 生成和 soft-target loss 消费，并允许显式关闭以复现 hard-label baseline。

#### Scenario: 显式关闭 soft target
- **WHEN** `loss.soft_targets.enabled=false` 或 `data.dataset.soft_beam_labels.enabled=false`
- **THEN** 系统 MUST 使用现有 hard-label supervised loss 路径

#### Scenario: 默认配置暴露参数
- **WHEN** 解析默认训练配置或 canonical beam objective 配置
- **THEN** 配置 MUST 包含 soft target 相关参数，包括 enable 开关、source、target_source、domain、sigma、circular、temperature 和 ignore index

### Requirement: Beam soft target 不等同于 KD
beam-aware soft label、angular soft target 和 beam smoothing target MUST 被视为 beam-space prior 或 supervised label smoothing，而不是 teacher-student KD。soft target loss MUST 不被命名、记录或汇总为 distillation loss。

#### Scenario: supervised soft target 日志命名
- **WHEN** supervised training 使用 `target_beam_distribution` 或等价 beam soft target
- **THEN** loss diagnostics MUST 使用 `loss/beam_soft_target`、`loss/beam_smoothing` 或等价非 KD 命名
- **AND** diagnostics MUST 不生成新的 `loss/kd_soft_label` 或 `loss/distillation` 字段表示该监督项

#### Scenario: soft target metadata 记录来源
- **WHEN** batch 或 run metadata 记录 beam soft target 的来源
- **THEN** metadata MUST 区分 `source_power_oracle`、`gaussian_from_hard_label`、`angular_smoothing` 或等价来源
- **AND** 新写出的 metadata MUST 不把 beam soft target 标记为 KD soft target

### Requirement: 旧 KD soft target 命名只读迁移
历史 artifact 中的 KD soft target 命名 MAY 只读兼容；当前训练 MUST 分离为 beam soft target supervised loss，并拒绝旧 distillation 配置。

#### Scenario: 历史 KD soft target 只读
- **WHEN** 历史 artifact 包含 `kd_soft_label` 或等价字段
- **THEN** 系统 MAY 只读兼容该字段
- **AND** 新训练产物 MUST 使用 beam soft target 命名

#### Scenario: evaluation 不使用 soft target 或 KD target
- **WHEN** validation 或 evaluation batch 同时包含 hard label、beam soft target 和可选 teacher output
- **THEN** hard-label Top-K、DBA、NRP 和 beam power 指标 MUST 使用 hard `target_beam`
- **AND** evaluation summary MUST 不用 soft target 指标替代 hard-label 主指标

### Requirement: V8 target adaptation beam topology soft labels
系统 MUST 支持在 `v8_target_prior_head` target adaptation 中基于 hard beam label 生成 beam topology soft label，并将其作为 supervised beam smoothing loss 使用。该 loss MUST 使用非 KD 命名和记录。

#### Scenario: 从 target support hard label 生成 soft label
- **WHEN** `hist_beam.variant=v8_target_prior_head` 且 `hist_beam.v8.use_soft_beam_label=true`
- **THEN** 系统 MUST 基于 labeled target_adapt support hard beam label 和 `hist_beam.v8.soft_label_sigma` 生成 beam soft distribution
- **AND** 每个 soft distribution 的概率和 MUST 在数值容差内等于 1
- **AND** 该生成过程 MUST NOT 读取 target-side beam_power、RSS profile、path fields、CSI 或 target_test label

#### Scenario: V8 soft label loss 使用非 KD 命名
- **WHEN** v8 target adaptation 使用 beam topology soft label 计算 supervised loss
- **THEN** diagnostics MUST 使用 `hist/v8/loss_final_soft_ce`、`loss/beam_soft_target`、`loss/beam_smoothing` 或等价非 KD 命名
- **AND** diagnostics MUST NOT 将该 loss 记录为 `loss/kd_soft_label`、`loss/distillation` 或 teacher-student KD loss

#### Scenario: soft label 关闭时回退 hard CE
- **WHEN** `hist_beam.v8.use_soft_beam_label=false`
- **THEN** v8 supervised final loss MUST 使用 hard-label CE 或明确记录 supervised final loss 不可用原因
- **AND** evaluation Top-K、NRP 和 prediction histogram MUST 继续使用 hard beam label

### Requirement: Soft beam labels follow calibrated topology
当 MMW dataset 启用 beam label calibration 且 soft beam label 启用时，系统 MUST 在 calibrated label space 中生成或重排 `target_beam_distribution`，并 MUST 保持该分布与 hard `target_beam` 的 horizon 和 class order 一致。

#### Scenario: Gaussian soft label 使用 calibrated label
- **WHEN** target-domain soft label 基于 hard label 和 circular Gaussian 生成，且 MMW calibration 已启用
- **THEN** Gaussian center MUST 使用 calibrated `target_beam`
- **AND** circular distance MUST 在 calibrated class order 中计算

#### Scenario: source power soft label 重排到 calibrated class order
- **WHEN** source-domain soft label 从 raw beam power/RSS vector 构造，且 MMW calibration 已启用
- **THEN** distribution class 维 MUST 按 raw→calibrated mapping 重排
- **AND** distribution mask 和 horizon 对齐 MUST 保持不变

#### Scenario: hard-label evaluation 仍使用 calibrated hard label
- **WHEN** validation 或 evaluation batch 同时包含 calibrated `target_beam` 和 `target_beam_distribution`
- **THEN** hard-label Top-K、DBA 和 validation/evaluation loss MUST 使用 calibrated `target_beam`
- **AND** metrics metadata MUST declare the calibrated label space

### Requirement: Circular soft target loss for MMW Town GPS v2
系统 MUST 为 MMW Town GPS-only v2 提供 circular soft target supervised loss。soft target MUST 使用 circular beam distance 构造 Gaussian distribution，并 MUST 归一化为概率分布。

#### Scenario: circular soft target wrap-around
- **WHEN** `num_beams=64` 且 target beam 为 `0`
- **THEN** circular soft target MUST 将 beam `63` 作为距离 1 的邻居
- **AND** distribution 的概率和 MUST 在数值容差内等于 1

#### Scenario: circular soft CE 参与训练
- **WHEN** v2 配置 `loss.type: circular_soft_ce`
- **THEN** 系统 MUST 使用 circular soft target 计算 supervised beam loss
- **AND** validation/evaluation metrics MUST 继续使用 hard target label

### Requirement: Focal and class-balanced circular soft loss
系统 MUST 支持在 circular soft CE 上启用 focal gamma 和 class-balanced weighting。class weight 模式 MUST 至少支持 `none`、`inverse_freq`、`inverse_sqrt_freq` 和 `effective_num`，并 MUST 从当前训练 split 的 label histogram 计算。

#### Scenario: class weight 默认关闭
- **WHEN** v2 配置未显式启用 class-balanced weighting
- **THEN** 系统 MUST 使用 `class_weight: none`
- **AND** weighted ablation MUST NOT 污染 unweighted ablation 的 summary

#### Scenario: effective_num 权重记录元数据
- **WHEN** v2 配置 `loss.class_weight: effective_num`
- **THEN** 系统 MUST 使用配置的 beta 计算 class weights
- **AND** run metadata MUST 记录 beta、label histogram、权重归一化策略和 fit split

### Requirement: Circular soft loss is not KD
MMW Town GPS v2 的 circular soft CE、focal circular soft CE 和 class-balanced circular loss MUST 作为 supervised beam smoothing loss 记录，不得标记为 teacher-student KD。

#### Scenario: loss 日志使用非 KD 命名
- **WHEN** v2 训练使用 circular soft loss
- **THEN** loss diagnostics MUST 使用 `loss/beam_circular_soft_ce`、`loss/beam_focal_circular_soft_ce` 或等价非 KD 命名
- **AND** diagnostics MUST NOT 生成新的 `loss/distillation` 或 `loss/kd_soft_label` 字段表示该 supervised loss

### Requirement: DBA-aware supervised beam loss
系统 MUST 支持 opt-in DBA-aware supervised beam loss。该 loss MUST 利用 beam topology、calibrated class order 或 beam distance 奖励接近 target 的预测，但 validation/evaluation MUST 继续使用 hard `target_beam` 的 Top-K 和 DBA。

#### Scenario: distance-aware loss 使用 hard label 构造
- **WHEN** 配置启用 DBA-aware loss 且 batch 只有 hard `target_beam`
- **THEN** 系统 MUST 基于 hard target 和 beam topology 构造 distance-aware target 或 weighting
- **AND** loss diagnostics MUST 记录 topology mode、sigma/temperature、distance mode 和 class order

#### Scenario: evaluation 仍使用 hard target
- **WHEN** validation 或 evaluation batch 包含 DBA-aware soft target 或 beam topology target
- **THEN** validation/evaluation Top-K、DBA 和 primary metric MUST 使用 hard `target_beam`
- **AND** summary MUST 不用 soft target accuracy 替代 hard-label 指标

### Requirement: Geometry-prior loss logging is non-retired
Geometry-prior loss、DBA-aware loss 和 teacher-guided stabilization MUST 使用当前 supervised/objective 命名，不得复用 retired KD loss 命名或旧 distillation runtime。

#### Scenario: supervised beam smoothing 命名
- **WHEN** 训练使用 circular Gaussian、distance-aware CE、ordinal/EMD-style loss 或 focal circular loss
- **THEN** logs MUST 使用 `loss/beam_dba_aware`、`loss/beam_circular_soft_ce`、`loss/beam_distance_ce` 或等价非 retired 命名
- **AND** logs MUST 不将该 loss 记录为 `loss/distillation` 或旧 `loss/kd_soft_label`

#### Scenario: teacher guidance 命名
- **WHEN** 训练使用 teacher logits/probabilities 作为 stabilization
- **THEN** logs MUST 使用 `loss/teacher_guidance` 或 `loss/geometry_teacher_kl`
- **AND** run metadata MUST 标记 teacher guidance 为 opt-in stabilization，而不是 retired KD workflow

### Requirement: DBA-aware loss ablation isolation
DBA-aware loss MUST 可独立开关，并 MUST 支持与 geometry prior、teacher guidance 和 curriculum 分别做 ablation。

#### Scenario: loss 关闭回退 baseline CE
- **WHEN** `loss.dba_aware.enabled=false` 或未声明 DBA-aware loss
- **THEN** 系统 MUST 使用现有 hard-label CE 或当前配置声明的 primary beam loss
- **AND** logs MUST 不生成 DBA-aware loss 字段

#### Scenario: ablation metadata 记录
- **WHEN** DBA-aware loss 与 teacher guidance 或 geometry-prior fusion 同时启用
- **THEN** metadata MUST 分别记录每个 objective 的 enabled、weight、temperature/sigma 和 sample count
- **AND** diagnostics MUST 能区分主 beam loss、DBA-aware loss 和 teacher-guidance loss

