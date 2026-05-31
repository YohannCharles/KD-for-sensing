## ADDED Requirements

### Requirement: V9 input-conditioned target adaptation
系统 MUST 提供 `v9_input_conditioned_target_adaptation` HiST-Beam 变体或等价 v8/v9 mode，用于在冻结 source backbone 的前提下组合 target-specific logits、受限 global target prior 和 sample-conditioned prototype logits。该能力 MUST 默认不改变 v0-v8 配置行为。

#### Scenario: 构建 V9 input-conditioned 变体
- **WHEN** 用户配置 `hist_beam.variant=v9_input_conditioned_target_adaptation` 或等价 v9 mode
- **THEN** 系统 MUST 构建 target adapter、target head、target prior bias、受限 `beta_prior` 和 prototype logits 计算组件
- **AND** 模型 forward MUST 输出 `logits`、`beam_logits`、`logits_final`、`target_logits`、`target_prior_bias`、`prototype_logits`、`features` 和 `hist_beam` metadata
- **AND** source logits MAY 作为诊断输出保留，但默认 MUST NOT 参与 final prediction

#### Scenario: V9 final logits 组合
- **WHEN** `hist_beam.v9.use_prototype_logits=true` 且 `hist_beam.v9.use_target_prior=true`
- **THEN** `logits_final` MUST 等价于 target logits、受限 target prior term 和 prototype logits 的配置化加权组合
- **AND** run metadata MUST 记录 `beta_prior_effective`、`eta_prototype`、prototype type、prototype temperature 和是否使用 source logits

#### Scenario: 旧 V8 默认行为保持不变
- **WHEN** 用户继续配置 `hist_beam.variant=v8_target_prior_head`
- **THEN** 系统 MUST 保持 v8 既有 forward、loss 和 freeze policy 语义
- **AND** v9 prototype logits、prior dropout 和 widened-prior marginal KL MUST 只在显式启用时生效

### Requirement: V9 global prior strength control
系统 MUST 为 v9 target prior 提供强度约束，避免 global prior 无界支配 final prediction。可训练 beta MUST 支持上限参数化，训练期间 MUST 支持 prior dropout，且 fixed beta ablation MUST 可配置。

#### Scenario: beta prior 上限参数化
- **WHEN** `hist_beam.v9.learnable_beta_prior=true` 且 `hist_beam.v9.beta_prior_max` 大于 0
- **THEN** 系统 MUST 将有效 beta 限制在 `[0, beta_prior_max]`
- **AND** diagnostics MUST 记录 beta 初始值、最终值、上限和参数化方式

#### Scenario: fixed beta ablation
- **WHEN** `hist_beam.v9.learnable_beta_prior=false`
- **THEN** 系统 MUST 使用配置的 fixed `beta_prior`
- **AND** optimizer MUST NOT 更新 beta prior 参数
- **AND** metrics MUST 标记该 run 为 fixed beta ablation

#### Scenario: prior dropout 训练生效
- **WHEN** `hist_beam.v9.prior_dropout` 大于 0 且模型处于训练阶段
- **THEN** 系统 MUST 按配置概率在 batch 或 sample 级别丢弃 global prior term
- **AND** diagnostics MUST 记录 prior dropout 概率和实际启用状态
- **AND** evaluation MUST 默认不随机丢弃 prior，除非用户显式请求诊断模式

### Requirement: V9 target support prototype logits
系统 MUST 支持基于 target_adapt labeled support features 构造 prototype logits，作为 sample-conditioned local calibration。prototype 构造 MUST 只使用 target_adapt labeled support，不得读取 target_test 或禁用 target-side oracle 字段。

#### Scenario: 构造 beam-level prototype logits
- **WHEN** `hist_beam.v9.prototype_type=beam` 且 target support features 可用
- **THEN** 系统 MUST 按 beam label 聚合 support feature prototype
- **AND** query prototype logits MUST 基于 query feature 与 beam prototype 的相似度或距离计算
- **AND** 缺失 support 的 beam MUST 被 mask、平滑 fallback 或记录 unavailable，不得伪造高置信 prototype

#### Scenario: 构造 sector-level prototype logits
- **WHEN** `hist_beam.v9.prototype_type=sector`
- **THEN** 系统 MUST 按 `sector_size` 聚合 support feature prototype
- **AND** 系统 MUST 将 sector prototype 分数映射到 beam logits 或显式输出 sector-only diagnostics
- **AND** metadata MUST 记录 `sector_size`、sector-to-beam 映射方式和每个 sector 的 support count

#### Scenario: prototype temperature 和权重可配置
- **WHEN** prototype logits 参与 final prediction
- **THEN** 系统 MUST 支持配置 prototype temperature `tau` 和权重 `eta_prototype`
- **AND** metrics MUST 记录 prototype Top-1、Top-3、Top-5、within3、MAE、prediction histogram 或不可用原因

### Requirement: V9 anti-collapse regularization
系统 MUST 支持可选 anti-collapse regularization，用于约束预测边际分布接近 widened target prior，而不是强行接近 uniform distribution。该 loss MUST 只在显式启用时参与训练。

#### Scenario: widened target prior 构造
- **WHEN** `hist_beam.v9.use_widened_prior_marginal_kl=true`
- **THEN** 系统 MUST 从 target_adapt labeled support labels 构造比 support prior 更平滑的 widened target prior
- **AND** 配置 MUST 暴露 widened prior 的 sigma 或 temperature
- **AND** diagnostics MUST 记录 widened prior top beams 和与原 support prior 的差异摘要

#### Scenario: prediction marginal KL loss
- **WHEN** widened-prior marginal KL 启用且 batch final logits 可用
- **THEN** 系统 MUST 基于 batch mean predicted probability 计算 marginal KL loss
- **AND** 该 loss MUST 按配置权重参与 total loss
- **AND** diagnostics MUST 使用非 KD 命名记录该 loss，不得伪装成 distillation loss

#### Scenario: 禁止 uniform collapse 目标
- **WHEN** anti-collapse regularization 启用
- **THEN** 系统 MUST NOT 默认把 prediction marginal 拉向 uniform distribution
- **AND** 若用户显式配置 uniform target，metadata MUST 标记为 diagnostic-only，不得作为默认主实验

### Requirement: V9 collapse diagnostics artifact
HiST-Beam v8/v9 source-only target evaluation 和 adapted target evaluation MUST 能输出 collapse 来源诊断产物，用于区分 source collapse、target prior collapse、target head collapse 和 prototype-conditioned recovery。

#### Scenario: 输出 histogram KL 诊断
- **WHEN** v8 或 v9 adapted target_test evaluation 完成且启用 collapse diagnostics
- **THEN** run directory MUST 包含 `collapse_diagnostics.json` 或等价 artifact
- **AND** artifact MUST 包含 support prior histogram、true histogram、prediction histogram、`kl_pred_support`、`kl_true_support`、`kl_pred_true` 和 `unique_pred_beams`

#### Scenario: 输出分支独立指标
- **WHEN** 模型可计算 target logits、prior term 和 final logits
- **THEN** collapse diagnostics MUST 分别记录 `target_logits_only`、`prior_only` 和 `target_logits_plus_prior` 的 Top-K、within3、MAE 和 prediction top beams
- **AND** 若 prototype logits 可用，diagnostics MUST 额外记录 `prototype_only` 和 `target_prior_plus_prototype` 的等价指标

#### Scenario: 输出 per-true-beam confusion
- **WHEN** target_test label 可用于最终离线评价
- **THEN** collapse diagnostics MUST 输出按 true beam 聚合的 confusion 摘要
- **AND** 摘要 MUST 覆盖 target true histogram 的 top beams
- **AND** 该 confusion MUST NOT 参与 adaptation 训练、prior 初始化、threshold selection 或 early stopping

### Requirement: V9 quick validation experiment modes
系统 MUST 提供小规模 v9 quick validation 实验模式，用于验证 collapse 来源和 prototype/local calibration 的贡献。默认矩阵 MUST 保持小而可解释，且每个 mode MUST 在 metadata 中标记实验目的。

#### Scenario: A3 collapse 来源诊断模式
- **WHEN** 用户启用 v9 Group A quick validation
- **THEN** 系统 MUST 支持 A3-base、A3-no-prior、A3-fixed-beta 和 A3-prior-dropout 四类配置或等价 ablation
- **AND** summary MUST 能横向比较 Top-K、within3、MAE、pred histogram coverage 和 beta diagnostics

#### Scenario: prototype ablation 模式
- **WHEN** 用户启用 v9 Group B quick validation
- **THEN** 系统 MUST 支持 beam prototype only、sector prototype only、A3+beam prototype 和 A3+sector prototype 四类配置或等价 ablation
- **AND** summary MUST 记录 prototype type、sector size、support count coverage 和 prototype unavailable reason

#### Scenario: unlabeled distribution regularization 可选模式
- **WHEN** 用户启用 v9 Group C quick validation
- **THEN** 系统 MUST 只使用 target_adapt 中允许的 labeled support 与未标注样本
- **AND** 系统 MUST NOT 读取 target_test label、beam_power、path fields 或 radio labels 参与 loss、threshold、temperature 或 prototype update
- **AND** 若 protocol metadata 无法证明该使用边界，Group C MUST 默认标记为 disabled 或 ineligible
