## ADDED Requirements

### Requirement: V8 target-prior HiST-Beam 变体
系统 MUST 提供可通过配置构建的 `v8_target_prior_head` HiST-Beam 变体。该变体 MUST 保留 source/shared logits 作为诊断输出，但默认最终 beam prediction MUST 由 target-specific logits 和 target prior bias 组成，而不是由 frozen source logits 加 private residual 主导。

#### Scenario: 构建 V8 target-prior 变体
- **WHEN** 用户配置 `hist_beam.variant=v8_target_prior_head`
- **THEN** 系统 MUST 构建 target adapter、target head、target prior bias 和可配置 `beta_prior`
- **AND** 模型 forward 输出 MUST 至少包含 `logits`、`beam_logits`、`logits_final`、`target_logits`、`target_prior_bias`、`features` 和 `hist_beam` metadata
- **AND** 若存在 source/shared beam logits，模型 MUST 以 `source_logits` 或兼容诊断键输出它们

#### Scenario: 默认 final logits 不使用 source logits
- **WHEN** `hist_beam.v8.use_source_logits_in_final=false`
- **THEN** `logits`、`beam_logits` 和 `logits_final` MUST 等价于 `target_logits + beta_prior * target_prior_bias`
- **AND** 系统 MUST NOT 将 v7 的 `logits_shared + alpha * delta_logits_private` 作为 v8 默认 final prediction

#### Scenario: 显式启用 source logits 融合
- **WHEN** `hist_beam.v8.use_source_logits_in_final=true`
- **THEN** 系统 MUST 按配置的 `lambda_src`、`lambda_tgt` 和 `beta_prior` 合成 final logits
- **AND** run metadata MUST 记录 source logits 参与了 final prediction

### Requirement: V8 target prior 初始化
系统 MUST 支持仅基于 target_adapt labeled support labels 初始化 v8 target prior。初始化 MUST 使用 Gaussian-smoothed beam histogram，写入模型 `target_prior_bias`，并可选择将该 bias 作为可训练参数继续 adaptation。

#### Scenario: 从 support labels 初始化 smoothed prior
- **WHEN** target adaptation 已选出 labeled target_adapt support subset 且 label budget 大于 0
- **THEN** 系统 MUST 使用这些 support labels 计算 Gaussian-smoothed beam prior
- **AND** 系统 MUST 将 `log(prior)` 写入模型 `target_prior_bias`
- **AND** 日志或 metrics MUST 记录 `[v8] target support label hist`、`[v8] smoothed target prior top beams` 和 `[v8] target_prior_bias top beams` 的等价信息

#### Scenario: support labels 为空时使用 uniform prior
- **WHEN** `v8_target_prior_head` 初始化 target prior 但 support labels 为空或 label budget 为 0
- **THEN** 系统 MUST 使用 uniform prior 作为 fallback
- **AND** metrics MUST 记录 fallback reason

#### Scenario: prior 初始化禁止使用 target test 或 physical oracle
- **WHEN** 系统初始化 v8 target prior
- **THEN** 系统 MUST NOT 读取 target_test label、target_test beam_power、target_test path fields、target-side CSI 或 radio/channel fields
- **AND** target_adapt 中未被选为 labeled support 的样本 label MUST NOT 用于 supervised prior 初始化

### Requirement: V8 target adaptation freeze policy
系统 MUST 提供 `v8_target_head_only` freeze policy，用于冻结 source backbone 和 source/shared prediction heads，只训练 v8 target branch、target prior 参数和显式启用的诊断头。

#### Scenario: 应用 V8 target head only 策略
- **WHEN** target adaptation 配置 `target_adaptation.freeze_policy=v8_target_head_only` 或等价 adaptation strategy
- **THEN** 系统 MUST 冻结 modality encoders、feature projections、fusion transformer、shared branch、source/shared beam head 和 physical beamspace head
- **AND** 系统 MUST 训练 `target_adapter`、`target_head`、`target_prior_bias`、可学习 `beta_prior`、启用的 `sector_head` 和启用的 `offset_head`
- **AND** metrics MUST 记录 trainable parameter names 或其 artifact path、trainable params、total params 和 trainable ratio

#### Scenario: 可选解冻最后一个 fusion block
- **WHEN** `hist_beam.v8.unfreeze_last_fusion_block=true`
- **THEN** 系统 MAY 只额外解冻最后一个 fusion block
- **AND** 配置或 optimizer metadata MUST 暴露该参数组使用的低学习率
- **AND** 默认配置 MUST 保持 `unfreeze_last_fusion_block=false`

### Requirement: V8 adaptation loss
系统 MUST 为 v8 target adaptation 提供独立 loss 组合。默认 supervised final loss MUST 支持 beam topology soft label，且 coarse-to-fine 诊断头启用时 MUST 计算 sector 和 offset loss。

#### Scenario: 计算 V8 soft final loss
- **WHEN** `v8_target_prior_head` 在 label budget 大于 0 的 target adaptation 中训练且 `hist_beam.v8.use_soft_beam_label=true`
- **THEN** 系统 MUST 基于 labeled support hard beam label 生成 soft beam distribution
- **AND** 系统 MUST 对 `logits_final` 或等价 final logits 计算 soft CE
- **AND** diagnostics MUST 记录 v8 final soft CE loss 和对应权重

#### Scenario: 计算 prior smoothness loss
- **WHEN** `v8_target_prior_head` 启用 `hist_beam.v8.loss_prior_smooth_weight > 0`
- **THEN** 系统 MUST 对相邻 beam 的 `target_prior_bias` 差分平方均值计算 smoothness loss
- **AND** 该 loss MUST 按配置权重参与 total loss

#### Scenario: 计算 coarse-to-fine 诊断 loss
- **WHEN** `hist_beam.v8.use_coarse_to_fine=true`
- **THEN** 模型 MUST 输出 `sector_logits` 和 `offset_logits`
- **AND** loss MUST 对 `beam_label // sector_size` 计算 sector CE
- **AND** loss MUST 对 `beam_label % sector_size` 计算 offset CE，并对最后一个不完整 sector 的非法 offset 做安全处理或在配置解析阶段拒绝不可整除设置

### Requirement: V8 诊断实验模式
系统 MUST 支持通过配置选择 v8 最小诊断实验模式，以区分 frozen representation 可分性、target prior 效果、source logits correction 效果和 coarse-to-fine 诊断效果。

#### Scenario: target linear probe 模式
- **WHEN** `hist_beam.v8.mode=target_linear_probe`
- **THEN** 系统 MUST 关闭 adapter 和 target prior
- **AND** final logits MUST 来自 target head 读取 frozen fused features 的输出

#### Scenario: target prior head 模式
- **WHEN** `hist_beam.v8.mode=target_prior_head`
- **THEN** 系统 MUST 启用 target adapter 和 target prior
- **AND** final logits MUST 默认为 `target_logits + beta_prior * target_prior_bias`

#### Scenario: source prior only 模式
- **WHEN** `hist_beam.v8.mode=source_prior_only`
- **THEN** 系统 MUST 允许 `lambda_src=1.0`、`lambda_tgt=0.0` 和 target prior correction
- **AND** run metadata MUST 标记该模式用于诊断 label prior correction 单独效果

#### Scenario: target prior coarse-to-fine 模式
- **WHEN** `hist_beam.v8.mode=target_prior_coarse_to_fine`
- **THEN** 系统 MUST 启用 target adapter、target prior 和 coarse-to-fine heads
- **AND** diagnostics MUST 包含 final beam、sector 和 offset 相关 loss 或不可用原因

### Requirement: V8 prototype classifier 诊断
系统 SHOULD 提供不参与训练的 v8 prototype classifier 诊断接口。若用户启用但实现或数据不足，系统 MUST 输出明确 unavailable reason，而不是静默忽略。

#### Scenario: 运行 evaluation-only prototype probe
- **WHEN** `hist_beam.v8.run_prototype_probe=true` 且 target support features 可用
- **THEN** 系统 MUST 基于 frozen backbone 提取 target support features 并按 beam 或 sector 构造 prototype
- **AND** target_test evaluation MUST 输出 prototype Top-1、Top-3、Top-5、NRP 和 prediction histogram，或记录缺失 power metric 的原因

#### Scenario: prototype probe 不可用
- **WHEN** `hist_beam.v8.run_prototype_probe=true` 但 prototype probe 尚未实现或 support features 不足
- **THEN** metrics MUST 记录 `prototype_probe_available=false`
- **AND** metrics MUST 记录机器可读 unavailable reason

### Requirement: HiST-Beam prediction histogram artifact
HiST-Beam source-only target evaluation 和 adapted target evaluation MUST 输出 prediction histogram 诊断产物，用于判断 source prior collapse 和 target prior correction 是否发生。

#### Scenario: source-only target eval 写出 histogram
- **WHEN** source-only target_test evaluation 完成
- **THEN** run directory MUST 包含 `prediction_hist.json` 或等价 artifact
- **AND** artifact MUST 包含 `true_hist`、`pred_hist`、`true_top_beams`、`pred_top_beams`、`mean_abs_beam_error`、`within_1_acc`、`within_2_acc` 和 `within_3_acc`

#### Scenario: adapted target eval 写出 histogram
- **WHEN** adapted target_test evaluation 完成
- **THEN** run directory MUST 包含 adaptation 后的 `prediction_hist.json` 或等价 artifact
- **AND** LOSO summary MUST 能引用或汇总该 histogram artifact

#### Scenario: histogram 不参与训练选择
- **WHEN** 系统生成 prediction histogram
- **THEN** histogram MUST 只在 evaluation 完成后基于 target_test prediction 和 target_test label 生成
- **AND** histogram MUST NOT 用于 adaptation threshold selection、prior 初始化、early stopping 或 optimizer update

### Requirement: Source long-tail de-bias 配置入口
系统 MAY 为 source training 提供 long-tail 去偏 loss 配置入口，但该入口 MUST 默认关闭，并 MUST 不改变旧实验的 source training loss。

#### Scenario: 默认使用既有 source CE
- **WHEN** 用户未显式设置 `source_train.loss_type`
- **THEN** 系统 MUST 使用既有 source training loss 语义
- **AND** 旧 v0-v7 quick validation 指标 MUST 不因新增配置入口改变

#### Scenario: 显式选择去偏 loss
- **WHEN** 用户设置 `source_train.loss_type=balanced_softmax` 或 `source_train.loss_type=logit_adjusted`
- **THEN** 系统 MUST 在配置和 diagnostics 中记录所选 loss type、class prior 来源和 tau
- **AND** 若实现尚不可用，系统 MUST 清晰失败或记录 unsupported reason，不得静默回退并声称已启用
