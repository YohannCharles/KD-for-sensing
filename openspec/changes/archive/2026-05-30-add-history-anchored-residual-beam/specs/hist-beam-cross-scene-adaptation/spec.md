## ADDED Requirements

### Requirement: History-anchored HiST-Beam 变体
HiST-Beam MUST 支持显式配置的 history-anchored 变体。该变体 MUST 在保留现有 sensing modality fusion、shared/private representation 和 adapter/prototype 框架的基础上，接收历史 beam anchor，并输出 residual/delta logits 与可重建的绝对 beam logits。

#### Scenario: 构建 history-anchored residual 变体
- **WHEN** 用户配置 `hist_beam.history_anchor.enabled=true` 且 `hist_beam.history_anchor.mode=residual_delta`
- **THEN** 系统 MUST 构建包含 beam-history embedding 或等价 conditioning 的 HiST-Beam 模型
- **AND** 模型 MUST 输出 residual logits、reconstructed beam logits、shared representation 和 private representation
- **AND** 输出 MUST 继续兼容现有 Top-K beam evaluation 流程

#### Scenario: 关闭 history anchor 保持旧变体语义
- **WHEN** 用户运行 `v0_flat`、`v3_decoupled`、`v6_radio_proto`、`v8_path_proto` 或 full fine-tuning baseline 且未显式启用 history anchor
- **THEN** 模型 forward MUST 不要求 `input_beam_batch`
- **AND** source-only evaluation MUST 与当前绝对 beam prediction 语义保持兼容

#### Scenario: history absolute classifier 作为消融
- **WHEN** 用户配置 `hist_beam.history_anchor.mode=absolute_with_history`
- **THEN** 模型 MAY 使用历史 beam embedding 预测绝对 beam logits
- **AND** summary MUST 将其标记为 history-input absolute classifier ablation，而不是 residual 主方法

### Requirement: Residual beam loss
启用 history-anchored residual 模式时，训练流程 MUST 使用 residual/delta label 计算主 beam loss，并 MAY 保留可配置的绝对 beam auxiliary loss。残差 loss MUST 支持多 horizon 输出，并 MUST 与现有 hierarchical、radio、path 和 orthogonality loss 组合。

#### Scenario: source training 计算 residual CE
- **WHEN** source training batch 包含合法 `input_beam` 和 future beam label
- **THEN** training loop MUST 计算 circular residual label
- **AND** training loop MUST 对 residual logits 计算 CE loss
- **AND** metrics MUST 记录 residual loss 和 reconstructed absolute Top-K

#### Scenario: 绝对 auxiliary loss 可配置
- **WHEN** 配置设置 `hist_beam.history_anchor.lambda_absolute_aux > 0`
- **THEN** training loop MAY 对 reconstructed absolute beam logits 计算 auxiliary CE
- **AND** 该 auxiliary loss MUST 使用真实 future beam label，而不是 residual label

#### Scenario: history anchor 缺失时失败而非静默降级
- **WHEN** history-anchored residual 模式的训练 batch 缺少合法历史 beam anchor
- **THEN** training MUST 失败并输出包含 `input_beam` 或 `last_beam` 的错误信息
- **AND** 系统 MUST NOT 静默改用绝对 beam CE 继续训练

### Requirement: Residual shared-private 解耦
history-anchored residual 模式下，HiST-Beam shared/private 解耦 MUST 将 shared branch 定义为相对传播 residual 表征，将 private branch 定义为场景私有校准表征。模型 MUST 在 metadata 中区分 residual shared prediction 与 private calibration。

#### Scenario: shared branch 预测 residual distribution
- **WHEN** 模型启用 shared/private 和 history-anchored residual 模式
- **THEN** shared branch MUST 产生用于 residual/delta prediction 的 representation 或 logits
- **AND** shared branch MAY 继续输出 path/radio/geometry auxiliary head
- **AND** shared branch MUST NOT 被解释为直接学习 source 场景绝对 beam prior 的主分支

#### Scenario: private branch 产生校准项
- **WHEN** 模型启用 private calibration
- **THEN** private branch 或 adapter MUST 能输出 logit bias、temperature、offset、prototype-conditioned correction 或等价场景私有校准项
- **AND** calibration metadata MUST 记录实际启用的校准类型和 trainable parameter count

#### Scenario: prototype 不直接替代 residual prediction
- **WHEN** radio 或 path prototype 与 history-anchored residual 模式同时启用
- **THEN** prototype MAY 作为 shared assignment、private calibration 或 auxiliary diagnostic
- **AND** prototype MUST NOT 绕过 residual head 直接输出最终 beam prediction，除非该 run 明确标记为非 residual 消融

### Requirement: History-anchored few-shot private calibration
target adaptation 在 history-anchored residual 模式下 MUST 支持低参数 private calibration。默认策略 MUST 冻结 source encoders、fusion backbone 和 shared residual branch，只训练配置允许的 private adapter、calibration head、logit bias、temperature、LayerNorm affine 或等价低参数模块。

#### Scenario: few-shot adaptation 冻结 shared residual backbone
- **WHEN** 用户运行 history-anchored residual target adaptation 且未显式选择 full fine-tuning
- **THEN** 系统 MUST 冻结 sensing encoders、fusion module 和 shared residual branch
- **AND** 系统 MUST 只训练配置允许的 private calibration 参数
- **AND** metrics MUST 记录 trainable parameter count、total parameter count 和 trainable ratio

#### Scenario: labeled target 使用 residual supervised loss
- **WHEN** `label_budget>0` 且 labeled target_adapt 样本存在合法 future beam label
- **THEN** adaptation MUST 基于 labeled target_adapt 样本计算 residual supervised loss
- **AND** unlabeled target_adapt 样本 MUST NOT 读取 future beam label 作为 supervised loss

#### Scenario: target sensitive supervision 保持可审计
- **WHEN** history-anchored residual adaptation 使用 target path、radio、beam_power 或 channel-derived supervision
- **THEN** run metadata MUST 记录对应 sensitive usage flag
- **AND** summary MUST 根据 profile 规则标记该 run 是否可用于主结论

### Requirement: History-anchored HiST-Beam 预测产物
history-anchored residual evaluation MUST 在现有 HiST-Beam predictions 和 metrics 基础上新增 residual 诊断字段。产物 MUST 同时保留 residual-space 信息和 reconstructed absolute beam 信息。

#### Scenario: predictions 保存 residual 字段
- **WHEN** source-only evaluation 或 adapted evaluation 完成 history-anchored residual run
- **THEN** predictions MUST 包含 sample id、scene、last_beam、true beam、true residual、predicted residual、top-k residual、predicted beam 和 top-k reconstructed beam
- **AND** predictions MUST 标明样本来自 `target_test`

#### Scenario: metrics 输出 residual 与绝对指标
- **WHEN** evaluation 完成 history-anchored residual run
- **THEN** metrics MUST 包含 residual accuracy 或 residual error diagnostic
- **AND** metrics MUST 包含 reconstructed absolute Top-1、Top-3、Top-5
- **AND** 若 beam_power 可用，metrics MUST 包含 reconstructed absolute prediction 的 normalized received power 和 beam power loss dB

#### Scenario: summary 可比较 residual 和 absolute baseline
- **WHEN** summary 汇总同一 source、target、budget 和 seed 下的 absolute baseline 与 residual run
- **THEN** summary MUST 输出 residual run 相对 absolute source-only 的 delta
- **AND** summary MUST 输出 residual run 相对 last-beam 和 Markov delta baseline 的 delta 或不可比原因
