## ADDED Requirements

### Requirement: HiST-Beam radio-semantic prototype variant
HiST-Beam MUST 在现有 flat、hierarchical、shared-private、adapter-only、adapter+coarse-prototype 和 full fine-tuning baseline 之外，支持 radio-semantic prototype variant。该 variant MUST 显式使用 radio-semantic label、shared radio prototype 和可选 radio-conditioned beam inference，并 MUST 与现有 `v5_adapter_proto` coarse/private prototype baseline 可区分。

#### Scenario: 构建 radio prototype variant
- **WHEN** 用户配置 HiST-Beam variant 为 `v6_radio_proto`、`adapter_radio_proto` 或等价 radio-semantic prototype 模式
- **THEN** 系统 MUST 构建 shared/private/adapted private 表征、radio head、beam head 和 radio prototype diagnostics
- **AND** variant metadata MUST 记录 `proto_type=radio_semantic`

#### Scenario: existing full fine-tuning baseline 不被重解释
- **WHEN** 用户配置现有 `v6_full_finetune` 或 `full_finetune`
- **THEN** 系统 MUST 继续按 full fine-tuning baseline 更新参数
- **AND** summary MUST 不把该 run 标记为 radio-semantic prototype method

### Requirement: HiST-Beam radio branch diagnostics
启用 radio-semantic HiST-Beam 时，模型输出、loss diagnostics 和 evaluation artifact MUST 包含足以证明 radio branch 生效的字段。普通非 radio 配置 MUST 不要求这些字段。

#### Scenario: radio branch 输出被记录
- **WHEN** radio-semantic 配置启用且 forward 完成
- **THEN** 模型 diagnostics MUST 包含 `radio_logits` 或等价 radio prediction 输出
- **AND** diagnostics MUST 包含 `num_radio_classes`、radio label mode 和 radio condition 是否启用

#### Scenario: radio loss no-op 可诊断
- **WHEN** 配置启用 `lambda_radio` 但 batch 没有合法 radio labels
- **THEN** loss diagnostics MUST 将 radio loss 标记为 unavailable 或 coverage 0
- **AND** 系统 MUST 不用 0 coverage 的 radio loss 证明 radio branch 已生效

### Requirement: HiST-Beam radio-conditioned beam head
HiST-Beam MUST 支持 radio-conditioned beam head 作为 opt-in 行为。启用时，beam head 输入 MUST 包含 shared representation、adapted private representation 和 radio assignment embedding；关闭时，系统 MUST 保持现有 shared/private beam head 行为。

#### Scenario: source 阶段使用 predicted radio assignment
- **WHEN** source training 启用 `use_radio_condition_in_beam_head`
- **THEN** 系统 MUST 从 `radio_logits` 的 soft assignment 计算 radio embedding
- **AND** beam logits MUST 来自包含该 embedding 的 beam head 输入

#### Scenario: target 阶段优先使用 source radio prototype assignment
- **WHEN** target adaptation 或 target_test evaluation 启用 radio condition 且 source radio prototypes 可用
- **THEN** 系统 MUST 使用 shared representation 到 `mu_radio_c` 的 assignment 计算 radio embedding
- **AND** 若 prototype artifact 不可用，系统 MUST 记录 fallback 或 unavailable reason
