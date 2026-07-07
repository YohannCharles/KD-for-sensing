## ADDED Requirements

### Requirement: Final ablation fusion and router controls
PCPG/router 本地实验能力 MUST 支持 final c2 消融需要的 `average`、`weighted_sum`、`raw_conf_gate`、`bprr`、`pcpg` 和 `supervised_router` fusion 对照，以及 router pattern、reliability、prototype margin、entropy、confidence 和 logit norm 特征开关。所有新增行为 MUST 显式 opt-in。

#### Scenario: fusion baseline 可 forward
- **WHEN** 配置选择 `weighted_sum`、`average`、`raw_conf_gate`、`bprr`、`pcpg` 或 `supervised_router`
- **THEN** 模型 forward MUST 返回 beam logits
- **AND** missing mask MUST 被正确应用到不可用模态
- **AND** 输出 MUST 无 NaN

#### Scenario: supervised router supervision 可关闭
- **WHEN** 配置 `router_supervision=none`
- **THEN** supervised router MUST 仍可训练和 forward
- **AND** oracle CE/distillation loss MUST 不加入训练 loss

### Requirement: Final prototype/head controls
PCPG/router 本地实验能力 MUST 支持 final c2 消融需要的 prototype alignment、modality prototype loss、circular/gaussian soft target 和 classifier head controls。关闭某个 loss 或 head feature 时，训练流程 MUST 继续可运行并在 diagnostics 或 config 中可审计。

#### Scenario: prototype alignment loss 可关闭
- **WHEN** 配置 `use_beam_prototype_alignment=false` 或 `beam_proto_align_weight=0.0`
- **THEN** 训练 loss MUST 不包含 prototype alignment 额外项
- **AND** 其它训练流程 MUST 保持可运行

#### Scenario: circular soft target 可关闭
- **WHEN** 配置 `use_circular_soft_targets=false` 且 `use_gaussian_beam_targets=false`
- **THEN** prototype target MUST fallback 到普通 hard CE/onehot 语义
- **AND** 配置和 summary MUST 记录该 ablation 状态
