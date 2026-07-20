## ADDED Requirements

### Requirement: H2R evidence profile 显式且参数公平
UMaskBeamJEPA 的动态 H2R Router SHALL 支持 `full`、`generic_confidence` 和 `prototype_topology` 三种 evidence profile，并通过固定宽度证据屏蔽保持 Router 参数量一致。未声明 profile 时 MUST 使用 `full` 并保持现有数值行为。

#### Scenario: 普通置信度与原型拓扑对照
- **WHEN** 两个 H2R 候选除 evidence profile 外配置相同
- **THEN** 两者的 frame-health 和 modality-residual MLP 输入维度与参数量相同
- **AND** `generic_confidence` 不消费 circular topology coordinates/dispersion
- **AND** `prototype_topology` 不消费 learned reliability、logit norm 或 latent cosine disagreement

#### Scenario: 非法 profile 失败
- **WHEN** 配置声明未知 evidence profile
- **THEN** 模型构建在训练前 fail closed 并列出允许值

### Requirement: JointCE 可作为 H2R 帧门控直接监督
当动态 Router 的 fused decision objective 具有正权重时，系统 SHALL 允许 H2R 的 frame-rank 辅助权重为零；非分层 Router 仍 MUST 拒绝非零 frame-rank 权重。

#### Scenario: JointCE-only H2R 配置
- **WHEN** H2R 声明 `joint_hard_ce`、正 fused decision weight 和零 frame-rank weight
- **THEN** 配置解析成功且 Joint fused logits 的梯度可到达 frame-health 参数

