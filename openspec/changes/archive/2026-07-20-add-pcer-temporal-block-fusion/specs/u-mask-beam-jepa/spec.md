## ADDED Requirements

### Requirement: U-Mask PCER opt-in forward 保持默认兼容
U-MaskBeamJEPA MUST 以内嵌 opt-in component 支持 block prototype evidence static fusion 和 counterfactual Router fusion。配置未声明 PCER 时，模型 MUST 不实例化 PCER 参数并保持 current forward、state dict 和训练 metadata 行为兼容。

#### Scenario: 默认 T2 路径
- **WHEN** canonical T2 recipe 不声明 `model.primary.pcer`
- **THEN** forward MUST 继续使用现有 masked temporal pooling 和 current Router
- **AND** 输出 logits MUST 与变更前在允许数值误差内一致

#### Scenario: PCER block mask
- **WHEN** PCER 收到 `modality_temporal_mask[B,T,M]`
- **THEN** 输出的缺失 block weight MUST 严格为零且每个样本可用 block weight 和 MUST 为一
- **AND** fused logits/features MUST 不消费缺失 block

### Requirement: 新旧 Router 配置互斥
counterfactual PCER Router 与 current confidence/prototype-center Router MUST 不同时影响 fused prediction 或 Router loss。A1 MUST 保留 current Router 原始逻辑，仅增加正确的 availability mask；A3 MUST 关闭 current Router 监督。

#### Scenario: A1 与 A3 构建
- **WHEN** launcher 构建 A1 和 A3
- **THEN** A1 MUST 只有 current Router 参数参与融合与 Router loss
- **AND** A3 MUST 只有 PCER block Router 参数参与融合与 Router loss
