## ADDED Requirements

### Requirement: U-MaskBeamJEPA 必须保留逐帧候选路由状态
U-MaskBeamJEPA MUST 在候选 Router 启用时保留 `[B,T,M,D]` latent、`[B,T,M,C]` prototype logits和`[B,T,M]` mask所需的路由状态，并 MUST 提供可对detached状态重新执行候选 Router 的统一接口。Current Router 默认路径 MUST 不承担额外逐帧head计算。

#### Scenario: 配对候选重新路由
- **WHEN** 训练扩展从control与joint view取得detached候选状态
- **THEN** 统一接口 MUST 重新执行启用的帧级和模态级 Router组件
- **AND** 冻结expert参数不得获得梯度

### Requirement: 候选 Router 必须保持共享输出契约
候选 MUST 继续输出 `router_gate_logits`、`router_gate_weights`、`supervised_router_gate_weights`、`reliability_fusion_weights`、`unimodal_logits`和`missing_mask`，并 MUST 在metadata中记录variant和启用组件。

#### Scenario: 评估候选 checkpoint
- **WHEN** 共享评估器读取候选forward输出
- **THEN** 缺失模态权重 MUST 为零且可用权重和为一
- **AND** 评估器 MUST 能由unimodal logits与最终模态权重重构融合logits

