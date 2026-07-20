# u-mask-beam-jepa Specification

## Purpose

定义 T2/S1 所需的 U-MaskBeamJEPA 最小模型和 loss 契约，限制 retained branch 为 supervised router、BPA、CMA 和 same-model consistency 所需实现。
## Requirements
### Requirement: U-MaskBeamJEPA 仅支持 T2/S1 active branches

U-MaskBeamJEPA current contract MUST 只保留 T2/S1 所需的 supervised router、BPA/prototype、embedded full-modal teacher、same-model temporal superset consistency、active BPA/CMA ablation，以及 claim-ineligible PCER direction search 消费的 opt-in block/hierarchical/mask-residual router 与 evidence-learning controls。系统 MUST 删除不服务这些方法的 model、loss、router 与 compatibility fields。

#### Scenario: T2 active branches 保持可用

- **WHEN** T2、S1、BPA/CMA ablation 或 active PCER direction search 构建其声明的 head、BPA、CMA、router、target 和 superset settings
- **THEN** model forward、loss 和 metadata MUST 保持可用
- **AND** 任何保留分支 MUST 能追溯到四方法、active T2 artifact 或 PCER direction-search change

### Requirement: 四模态输入与 mask 语义稳定

模型 MUST 消费 image、radar、gps、lidar 的统一顺序、availability mask 与 temporal metadata，并输出 beam logits及 retained training payload。Opt-in flat/hierarchical/mask-residual router MUST 对缺失块输出零权重；hierarchical alpha/beta MUST 分别在可用模态和模态内可用时间上归一化。

#### Scenario: 缺失模态训练

- **WHEN** batch 标记一个或多个模态不可用
- **THEN** model MUST 使用当前 mask 运行且保持至少一个可用模态
- **AND** 不得构建独立 teacher model 或读取 checkpoint teacher

#### Scenario: 层级与 residual 路由

- **WHEN** direction search 启用 hierarchical 或 mask-residual router
- **THEN** 最终 block weight 和 MUST 为一且不可用 block MUST 严格为零
- **AND** residual MUST 在可用位置近似零均值且 residual scale MUST 可训练

### Requirement: U-Mask active branch 必须匹配训练与参数语义
U-Mask MUST 将 inactive classifier/prototype/router branch 冻结并从 optimizer trainable parameter 集合排除；metadata MUST 声明 active head 与 trainable parameter count。`reliability_mean` 在 router oracle weight 为零时 MUST 不构造 oracle-loss gradient graph，但仍可输出明确的 disabled diagnostics。

#### Scenario: T2-CLS 构建
- **WHEN** T2-CLS 使用 classifier head 构建模型
- **THEN** prototype branch MUST 不参与 optimizer 或 trainable parameter count
- **AND** metadata MUST 记录 classifier 为 active head

#### Scenario: reliability mean 训练
- **WHEN** `fusion_type=reliability_mean` 且 router oracle weight 为零
- **THEN** training MUST 不计算 router oracle loss
- **AND** diagnostics MUST 标记该项为 disabled

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

### Requirement: Joint fused-logit 决策对齐
UMaskBeamJEPA 的候选动态 Router 配对训练 SHALL 将声明的互斥决策目标应用于 Joint corrupted view 的最终 fused logits，同时保持 control 与 Joint view availability 完全一致，且不得读取 corruption 类型、严重度或状态矩阵作为模型或 loss 特征。

#### Scenario: 配对视图应用决策目标
- **WHEN** dynamic Router paired Joint 训练启用且已构造相同 availability 的 control/joint 输出
- **THEN** loss 使用 Joint fused logits、beam label 及目标所需的可选 power 计算声明的决策监督

#### Scenario: Power 仅进入 loss
- **WHEN** 所选目标需要 future beam power
- **THEN** power tensor 只传入 loss，Router forward 输入与输出 schema 保持不变

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
