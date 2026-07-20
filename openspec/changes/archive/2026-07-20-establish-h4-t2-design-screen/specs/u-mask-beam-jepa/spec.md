## MODIFIED Requirements

### Requirement: U-MaskBeamJEPA 仅支持 T2/S1 active branches
U-MaskBeamJEPA current contract MUST 只保留 T2/S1 所需的 supervised router、BPA/prototype、embedded full-modal teacher、same-model temporal superset consistency，以及 active BPA/CMA ablation 和 H4 design-screening 消费的 head/fusion/temporal controls。默认 T2/S1 MUST 保持 supervised router 与 masked-mean temporal pooling；受限 screening 分支仅可启用显式的 reliability-mean fusion、mask-aware temporal attention、registry-backed scratch encoder 和 GPS MLP capacity/noise controls。系统 MUST 删除不服务这些方法的 model、loss、router 与 compatibility fields。

#### Scenario: 默认 T2 active branches 保持可用
- **WHEN** T2、S1 或 BPA/CMA ablation 构建其声明的 head、BPA、CMA、router 和 superset settings
- **THEN** model forward、loss 和 metadata MUST 保持可用
- **AND** 默认 fusion/pooling MUST 分别为 supervised router 与 masked mean

#### Scenario: 受控结构筛选保持 mask 语义
- **WHEN** T2 design-screening 启用 reliability-mean fusion 或 mask-aware temporal attention
- **THEN** model MUST 消费四模态 availability 和 temporal masks，且每个样本至少保留一个有效 cell
- **AND** 无效模态或时间 cell MUST 不影响归一化融合/汇聚权重

## ADDED Requirements

### Requirement: 结构候选 metadata 可审计
U-MaskBeamJEPA MUST 在 training metadata 中记录 effective fusion type、temporal pooling type、其参数数量、encoder configs 与 GPS training-time noise setting，使 checkpoint/evaluation provenance 能区分 H4 control 和设计候选。

#### Scenario: 读取 design candidate metadata
- **WHEN** 训练或评估一个非默认 T2 design candidate
- **THEN** metadata MUST 包含 effective fusion、pooling、encoder 与 GPS 配置
- **AND** 缺失这些字段的 candidate MUST 不进入汇总
