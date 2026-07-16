# u-mask-beam-jepa Specification

## Purpose

定义 T2/S1 所需的 U-MaskBeamJEPA 最小模型和 loss 契约，限制 retained branch 为 supervised router、BPA、CMA 和 same-model consistency 所需实现。

## Requirements

### Requirement: U-MaskBeamJEPA 仅支持 T2/S1 active branches

U-MaskBeamJEPA current contract MUST 只保留 T2/S1 所需的 supervised router、BPA/prototype、embedded full-modal teacher、same-model temporal superset consistency，以及 active BPA/CMA ablation 消费的 head/fusion controls。系统 MUST 删除不服务这些方法的 model、loss、router 与 compatibility fields。

#### Scenario: T2 active branches 保持可用

- **WHEN** T2、S1 或 BPA/CMA ablation 构建其声明的 head、BPA、CMA、router 和 superset settings
- **THEN** model forward、loss 和 metadata MUST 保持可用
- **AND** 任何保留分支 MUST 能追溯到四方法或 active T2 artifact

### Requirement: 四模态输入与 mask 语义稳定

模型 MUST 消费 image、radar、gps、lidar 的统一顺序、availability mask 与 temporal metadata，并输出 beam logits 及 retained training payload。

#### Scenario: 缺失模态训练

- **WHEN** batch 标记一个或多个模态不可用
- **THEN** model MUST 使用当前 mask 运行且保持至少一个可用模态
- **AND** 不得构建独立 teacher model 或读取 checkpoint teacher
