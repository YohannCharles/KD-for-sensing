## MODIFIED Requirements

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
