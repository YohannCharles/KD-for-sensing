## MODIFIED Requirements

### Requirement: U-MaskBeamJEPA 仅支持 T2/S1 active branches

U-MaskBeamJEPA current contract MUST 只保留 T2/S1 所需的 supervised router、BPA/prototype、embedded full-modal teacher、same-model temporal superset consistency，以及 active BPA/CMA ablation 消费的 head/fusion controls。系统 MUST 删除不服务这些方法的 model、loss、router 与 compatibility fields。

#### Scenario: T2 active branches 保持可用

- **WHEN** T2、S1 或 BPA/CMA ablation 构建其声明的 head、BPA、CMA、router 和 superset settings
- **THEN** model forward、loss 和 metadata MUST 保持可用
- **AND** 任何保留分支 MUST 能追溯到四方法或 active T2 artifact
