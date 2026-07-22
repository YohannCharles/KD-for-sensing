## MODIFIED Requirements

### Requirement: U-MaskBeamJEPA 仅支持 T2/S1、BCACL U2 与 CMSBL

U-MaskBeamJEPA current contract MUST 只保留 T2/S1 所需的 supervised router、BPA/prototype、embedded full-modal teacher、same-model temporal superset consistency、active BPA/CMA ablation，以及 BCACL U2 private/shared supervision 和 CMSBL 训练状态。系统 MUST 删除 PCER、PGCD、候选动态 Router、cached reroute、BCACL relation teacher/quality/two-stage 和 compatibility fields。

#### Scenario: 构建 current 主线

- **WHEN** T2、S1、BPA/CMA、BCACL U2 或 CMSBL 构建模型与 loss
- **THEN** model forward、loss、metadata 和 15-pattern evaluation MUST 保持可用
- **AND** 默认 T2 推理 MUST 不实例化 BCACL/CMSBL 参数或读取本地产物

### Requirement: CMSBL per-sample loss 不改变 disabled path

loss owner MUST 只在 CMSBL hard-mask 启用时返回 per-sample fusion CE、fused prototype loss和按样本可用模态归一化的 modality prototype loss。CMSBL 关闭时 MUST 保持现有标量归约顺序与数值行为。

#### Scenario: 同一 batch 包含不同 mask

- **WHEN** CMSBL hard-mask 对 single、double 和 full 样本归约 loss
- **THEN** 每个样本的 restoration MUST 先按自身可用模态归一化再施加 mask 权重
- **AND** 不得仅因可用模态更多而获得更大 loss 尺度
