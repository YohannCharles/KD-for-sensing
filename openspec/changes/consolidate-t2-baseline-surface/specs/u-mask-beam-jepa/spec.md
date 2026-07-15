## MODIFIED Requirements

### Requirement: No-JEPA prototype and KD training options
U-MaskBeamJEPA MUST 支持在 `use_jepa_loss=false` 时启用 beam prototype alignment、embedded full-modal teacher CE 和 T2 same-model temporal superset consistency。系统 MUST 不提供 `use_full_to_partial_kd`、`kd_teacher_mode` 或外部 checkpoint teacher 支线；关闭 prototype 与 superset consistency 时，总损失 MUST 回退到当前配置声明的 beam CE 和 embedded teacher CE。

#### Scenario: no-JEPA T2 payload
- **WHEN** `use_jepa_loss=false` 且启用 prototype 或 superset consistency
- **THEN** training extension MUST 暴露 student fused feature、可用 modality features、student logits 与同模型 full-modal logits
- **AND** 不得要求 Gaussian JEPA NLL 字段或外部 teacher artifact

#### Scenario: 关闭 T2 增强
- **WHEN** prototype 和 superset consistency 均关闭
- **THEN** 总损失 MUST 不生成 prototype、KD 或 superset consistency 标量
- **AND** 系统 MUST 不解析 full-to-partial KD 配置

### Requirement: S1-S4 temporal-router 不属于 current U-Mask contract
U-MaskBeamJEPA current contract MUST 只保留 T2/S1 所需的 `supervised_router`、BPA/prototype、embedded full-modal teacher 和 same-model temporal superset consistency，以及 active BPA/CMA ablation 明确消费的 head/fusion controls。系统 MUST 删除与这些方法无关的 protected C2 fusion/loss branches、S1-S4 temporal router、full-to-partial teacher stabilization 和其 compatibility fields。

#### Scenario: T2 active branches 保持可用
- **WHEN** T2、S1 或 BPA/CMA ablation 构建其声明的 head、BPA、CMA、router 和 superset settings
- **THEN** model forward、loss 和 metadata MUST 保持可用
- **AND** 任何保留分支 MUST 能追溯到四方法或 active T2 artifact
