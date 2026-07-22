## ADDED Requirements

### Requirement: U-MaskBeamJEPA 必须提供默认关闭的只读 intermediate return
U-MaskBeamJEPA MUST 在显式 `return_intermediates=true` 时从原 forward 返回 block、prototype-normalized、prototype logits、modality、router 和 fused 中间状态。该选项 MUST 只增加 detached-by-caller 的诊断 payload，不得改变 logits、mask、参数、state dict 或默认训练/推理计算。

#### Scenario: 默认 forward
- **WHEN** 调用方未声明 `return_intermediates` 或传入 false
- **THEN** forward 输出、融合 logits 和 state dict MUST 与变更前保持兼容
- **AND** 不得额外计算仅诊断需要的 block prototype logits

#### Scenario: 诊断 forward
- **WHEN** 调用方显式传入 `return_intermediates=true`
- **THEN** 输出 MUST 包含带明确 `[B,T,M,*]` 或 `[B,M,*]` shape 的真实中间 tensor
- **AND** 不存在的 encoder 内部 pre-projection tensor MUST 由抽取器 hook 获取或标记 unavailable，而不是复制 model forward
