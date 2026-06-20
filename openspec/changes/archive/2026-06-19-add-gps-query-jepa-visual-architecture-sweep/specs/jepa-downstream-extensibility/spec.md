# jepa-downstream-extensibility Specification Delta

## ADDED Requirements

### Requirement: Downstream visual token source variants
JEPA downstream image encoder MUST 能在 opt-in 配置下消费不同 visual token sources，包括 JEPA patch tokens、overlap/conv/local visual encoder tokens、CNN feature-map tokens 和多尺度 tokens。默认输出 MUST 继续保持现有 `[B,T,D]` image feature 契约，除非配置显式启用 token-aware fusion。

#### Scenario: 新 token source 通过 pooler 消费
- **WHEN** `jepa_context_image` 或等价 downstream image encoder 使用新的 visual token source
- **THEN** 系统 MUST 将 tokens `[B,T,N,D]` 和 token metadata 传给配置的 pooler
- **AND** pooler 默认输出 MUST 为现有 projector/core 可消费的 `[B,T,D]`

#### Scenario: CNN feature-map tokens 保留来源 metadata
- **WHEN** downstream 使用 CNN layer feature map tokens
- **THEN** metadata MUST 记录 backbone type、selected stages、feature grid、token count、pretrained/freeze policy 和 projection dimension
- **AND** 系统 MUST 区分该候选是 JEPA reuse、JEPA-style retrain 还是 supervised-only anchor

### Requirement: K-token downstream fusion opt-in
JEPA downstream MUST 支持显式 opt-in 的 K-token output mode，用于保留 GPS-query、content-query 或多尺度 query tokens 给 token-aware representation core。未启用时，mean/GPS-query/hybrid/Predictive GPS-query++ pooler MUST 继续输出 `[B,T,D]`。

#### Scenario: 默认 pooler 输出不变
- **WHEN** 配置未声明 K-token output mode
- **THEN** JEPA downstream pooler MUST 输出 `[B,T,D]`
- **AND** 现有 beam head、loss、metrics 和 ModelOutput adaptation MUST 无需新增分支

#### Scenario: 启用 K-token output mode
- **WHEN** 配置声明 pooler 输出 `[B,T,K,D]` 或等价 token-aware output
- **THEN** 配置 MUST 同时声明能消费该输出的 representation core 或 adapter
- **AND** runtime metadata MUST 记录 `output_mode`、`k_tokens`、token source 和 core type

#### Scenario: 不兼容 core 被拒绝
- **WHEN** pooler 输出 K-token representation 但 representation core 只接受 `[B,K_modality,T,D]` 或 `[B,T,D]` 帧级输入
- **THEN** 系统 MUST 在配置加载或模型构建时抛出清晰错误
- **AND** 错误信息 MUST 指出 pooler output mode 与 core input contract 不兼容

### Requirement: Visual token diagnostics for downstream sweep
JEPA downstream architecture variants MUST 写出统一 visual token diagnostics。diagnostics MUST 能区分 token count、attention map shape、branch/gate weights、pooler output mode、checkpoint policy 和 condition feature source。

#### Scenario: GPS-query attention diagnostics 记录 token grid
- **WHEN** GPS-query 类 pooler 启用 attention diagnostics
- **THEN** diagnostics MUST 记录 attention map shape、token grid、token count、query count、attention entropy 或 peakiness summary
- **AND** attention map MUST detach 后用于日志或诊断，训练主损失 MUST 不依赖诊断张量

#### Scenario: predictive pooler 记录 branch 来源
- **WHEN** Predictive GPS-query++ 或 hybrid residual query pooler 完成 forward
- **THEN** diagnostics MUST 记录 content branch、GPS residual branch、temporal branch 或 equivalent branch availability
- **AND** diagnostics MUST 不直接消费 target label、beam power oracle 或 benchmark condition id 作为模型输入
