## MODIFIED Requirements

### Requirement: K-token downstream fusion opt-in
JEPA downstream MUST 支持显式 opt-in 的 K-token output mode，用于保留 GPS-query、content-query 或多尺度 query tokens 给 token-aware representation core 或显式 token readout。未启用时，mean/GPS-query/hybrid/Predictive GPS-query++ pooler MUST 继续输出 `[B,T,D]`。启用 K-token output mode 时，系统 MUST 记录 token source、readout type、是否 trainable、`k_tokens` 和 core/input compatibility metadata。

#### Scenario: 默认 pooler 输出不变
- **WHEN** 配置未声明 K-token output mode
- **THEN** JEPA downstream pooler MUST 输出 `[B,T,D]`
- **AND** 现有 beam head、loss、metrics 和 ModelOutput adaptation MUST 无需新增分支

#### Scenario: 启用 K-token output mode
- **WHEN** 配置声明 pooler 输出 `[B,T,K,D]` 或等价 token-aware output
- **THEN** 配置 MUST 同时声明能消费该输出的 representation core、adapter 或 token readout
- **AND** runtime metadata MUST 记录 `output_mode`、`k_tokens`、token source、core type 和 token readout type

#### Scenario: 不兼容 core 被拒绝
- **WHEN** pooler 输出 K-token representation 但 representation core、adapter 或 readout 不能消费该 token shape
- **THEN** 系统 MUST 在配置加载或模型构建时抛出清晰错误
- **AND** 错误信息 MUST 指出 pooler output mode、实际 output shape 和 core/readout input contract 不兼容

#### Scenario: legacy token-aware transformer 可审计
- **WHEN** 配置使用现有 `token_aware_transformer` 消费 K-token features 且未声明显式 readout
- **THEN** metadata MUST 将 readout 标记为 `legacy_uniform_mean` 或等价值
- **AND** metadata MUST 记录该路径最终会对 token/channel 维做均值聚合
- **AND** 旧 checkpoint 和旧 final config MUST 不被误标记为 learned readout

#### Scenario: learned token readout 显式 opt-in
- **WHEN** 配置声明 learned、weighted 或 attention-based token readout
- **THEN** 系统 MUST 构建对应 readout 并输出现有 beam head 可消费的 `[B,T,D]` feature
- **AND** readout MUST 记录 trainable parameter count、readout weight summary 或等价 diagnostics
- **AND** 默认 mean、GPS-query frame、hybrid residual query 和 Predictive GPS-query++ 配置 MUST 不改变语义

#### Scenario: token readout 不读取 oracle 信息
- **WHEN** token readout 或 token-aware core forward
- **THEN** readout MUST NOT 读取 target beam、beam power oracle、sample label、P0-P5 condition id 或 evaluation metric
- **AND** condition metadata MAY 只用于 diagnostics、masking 可用性或离线分组统计
