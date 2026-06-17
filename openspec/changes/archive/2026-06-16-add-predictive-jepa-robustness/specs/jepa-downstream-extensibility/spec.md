## ADDED Requirements

### Requirement: Hybrid residual query pooler
JEPA downstream MUST 支持一个可注册 hybrid residual query pooler，用于结合 patch-token mean pooling、learned content query pooling 和 GPS residual query pooling。该 pooler MUST 默认输出 `[B,T,D]`，并 MUST 保持现有 mean 和 GPS-query pooler 行为兼容。

#### Scenario: Hybrid pooler 构建和输出
- **WHEN** `jepa_context_image` 配置 `pooler.type: hybrid_residual_query` 或等价类型
- **THEN** 系统 MUST 构建 mean/content/GPS residual query 路径
- **AND** forward MUST 接收 JEPA patch tokens `[B,T,N,D]` 与可选 GPS condition features `[B,T,C]`
- **AND** 输出 MUST 为现有 modular sequence projector 可消费的 `[B,T,D]`

#### Scenario: GPS 只作为 residual bias
- **WHEN** hybrid pooler 同时计算 mean/content query latent 和 GPS-query latent
- **THEN** GPS-query latent MUST 作为相对 mean/content anchor 的 residual 修正参与输出
- **AND** pooler MUST 提供配置项或初始化策略，避免 GPS-query path 在训练初期完全覆盖 mean/content anchor

### Requirement: Temporal predicted latent auxiliary branch
JEPA downstream image encoder MUST 在 opt-in 配置下暴露 current latent 与 temporal predicted latent auxiliary branch。该行为 MUST 不改变默认 forward 输出、checkpoint schema 或现有 mean/GPS-query baseline 语义。

#### Scenario: 暴露 current 和 predicted latent
- **WHEN** 配置启用 predictive auxiliary branch 且输入序列提供历史帧
- **THEN** encoder MUST 记录或返回 `current_latent`、`temporal_predicted_latent`、branch availability、source history range 和 fallback metadata
- **AND** temporal prediction MUST 只使用当前时间步之前的历史 latent，不得读取未来帧

#### Scenario: 历史不足可诊断降级
- **WHEN** 历史长度不足以生成 temporal predicted latent
- **THEN** encoder MUST 按配置使用 raw、skip、zero 或 clamp fallback
- **AND** metadata MUST 记录 insufficient history count 和 fallback strategy

### Requirement: Feature-consistency fusion diagnostics
JEPA downstream predictive fusion MUST 支持基于 latent 一致性的 gate 或 helper，用于融合 current image latent、temporal predicted latent 和 GPS residual 信息。该 gate MUST 不直接读取 benchmark condition id。

#### Scenario: Gate 输入不包含 condition id
- **WHEN** feature-consistency gate forward
- **THEN** gate MUST 只消费 latent tensors、valid masks、observability score、GPS delay/reliability 或等价连续特征
- **AND** gate MUST NOT 直接消费 `c_idx`、`d_idx`、`predictive_condition_id`、`gps_condition` 或 `image_condition`

#### Scenario: 写出 consistency diagnostics
- **WHEN** predictive JEPA model forward 完成
- **THEN** output 或 runtime metadata MUST 包含 current/predicted/GPS residual branch availability、gate weights 或 equivalent scores、latent consistency summary 和 warnings
- **AND** 普通 JEPA mean/GPS-query baseline 在未启用该功能时 MUST 不要求这些 diagnostics
