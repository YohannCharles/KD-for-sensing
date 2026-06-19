## ADDED Requirements

### Requirement: Predictive GPS-query++ downstream pooler
JEPA downstream MUST support an opt-in Predictive GPS-query++ path that combines current content latent, GPS-conditioned residual latent, and causal temporal predicted latent. This path MUST preserve existing mean-pooling and `gps_query_attention` behavior unless explicitly selected by configuration.

#### Scenario: 构建 Predictive GPS-query++ pooler
- **WHEN** `jepa_context_image` 配置声明 `pooler.type: predictive_gps_query` 或等价 opt-in 类型
- **THEN** 系统 MUST 构建 content-query anchor、GPS-query residual path、temporal latent predictor 和 reliability-aware gate
- **AND** pooler 输出 MUST 保持 `[B,T,D]`，可被现有 projector、representation core 和 beam head 消费
- **AND** 现有 `pooling: mean`、`pooling: gps_query_attention` 和 `pooler.type: hybrid_residual_query` 配置 MUST 不改变语义

#### Scenario: GPS path 作为 residual 条件
- **WHEN** Predictive GPS-query++ 同时计算 content latent 和 GPS-query latent
- **THEN** GPS-query latent MUST 作为相对 content 或 mean anchor 的 residual/bias 参与输出
- **AND** 配置 MUST 提供 residual scale、initialization 或 gating 机制，避免 GPS path 在训练初期完全覆盖 content anchor

#### Scenario: 输出 GPS-query++ diagnostics
- **WHEN** Predictive GPS-query++ forward 完成
- **THEN** runtime diagnostics MUST 包含 content branch、GPS residual branch、temporal predicted branch 的 availability、gate weights 或 equivalent scores
- **AND** diagnostics MUST 记录 residual scale、GPS-query attention summary、temporal source history range 和 fallback/warning 状态

### Requirement: Causal temporal latent predictor
JEPA downstream predictive path MUST support a causal temporal latent predictor that predicts current or future image latent from prior image latents only. The predictor MUST be opt-in and MUST NOT read future frames, target labels, beam powers, or sample order beyond the current batch/time sequence.

#### Scenario: 使用历史 latent 预测当前 latent
- **WHEN** Predictive GPS-query++ 启用 temporal predictor 且输入序列提供当前步之前的历史 image latent
- **THEN** predictor MUST produce `temporal_predicted_latent` aligned with the current prediction step
- **AND** metadata MUST record history window、source history range、predictor type、availability mask 和 insufficient-history fallback

#### Scenario: 拒绝 future leak
- **WHEN** temporal predictor 生成 step `t` 的 predicted latent
- **THEN** predictor MUST only consume image latent from steps `< t`
- **AND** tests MUST cover that source history range never includes `t` or future steps

#### Scenario: 历史不足可审计降级
- **WHEN** 当前样本没有足够历史 latent
- **THEN** predictor MUST use configured `raw`、`skip`、`zero` 或 `clamp` fallback
- **AND** diagnostics MUST record affected count and fallback strategy

### Requirement: Predictive JEPA auxiliary latent objectives
Predictive GPS-query++ training MAY enable auxiliary latent objectives that encourage temporal predicted latent and corrupt-view latent to align with clean target latent. These objectives MUST be opt-in and MUST NOT change default beam-only training unless configured.

#### Scenario: 启用 latent prediction loss
- **WHEN** training config declares predictive latent auxiliary loss
- **THEN** training MUST compute a loss between predicted/corrupt latent and clean detached target latent or configured target representation
- **AND** loss logging MUST include objective name、weight、sample count 和 whether target latent is detached

#### Scenario: 默认训练保持兼容
- **WHEN** config does not declare predictive latent auxiliary losses
- **THEN** supervised beam loss、metrics、checkpoint workflow 和 model output adaptation MUST remain unchanged

### Requirement: GPS-query++ metadata and compatibility
Predictive GPS-query++ runs MUST be distinguishable from existing JEPA GPS-query baseline runs in final config and runtime metadata.

#### Scenario: 写出架构 metadata
- **WHEN** Predictive GPS-query++ model writes final config or runtime metadata
- **THEN** metadata MUST include pooler type、content query count、GPS query count、temporal predictor type、reliability gate type、residual scale and enabled auxiliary losses
- **AND** metadata MUST include source JEPA checkpoint path and whether context encoder is frozen

#### Scenario: 旧 GPS-query checkpoint 不被误加载为 GPS-query++
- **WHEN** loader receives a checkpoint whose metadata indicates `gps_query_attention`
- **THEN** system MUST NOT silently treat it as Predictive GPS-query++
- **AND** incompatible missing/unexpected keys MUST produce clear diagnostics unless user explicitly requests non-strict transfer
