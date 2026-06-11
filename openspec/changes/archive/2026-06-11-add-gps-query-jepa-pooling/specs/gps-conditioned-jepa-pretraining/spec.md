## MODIFIED Requirements

### Requirement: JEPA context encoder 下游复用
系统 MUST 提供 supervised fusion 可配置使用的 JEPA context image encoder 初始化入口。该入口 MUST 从 JEPA checkpoint 中抽取 `context_encoder` 权重，MUST 同时兼容仅包含 state dict 的 objective checkpoint 和包含 `state_dict` 的恢复 checkpoint。默认模式 MUST 输出现有 fusion representation core 可消费的帧级 image 特征 `[B,T,D]`，并保持 patch-token mean pooling 兼容；显式配置 `pooling: gps_query_attention` 时，MUST 用 GPS 条件特征从 JEPA patch tokens 中 attention-pool 出 `[B,T,D]` image 特征，且 MUST 不要求重训 JEPA Stage 1、target encoder EMA 或 latent prediction loss。

#### Scenario: 从 JEPA best checkpoint 初始化 supervised image encoder
- **WHEN** supervised fusion 配置将 image encoder 设置为 `jepa_context_image` 且 checkpoint 指向 JEPA `best.pth`
- **THEN** 系统 MUST 加载 checkpoint 中的 `context_encoder.*` 权重
- **AND** image encoder forward MUST 将 `[B, T, 3, H, W]` 输入转换为 `[B, T, D]` 特征
- **AND** 未显式设置 pooling 时 MUST 使用 mean pooling 以保持既有配置兼容

#### Scenario: 从 JEPA last checkpoint 初始化 supervised image encoder
- **WHEN** supervised fusion 配置将 image encoder 设置为 `jepa_context_image` 且 checkpoint 指向 JEPA `last.pth`
- **THEN** 系统 MUST 从 checkpoint payload 的 `state_dict` 字段抽取 `context_encoder.*` 权重
- **AND** 输出特征维度 MUST 与 fusion 配置的 image encoder `output_dim` 一致

#### Scenario: GPS-query pooling 下游复用 JEPA context encoder
- **WHEN** supervised fusion 配置将 image encoder 设置为 `jepa_context_image`、设置 `pooling: gps_query_attention`，且提供同 batch/time 的 GPS 条件特征
- **THEN** 系统 MUST 继续只加载 checkpoint 中的 `context_encoder.*` 权重作为 JEPA 视觉 backbone
- **AND** 系统 MUST 使用 GPS 条件特征生成 query，对 context encoder patch tokens 执行 attention pooling
- **AND** image encoder 输出 MUST 保持 `[B,T,D]`
- **AND** 系统 MUST NOT 要求 JEPA target encoder、EMA 更新、JEPA latent loss、distiller 或外部 frozen teacher

#### Scenario: GPS-query pooling 缺少条件特征
- **WHEN** supervised fusion 配置启用 `pooling: gps_query_attention`，但 image encoder forward 未收到 GPS 条件特征
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出 `jepa_context_image` 的 GPS-query pooling 需要 GPS condition feature

#### Scenario: fair 下游配置复用多场景 JEPA checkpoint
- **WHEN** supervised BeamBench-fair JEPA 复用配置被加载
- **THEN** random-mask JEPA 配置 MUST 指向 `outputs/deepsense6g_gps_conditioned_jepa_full_s32_s34_lowmem/checkpoints/best.pth` 或 `last.pth`
- **AND** GPS-biased JEPA 配置 MUST 指向 `outputs/deepsense6g_gps_conditioned_jepa_gps_biased_s32_s34_lowmem/checkpoints/best.pth`
- **AND** 配置 MUST NOT 默认引用 scene31-only JEPA checkpoint
- **AND** GPS-query pooling 派生配置 MUST 继承 GPS-biased 多场景 checkpoint 口径
