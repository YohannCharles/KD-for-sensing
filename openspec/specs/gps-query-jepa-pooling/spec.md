# gps-query-jepa-pooling Specification

## Purpose
定义 GPS-query attention pooling 在 JEPA 下游 image encoder 中的边界、输入输出契约、诊断信息和派生配置要求，使 Image+GPS beam prediction 能在不恢复 JEPA 预训练分支或旧 KD 路线的前提下复用 GPS 条件视觉 token 聚合。
## Requirements
### Requirement: GPS-query Attention Pooling 模块
系统 MUST 提供 `GPSQueryPool` 模块，用 GPS/motion 条件特征生成 query，并从 JEPA patch tokens 中聚合 beam prediction 相关的视觉局部信息。该模块 MUST 接收 patch tokens `[B,T,N,D]` 与条件特征 `[B,T,C]`，MUST 支持可配置 `k_queries`、`num_heads`、`condition_dim`、`latent_dim` 和 dropout，默认 MUST 输出 `[B,T,D]`。

#### Scenario: K-query 池化输出帧级特征
- **WHEN** `GPSQueryPool` 接收 patch tokens `[B,T,N,D]` 和 GPS 条件特征 `[B,T,C]`
- **THEN** 系统 MUST 生成 `K` 个 query token 并对每帧 `N` 个 patch tokens 执行 cross-attention
- **AND** 默认输出 MUST 为 `[B,T,D]`
- **AND** 输出 MUST 经 `LayerNorm` 或等价归一化稳定训练

#### Scenario: attention map 诊断
- **WHEN** `GPSQueryPool` 启用 attention diagnostics
- **THEN** forward MUST 返回或记录与 batch/time/query/patch 对齐的 attention map
- **AND** attention map 的平均 head 形状 MUST 为 `[B,T,K,N]` 或在 metadata 中明确等价形状
- **AND** attention map MUST detach 后用于日志或诊断，训练主损失 MUST 不依赖诊断张量

#### Scenario: 条件维度校验
- **WHEN** GPS 条件特征的 batch/time 维与 patch tokens 不一致
- **THEN** 系统 MUST 抛出包含 patch token shape 与条件 feature shape 的清晰错误
- **AND** 系统 MUST 不静默广播错误的 GPS 条件

### Requirement: JEPA context image GPS-query pooling
系统 MUST 允许 `jepa_context_image` encoder 在显式配置 `pooling: gps_query_attention` 时，用 `GPSQueryPool` 替代 patch-token mean pooling。该模式 MUST 继续从 JEPA checkpoint 加载 `context_encoder` 权重，MUST 不要求 JEPA target encoder、latent prediction loss 或 EMA 更新，并 MUST 输出现有 modular sequence projector 可消费的 `[B,T,D]` image feature。

#### Scenario: 启用 GPS-query pooling
- **WHEN** 用户配置 image encoder `type: jepa_context_image` 且 `pooling: gps_query_attention`
- **THEN** 系统 MUST 构建 JEPA context encoder 和 `GPSQueryPool`
- **AND** image encoder forward MUST 接收同 batch/time 的 GPS 条件 feature
- **AND** forward 输出 MUST 保持 `[B,T,D]`

#### Scenario: 缺失 GPS 条件时报错
- **WHEN** `pooling: gps_query_attention` 的 `jepa_context_image` forward 未收到 GPS 条件 feature
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出 GPS-query pooling 需要 GPS condition feature

#### Scenario: mean pooling 默认兼容
- **WHEN** 用户未设置 `pooling` 或设置 `pooling: mean`
- **THEN** `jepa_context_image` MUST 保持现有 patch token mean pooling 行为
- **AND** forward MUST 继续只需要 image input
- **AND** 现有 JEPA downstream 配置 MUST 无需修改即可加载和 forward

### Requirement: fair_gps_biased 派生配置
系统 MUST 提供基于当前 `fair_gps_biased` GPS-biased JEPA 下游主线的 GPS-query pooling 配置。该配置 MUST 复用 GPS-biased JEPA checkpoint、Image+GPS 模态、BeamBench-fair 或同口径 2604-style 数据切分、supervised beam objective、64-beam label space 和既有训练 recipe，只显式改变 image encoder pooling 与相关 metadata。

#### Scenario: GPS-query fair 配置加载
- **WHEN** 开发者加载 GPS-query pooling 派生配置
- **THEN** 配置 MUST 使用 `model.primary.type: modular_sequence`
- **AND** image encoder MUST 为 `jepa_context_image`
- **AND** image encoder `pooling` MUST 为 `gps_query_attention`
- **AND** image encoder checkpoint MUST 指向 GPS-biased JEPA 多场景 checkpoint，而不是 scene31-only checkpoint

#### Scenario: 保留 fair_gps_biased baseline
- **WHEN** 开发者加载现有 `fair_gps_biased` mean-pooling 配置
- **THEN** 配置 MUST 继续使用 `pooling: mean` 或等价默认行为
- **AND** 新 GPS-query 配置 MUST 不替换、删除或重命名现有 baseline 配置

### Requirement: GPS-query pooling metadata
系统 MUST 在最终配置或 runtime metadata 中记录 JEPA downstream pooling 结构，至少包含 pooling 类型、GPS-query 是否启用、`k_queries`、`num_heads`、条件来源、JEPA checkpoint 路径和是否 freeze image encoder。

#### Scenario: 写出 GPS-query metadata
- **WHEN** GPS-query pooling 下游训练完成并写出 `final_config.yaml` 或 runtime metadata
- **THEN** metadata MUST 标记 `pooling` 为 `gps_query_attention`
- **AND** metadata MUST 记录 `k_queries`、`num_heads`、`condition_source` 和 JEPA checkpoint 路径
- **AND** metadata MUST 能与 mean-pooling `fair_gps_biased` baseline 区分
