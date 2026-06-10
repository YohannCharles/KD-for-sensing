# modular-sequence-model Specification

## Purpose
定义模块化序列模型、encoder/core/head 组合和单模态/fusion 复用边界，约束新增模态组件在统一结构中扩展。
## Requirements
### Requirement: 模块化序列模型结构
系统 MUST 提供新的模块化序列模型注册入口，用于组合 `encoders`、`projectors`、`representation_core` 和 `heads`。该入口 MUST 支持 image-only 和多模态 fusion 实验，并且 MUST 不要求修改训练、验证和评估循环主体。

#### Scenario: 构建 image-only 模块化模型
- **WHEN** 配置选择模块化序列模型且启用模态为 `["image"]`
- **THEN** 系统 MUST 构建 image encoder、image projector、representation core 和 beam head
- **AND** forward MUST 接收 `image_batch`
- **AND** 输出 MUST 兼容现有 beam prediction loss、metric 和 `ModelOutput` 适配逻辑

#### Scenario: 构建多模态模块化模型
- **WHEN** 配置选择模块化序列模型且启用多个合法模态
- **THEN** 系统 MUST 按模态契约构建对应 encoder 和 projector
- **AND** 所有启用模态的 encoder 输出 MUST 统一为 `[B, T, D]`
- **AND** representation core MUST 接收统一后的单模态或多模态 representation

### Requirement: Encoder 和 projector 契约
模块化序列模型中的每个 encoder MUST 只负责把原始模态输入编码为 `[B, T, D_raw]`。每个 projector MUST 将对应模态的 `D_raw` 映射到统一 `d_model`，并输出 `[B, T, d_model]`。多模态堆叠后 MUST 形成可被 token 或 fusion core 消费的 `[B, K, T, d_model]` 或等价结构。

#### Scenario: 单模态 projector
- **WHEN** image encoder 输出 `[B, T, D_raw]`
- **THEN** image projector MUST 输出 `[B, T, d_model]`
- **AND** 后续 core MUST 不依赖 ResNet-18 或 motion CNN 的内部通道数

#### Scenario: 多模态时间维对齐
- **WHEN** 模块化模型启用 image、radar、GPS、LiDAR 或 mmWave 中的多个模态
- **THEN** 系统 MUST 校验各模态 projector 输出的 batch 和 time 维一致
- **AND** 不一致时 MUST 抛出包含模态名和实际 shape 的清晰错误

### Requirement: Representation core 可插拔
模块化序列模型 MUST 支持可插拔 representation core，用于统一表达单模态时序建模、early concat GRU、token transformer 或等价实现。core MUST 接收统一 embedding，不得直接读取 dataset 字段或执行模态特定预处理。

#### Scenario: single_gru core
- **WHEN** 用户配置 `representation_core.type: single_gru`
- **THEN** core MUST 对 `[B, T, d_model]` 输入执行时序建模
- **AND** core 输出 MUST 可被 beam head 转换为 `[B, T, num_classes]` logits

#### Scenario: token transformer core
- **WHEN** 用户配置 token transformer 风格 core 且启用多个模态
- **THEN** core MUST 基于模态和时间 token 建模跨模态关系与时间关系
- **AND** core MUST 不要求调用 legacy fusion 拼接层

### Requirement: Task heads 与输出契约
模块化序列模型 MUST 通过 head 产生任务输出。beam classification head MUST 输出 `[B, T, num_classes]` logits；可选辅助 head MAY 输出 blockage 或 position regression，但 loss 选择和 label 映射 MUST 保持在训练配置或训练侧逻辑中。

#### Scenario: beam head 输出
- **WHEN** 模块化模型执行 beam prediction forward
- **THEN** 输出 MUST 包含 beam logits
- **AND** logits 的 batch/time/class 维 MUST 与现有训练和评估流程兼容

#### Scenario: 辅助 head 不破坏 beam 训练
- **WHEN** 配置启用 blockage 或 position auxiliary head
- **THEN** 模型输出 MUST 同时保留 beam logits
- **AND** 未配置辅助 loss 时训练流程 MUST 能忽略辅助输出并继续 beam-only 训练

### Requirement: 新入口不破坏保留模型
模块化序列模型 MUST 作为新注册入口存在，不得替换或重命名现有单模态或保留的 legacy fusion 注册名。实现 MAY 复用现有 encoder/core 代码，但 MUST 保持保留模型构造参数和 forward 语义兼容。

#### Scenario: 保留注册名仍可构建
- **WHEN** 构建流程导入默认模型组件后请求 `image_teacher`、`image_student`、`fusion_teacher`、`fusion_student` 或当前保留 fusion 注册名
- **THEN** 系统 MUST 继续返回对应现有模型
- **AND** 这些模型 MUST 不要求新增模块化配置字段

#### Scenario: 新注册名独立构建
- **WHEN** 用户配置新的模块化序列模型注册名
- **THEN** 系统 MUST 只使用该入口解析 encoder、projector、core 和 head 配置
- **AND** 错误信息 MUST 指向缺失或非法的模块化子配置

### Requirement: Snapshot frame representation core
模块化序列模型 MUST 提供 `snapshot_frame` representation core，用于无历史窗口的当前帧预测。该 core MUST 支持单模态 `[B, 1, D]` 输入和多模态 `[B, K, 1, D]` 输入，并输出可被现有 heads 消费的 `[B, 1, D_out]` 表示。

#### Scenario: 单模态 snapshot core
- **WHEN** `modular_sequence` 配置启用单个模态并设置 `representation_core.type: snapshot_frame`
- **THEN** core MUST 接收该模态 projector 输出 `[B, 1, d_model]`
- **AND** core MUST 输出 `[B, 1, output_dim]`
- **AND** beam head MUST 生成 `[B, 1, num_classes]` logits

#### Scenario: 多模态 snapshot core
- **WHEN** `modular_sequence` 配置启用多个模态并设置 `representation_core.type: snapshot_frame`
- **THEN** core MUST 接收堆叠后的 `[B, K, 1, d_model]` 模态表示
- **AND** core MUST 只在当前帧的 `K` 个模态表示之间执行融合
- **AND** core MUST 输出 `[B, 1, output_dim]`

#### Scenario: 拒绝历史时间维
- **WHEN** `snapshot_frame` core 收到时间维 `T` 不等于 1 的输入
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出 snapshot baseline 需要 `seq_len=1` 和 `num_pred=1`

#### Scenario: 不创建 GRU 模块
- **WHEN** 开发者构建启用 `snapshot_frame` core 的 `modular_sequence` 模型
- **THEN** 模型 MUST 不创建 `single_gru` 或 `early_concat_gru`
- **AND** 模型模块树中 MUST 不包含 GRU、RNN 或 LSTM 子模块

### Requirement: Snapshot core 辅助 head 兼容
`snapshot_frame` core MUST 保持模块化模型的 head 输出契约。启用遮挡或位置 auxiliary heads 时，辅助输出 MUST 与 `num_pred=1` 的 next-frame horizon 对齐。

#### Scenario: Snapshot 遮挡输出
- **WHEN** snapshot `modular_sequence` 配置启用 `auxiliary_heads.occlusion`
- **THEN** forward 输出 MUST 包含形状 `[B, 1]` 的 `occlusion_logits`
- **AND** 主 beam logits MUST 继续保持 `[B, 1, num_classes]`

#### Scenario: Snapshot 位置输出
- **WHEN** snapshot `modular_sequence` 配置启用 `auxiliary_heads.position`
- **THEN** forward 输出 MUST 包含形状 `[B, 1, 2]` 的 `position`
- **AND** 输出 MUST 能被现有 objective-aware loss 和 metrics 消费

### Requirement: Next-beam query Transformer representation core
模块化序列模型 MUST 支持 `next_beam_query_transformer` representation core，用于多模态历史输入的下一时刻单步预测。该 core MUST 接收多模态 projector 输出 `[B, K, T, D]`，注入 time embedding 与 modality embedding，追加 learned next-beam query token，并输出可被现有 heads 消费的 `[B, 1, D_out]` 表征。

#### Scenario: 构建 next-beam query core
- **WHEN** 用户配置 `model.primary.representation_core.type: next_beam_query_transformer`
- **THEN** 系统 MUST 构建注册到 `REPRESENTATION_CORES` 的 next-beam query Transformer core
- **AND** core 配置 MUST 支持 `d_model`、`modality_count`、`num_heads`、`num_layers`、`dropout`、`max_seq_len` 和 `output_dim`

#### Scenario: 多模态历史 token 输入
- **WHEN** `next_beam_query_transformer` core 收到 `[B, K, T, D]` 输入
- **THEN** core MUST 校验 `K` 与配置的 `modality_count` 一致
- **AND** core MUST 校验 `D` 与配置的 `d_model` 一致
- **AND** core MUST 输出 `[B, 1, D_out]`

#### Scenario: 拒绝单模态三维输入
- **WHEN** `next_beam_query_transformer` core 收到 `[B, T, D]` 输入
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出 next-beam query Transformer 需要多模态 `[B, K, T, D]` 输入

#### Scenario: 时间长度超过上限
- **WHEN** 输入时间维 `T` 大于配置的 `max_seq_len`
- **THEN** 系统 MUST 拒绝 forward
- **AND** 错误信息 MUST 包含实际 `T` 和 `max_seq_len`

### Requirement: Next-beam query Transformer token embedding
`next_beam_query_transformer` MUST 显式区分模态来源、时间位置和查询 token。历史 token MUST 加上 modality embedding 与 time embedding；learned next-beam query MUST 作为独立 token 参与 Transformer 编码，并且最终输出 MUST 来自该 query token。

#### Scenario: 注入模态和时间 embedding
- **WHEN** core 对 `[B, K, T, D]` 历史 token 执行 forward
- **THEN** 每个历史 token MUST 加上对应模态 embedding
- **AND** 每个历史 token MUST 加上对应时间位置 embedding

#### Scenario: 使用 query token 输出
- **WHEN** Transformer 编码完成
- **THEN** core MUST 取 learned next-beam query token 的编码结果作为输出
- **AND** core MUST NOT 对全部历史 token 简单 mean pooling 作为 next-query 主输出

### Requirement: Next-beam query Transformer 与模块化 head 兼容
`next_beam_query_transformer` MUST 保持模块化模型的 head 输出契约。对于 beam prediction，现有 `beam_head` MUST 能消费 core 输出 `[B, 1, D_out]` 并产生 `[B, 1, num_classes]` logits。

#### Scenario: beam head 消费 next-query 输出
- **WHEN** `ModularSequenceModel` 使用 `next_beam_query_transformer` core 和 `beam_head`
- **THEN** model forward MUST 返回 `logits` 字段
- **AND** `logits` 形状 MUST 为 `[B, 1, num_classes]`

#### Scenario: 保留中间诊断字段
- **WHEN** `ModularSequenceModel` 使用 `next_beam_query_transformer`
- **THEN** model forward MUST 继续返回 `input_features`、`output_features`、`modalities`、`modality_features` 和 `encoder_features`
- **AND** `output_features` 时间维 MUST 为 `1`

