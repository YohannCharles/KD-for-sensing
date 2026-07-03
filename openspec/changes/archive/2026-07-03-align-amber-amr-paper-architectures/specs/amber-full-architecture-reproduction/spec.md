## MODIFIED Requirements

### Requirement: AMBER full architecture core
系统 MUST 提供 paper-aligned AMBER full architecture core，用于融合 image、LiDAR、radar 和 GPS 四个模态表征。该 core MUST 支持四模态输入 embedding、spatial/time/modality positional embedding、learnable fusion token、modality-specific transformer、modality-fusion transformer 和 beam prediction head 所需输出。AMBER full MUST NOT 构建历史 beam index 输入、beam encoder、`learned_history_beam_token` 或等价第五输入 token。
AMBER full MUST 支持运行时输入时间长度 `T` 在 `1 <= T <= max_seq_len` 范围内变化；本地默认配置 MUST 使用 `seq_len=2`、`num_pred=1`。

#### Scenario: 构建 AMBER full 模块化配置
- **WHEN** 配置声明 `model.primary.type: modular_sequence` 且 representation core 选择 AMBER full architecture
- **THEN** 系统 MUST 构建 image、LiDAR、radar 和 GPS 四个输入路径
- **AND** representation core MUST 接收统一的四模态 token 表征
- **AND** beam head MUST 能消费 AMBER fusion representation 并输出 beam logits
- **AND** metadata MUST NOT 把历史 beam token 标记为 enabled

#### Scenario: AMBER full 不覆盖 AMBER-lite
- **WHEN** 用户加载 AMBER-lite 配置
- **THEN** 系统 MUST 继续构建 AMBER-lite core
- **AND** 系统 MUST NOT 静默切换到 AMBER full architecture core

### Requirement: CMA class-query contrastive module
系统 MUST 为 AMBER full reproduction 提供 Class-Former-aided Modality Alignment 组件。该组件 MUST 使用 fusion class query 和可用模态 class queries，通过 cross-attention 或等价 query-to-token attention 得到 class-level embeddings，并基于 fusion query 与可用模态 query 的正样本关系计算 contrastive training payload。该组件 MUST 支持配置化 temperature、embedding dimension 和 loss weight。

#### Scenario: CMA payload 可计算 contrastive loss
- **WHEN** AMBER full 配置启用 CMA contrastive loss
- **THEN** 模型 forward MUST 输出 fusion query embedding、modality query embeddings、availability mask 和 contrastive logits 或等价 loss-ready tensors
- **AND** loss helper MUST 根据配置 temperature 和 weight 计算可反向传播的标量 loss
- **AND** loss helper MUST 使用 class-query payload，而不是仅使用 pooled fusion/modality feature 的简化余弦 logits

#### Scenario: CMA 不污染普通配置
- **WHEN** 配置未启用 AMBER CMA loss
- **THEN** 训练流程 MUST 不要求模型输出 CMA payload
- **AND** 普通 focal loss 或现有 beam prediction loss MUST 保持原语义

## ADDED Requirements

### Requirement: AMBER ResNet18 spatial-token encoders
AMBER full 配置 MUST 为 image、radar 和 LiDAR 使用 ResNet18-backed encoder，并开启预训练权重配置。用于 AMBER full 的 image、radar 和 LiDAR encoder MUST 能保留 feature-map spatial tokens 或等价 tokenized 表征供 AMBER core 使用；GPS MAY 保持 MLP 单 token 表征。

#### Scenario: AMBER full encoder 配置对齐论文修订
- **WHEN** 用户加载 `configs/fusion/amber_full_architecture.yaml`
- **THEN** image、radar 和 LiDAR encoder MUST 声明 ResNet18-backed 类型
- **AND** 三者 MUST 开启 pretrained/weights 配置
- **AND** AMBER core MUST 能从这些 encoder 接收空间 token 或等价 token 表征

#### Scenario: 无历史 beam 输入
- **WHEN** AMBER full model forward 运行
- **THEN** 模型 MUST 只要求 image、radar、GPS 和 LiDAR batch
- **AND** 模型 MUST NOT 要求或生成历史 beam index input token

#### Scenario: AMBER full 支持非默认输入长度
- **WHEN** AMBER full model 收到与配置默认值不同但不超过 `max_seq_len` 的输入时间长度
- **THEN** core MUST 使用对应长度的位置编码和 attention mask
- **AND** beam logits 的时间维 MUST 匹配实际输入时间长度
