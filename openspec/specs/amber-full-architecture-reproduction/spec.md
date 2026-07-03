# amber-full-architecture-reproduction Specification

## Purpose
定义 AMBER full architecture 在本仓库中的本地架构复现边界，包括组件化 core、缺失模态 attention mask、auxiliary branches、CMA、loss composition、metadata、claim 和验证要求。

## Requirements
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

#### Scenario: AMBER full 支持非默认输入长度
- **WHEN** AMBER full model 收到与配置默认值不同但不超过 `max_seq_len` 的输入时间长度
- **THEN** core MUST 使用对应长度的位置编码和 attention mask
- **AND** beam logits 的时间维 MUST 匹配实际输入时间长度

### Requirement: 缺失模态感知 attention mask
AMBER full core MUST 使用 availability metadata 构造缺失模态感知 attention mask。fusion token MUST 只能关注可用模态 token；缺失模态 token MAY 使用 learned mask token 表征，但不得作为可用信息参与 fusion attention。

#### Scenario: fusion token 不关注缺失模态
- **WHEN** batch metadata 标记某个模态在某些样本或时间步不可用
- **THEN** AMBER full core MUST 在 attention scores 或等价 mask 中屏蔽对应不可用 token
- **AND** focused test MUST 能证明缺失 token 不改变 fusion token 对可用 token 的 attention 归一化语义

#### Scenario: 普通 baseline 不要求 availability metadata
- **WHEN** 非 AMBER full baseline 运行
- **THEN** missing-modality metadata MUST NOT 成为必需输入
- **AND** 现有 modular_sequence、JEPA、GPS-only 和 AMBER-lite baseline MUST 保持可构建和可 forward

### Requirement: 训练期 AMBER auxiliary branches
AMBER full architecture MUST 在训练期支持论文结构中的 modality-specific branch、fusion branch、input embedding reconstruction/alignment payload、CMA payload 和 beam prediction payload。推理期 MUST 只要求 fusion representation 到 beam head 的主路径可用。

#### Scenario: 训练 forward 返回 auxiliary payload
- **WHEN** AMBER full model 处于 training mode 且配置启用 auxiliary losses
- **THEN** forward 输出 MUST 包含 beam logits 和 AMBER auxiliary payload
- **AND** auxiliary payload MUST 至少标记 modality-specific features、fusion token/features、CMA embeddings 或 logits、availability mask provenance 和 loss-ready tensors

#### Scenario: 推理 forward 保持主输出
- **WHEN** AMBER full model 处于 eval mode 或配置禁用 auxiliary losses
- **THEN** forward 输出 MUST 至少包含 beam logits
- **AND** `adapt_model_output` MUST 能消费该输出而无需 AMBER 专用训练分支

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

### Requirement: AMBER full loss composition
AMBER full training MUST 支持 beam focal loss、embedding L2/alignment loss 和 CMA contrastive loss 的加权总损失。损失接入 MUST 复用现有 loss/objective 扩展点，不得复制训练循环。

#### Scenario: 加权总损失
- **WHEN** AMBER full training batch 包含 beam target 且模型输出 auxiliary payload
- **THEN** training loss MUST 计算配置启用的 beam focal、L2/alignment 和 CMA contrastive 分量
- **AND** runtime metrics MUST 能记录各分量标量和 total loss

#### Scenario: AMBER full 缺少 auxiliary payload 早失败
- **WHEN** AMBER full 配置启用 auxiliary loss 但模型输出缺少必要 payload
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出缺失的 AMBER loss payload 字段

### Requirement: AMBER full metadata and claim boundary
AMBER full reproduction MUST 记录可审计 metadata，并 MUST 标记为本地 architecture reproduction。缺少真实严格可比训练和评估证据时，系统 MUST NOT 声称官方 AMBER 数值复现。

#### Scenario: metadata 记录架构与训练策略
- **WHEN** AMBER full model 被构建或训练
- **THEN** metadata MUST 包含 `reproduction_scope: amber_full_local`、enabled modalities、history beam usage、core type、mask strategy、CMA enabled、auxiliary loss weights、consumes missing metadata 和 output boundary
- **AND** 模型架构摘要 MUST 能记录 AMBER full core 的参数量和组件类别

#### Scenario: claim 不自动升级
- **WHEN** AMBER full metrics、summary 或文档缺少 strict comparability fields、真实 checkpoint 或 condition-level evaluation
- **THEN** row MUST 标记为 pending、unverified、unavailable 或 not_comparable
- **AND** row MUST NOT 进入 official reproduction 或 strict ranking claim

### Requirement: AMBER full validation boundary
AMBER full implementation MUST 提供 focused tests，覆盖 registry/config build、synthetic forward、mask attention、auxiliary loss、metadata、architecture summary 和 AMBER-lite 回归。测试 MUST 不读取真实 `dataset/`、checkpoint、cache 或本地运行产物。

#### Scenario: synthetic tests 覆盖核心行为
- **WHEN** 开发者运行 AMBER full focused tests
- **THEN** tests MUST 使用 synthetic tensors 或 dry-run manifest 验证 AMBER full forward、缺失 mask、loss composition 和 metadata
- **AND** tests MUST NOT 依赖真实 DeepSense6G 文件、外部权重或训练输出

#### Scenario: 输出产物保持 ignored
- **WHEN** AMBER full training、evaluation 或 diagnostics 生成本地产物
- **THEN** checkpoint、metrics、prediction、cache、figures 和 reports MUST 写入 ignored `outputs/analysis/local_baselines/amber_full_architecture/` 或用户显式指定 output root
- **AND** 源码变更 MUST 只包含代码、配置、测试、OpenSpec 和文档
