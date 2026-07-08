## MODIFIED Requirements

### Requirement: WCL 2025 missing-modality model
WCL 2025 / RMBP-MM local substitute MUST 支持论文对齐的缺失模态 beam prediction 结构。实现 MUST 优先使用可组合 encoder/projector/core/head 或窄 workflow owner。本地可训练配置 MUST 使用 `seq_len=2`、`num_pred=1`，并且只启用 `image`、`radar`、`gps`、`lidar` 四个模态；论文中的 beam measurement / historical beam 或其它非允许模态不得作为模型输入。

#### Scenario: 构建 local substitute 模型
- **WHEN** 配置声明 WCL 2025 local substitute
- **THEN** 系统 MUST 构建论文对齐的 per-modality encoder 和 missing-modality fusion 结构
- **AND** metadata MUST 记录 enabled modalities、missing-modality strategy、fusion type、paper alignment 和 deviation
- **AND** metadata MUST NOT 声明使用 `mmwave`、`csi` 或 beam measurement 输入

#### Scenario: whole-model exception 需要理由
- **WHEN** WCL 2025 结构无法表达为可组合组件并需要新增完整模型
- **THEN** design 或 implementation note MUST 说明 whole-model exception 理由
- **AND** tasks MUST 覆盖 registry build、synthetic forward、ModelOutput adaptation 和 metadata tests

#### Scenario: local substitute 默认窗口和模态受限
- **WHEN** 用户加载 `configs/fusion/experiments/wcl2025_missing_modality/local_substitute.yaml`
- **THEN** data 和 model 配置 MUST 声明 `seq_len=2` 与 `num_pred=1`
- **AND** `model.primary.modalities` MUST 等于 `["image", "radar", "gps", "lidar"]`
- **AND** 配置 MUST NOT 启用 `mmwave` 或其它非允许输入模态

## ADDED Requirements

### Requirement: RMBP-MM channel-attention fusion core
RMBP-MM local substitute MUST 提供可注册的 representation core，实现论文的 adaptive multimodal feature fusion：对 modality feature matrix 沿 feature 维执行 global average pooling 和 global max pooling，两个 pooled 向量经过共享 bottleneck MLP 后相加并 sigmoid 得到 modality attention weights，再用 missing-modality mask 将缺失模态权重置零，最后生成可被 beam head 消费的融合表征。

#### Scenario: channel-attention core 可构建和 forward
- **WHEN** `modular_sequence` 配置选择 `representation_core.type: rmbp_channel_attention_fusion`
- **THEN** 系统 MUST 通过 `REPRESENTATION_CORES` 构建该 core
- **AND** core MUST 接收多模态 `[B, K, T, D]` 输入
- **AND** core MUST 输出 `[B, T, output_dim]` 表征

#### Scenario: 缺失模态权重置零
- **WHEN** core 收到 availability mask 标记某个模态缺失
- **THEN** 对应模态 attention weight MUST 为 0
- **AND** 输出 MUST 只由可用模态和可学习 fusion projection 产生
- **AND** diagnostics 或 metadata MUST 能记录 channel attention 和 missing mask 语义

### Requirement: RMBP-MM missing-modality augmentation helper
RMBP-MM workflow MUST 提供本地 batch augmentation helper，用于表达论文的 random modality masking 和 similarity-based modality imputation。该 helper MUST 只修改输入模态 tensor 及对应 valid/missing mask，不得修改 target beam、beam power、sample id、split metadata 或其它 protected target 字段。

#### Scenario: random available modality masking
- **WHEN** batch 中一个样本至少有两个可用输入模态
- **THEN** augmentation helper MAY 随机选择一个可用模态置为 zero-filled missing input
- **AND** 对应 `<modality>_valid_mask` MUST 标记为 false
- **AND** helper metadata MUST 记录 masked modality 和 seed

#### Scenario: similarity-based imputation
- **WHEN** batch 中存在同 beam label 且可提供目标缺失模态的 donor 样本
- **THEN** augmentation helper MAY 将 donor 的该模态输入复制给当前样本
- **AND** helper metadata MUST 记录 imputed modality、donor index 和 similarity source
- **AND** 找不到 donor 时 MUST 使用 zero-imputation fallback 并记录 skipped/fallback count
