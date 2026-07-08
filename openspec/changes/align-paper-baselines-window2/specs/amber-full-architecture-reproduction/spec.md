## MODIFIED Requirements

### Requirement: AMBER full architecture core
系统 MUST 提供 paper-aligned AMBER full architecture core，用于融合 image、LiDAR、radar 和 GPS 四个模态表征。该 core MUST 支持四模态输入 embedding、spatial/time/modality positional embedding、learnable fusion token、modality-specific transformer、modality-fusion transformer、modality indicator reweighting、L2 regularization payload 和 beam prediction head 所需输出。AMBER full MUST NOT 构建历史 beam index 输入、beam encoder、`learned_history_beam_token` 或等价第五输入 token。
AMBER full MUST 支持运行时输入时间长度 `T` 在 `1 <= T <= max_seq_len` 范围内变化；本地默认配置 MUST 使用 `seq_len=2`、`num_pred=1`，且只启用 `image`、`radar`、`gps`、`lidar` 四个模态。

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

#### Scenario: AMBER full 默认窗口和模态受限
- **WHEN** 用户加载 `configs/fusion/amber_full_architecture.yaml`
- **THEN** data 和 model 配置 MUST 声明 `seq_len=2` 与 `num_pred=1`
- **AND** `model.primary.modalities` MUST 等于 `["image", "radar", "gps", "lidar"]`
- **AND** 配置 MUST NOT 启用 `mmwave`、`csi`、历史 beam index 或其它非允许模态输入

### Requirement: AMBER ResNet18 spatial-token encoders
AMBER full 配置 MUST 为 radar 和 LiDAR 使用 ResNet18-backed spatial-token encoder，并 MUST 为 image 使用 ResNet34-backed spatial-token encoder。用于 AMBER full 的 image、radar 和 LiDAR encoder MUST 开启预训练权重配置，并 MUST 能保留 feature-map spatial tokens 或等价 tokenized 表征供 AMBER core 使用；GPS MAY 保持 MLP 单 token 表征。

#### Scenario: AMBER full encoder 配置对齐论文修订
- **WHEN** 用户加载 `configs/fusion/amber_full_architecture.yaml`
- **THEN** image encoder MUST 声明 ResNet34-backed spatial-token 类型
- **AND** radar 和 LiDAR encoder MUST 声明 ResNet18-backed spatial-token 类型
- **AND** image、radar 和 LiDAR 三者 MUST 开启 pretrained/weights 配置
- **AND** AMBER core MUST 能从这些 encoder 接收空间 token 或等价 token 表征

#### Scenario: 无历史 beam 输入
- **WHEN** AMBER full model forward 运行
- **THEN** 模型 MUST 只要求 image、radar、GPS 和 LiDAR batch
- **AND** 模型 MUST NOT 要求或生成历史 beam index input token

### Requirement: AMBER full loss composition
AMBER full training MUST 支持 beam focal loss、modality indicator L2 regularization loss 和 CMA contrastive loss 的加权总损失。损失接入 MUST 复用现有 loss/objective 扩展点，不得复制训练循环。系统 MAY 保留 embedding alignment diagnostics，但论文对齐 L2 分量 MUST 优先使用 AMBER core 输出的 modality indicator payload。

#### Scenario: 加权总损失
- **WHEN** AMBER full training batch 包含 beam target 且模型输出 auxiliary payload
- **THEN** training loss MUST 计算配置启用的 beam focal、L2 regularization 和 CMA contrastive 分量
- **AND** runtime metrics MUST 能记录各分量标量和 total loss

#### Scenario: AMBER full 缺少 auxiliary payload 早失败
- **WHEN** AMBER full 配置启用 auxiliary loss 但模型输出缺少必要 payload
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出缺失的 AMBER loss payload 字段

#### Scenario: modality indicator L2 payload 可审计
- **WHEN** AMBER full model 处于 training mode 且 auxiliary loss 启用
- **THEN** auxiliary payload MUST 包含 modality indicator weights、availability mask 和可反向传播的 L2 regularization tensor
- **AND** metadata MUST 记录该 L2 分量来自 AMBER modality indicator，而不是旧式 residual 或 retired route
