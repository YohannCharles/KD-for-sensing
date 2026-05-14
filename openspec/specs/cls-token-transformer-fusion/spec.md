# cls-token-transformer-fusion Specification

## Purpose
TBD - created by archiving change add-cls-token-transformer-fusion-default. Update Purpose after archive.
## Requirements
### Requirement: CLS-token Transformer fusion 模型构建
系统 MUST 提供可通过 registry 构建的 CLS-token Transformer fusion 模型。该模型 MUST 使用现有 `experiment.task: fusion` 输入契约，支持 `image`、`radar`、`gps`、`lidar`、`mmwave` 的任意合法非空组合，并默认适配五模态融合。

#### Scenario: 构建五模态 CLS-token Transformer fusion
- **WHEN** 用户配置 `model.student.type: cls_token_transformer_fusion` 且 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]`
- **THEN** 系统 MUST 构建五个模态 encoder 或 projector
- **AND** 模型 MUST 接收现有 fusion 输入键对应的五个张量
- **AND** 模型 MUST 输出 beam logits

#### Scenario: 构建任意模态子集
- **WHEN** 用户配置 `cls_token_transformer_fusion` 且 `modalities` 为任意合法非空模态组合
- **THEN** 模型 MUST 只要求该组合对应的输入张量
- **AND** 模型 diagnostics 中的模态顺序 MUST 与标准化后的 `modalities` 一致

#### Scenario: 不引入新的训练入口
- **WHEN** 用户使用 CLS-token Transformer fusion 配置训练或评估
- **THEN** 系统 MUST 继续复用现有 fusion batch preparation、forward helper、loss、metric 和 checkpoint 流程
- **AND** 系统 MUST 不要求新增训练脚本或评估脚本入口

### Requirement: Token 序列化与嵌入
CLS-token Transformer fusion MUST 将每个启用模态的帧级特征映射到统一 `d_model` 后序列化为 token。五模态场景下，每个历史时间步 MUST 产生 `5 x d_model` 的模态 token 序列；完整 Transformer 输入 MUST 在所有模态 token 前添加一个可学习 CLS token。

#### Scenario: 五模态 token 序列化
- **WHEN** encoder/projector 产生五个模态的特征 `[B, T, d_model]`
- **THEN** 模型 MUST 按固定模态顺序和时间维将其序列化为 `[B, T*5, d_model]`
- **AND** 每个时间步内部 MUST 保留五个独立模态 token，而不是先拼接为单个 `5*d_model` 向量

#### Scenario: CLS token 前置
- **WHEN** 模型构造 Transformer 输入序列
- **THEN** 模型 MUST 在序列首位添加可学习 CLS token
- **AND** Transformer 输入长度 MUST 为 `1 + T*K`，其中 `K` 为启用模态数

#### Scenario: token-type embedding 区分传感器
- **WHEN** 模态 token 进入 Transformer Encoder
- **THEN** 每个模态 token MUST 加上对应传感器类型的 token-type embedding
- **AND** CLS token MUST 使用独立的 CLS 类型编码或等价独立参数

#### Scenario: time embedding 区分历史时间
- **WHEN** 不同历史时间步的模态 token 进入 Transformer Encoder
- **THEN** 模型 MUST 为模态 token 添加 time embedding
- **AND** 相同模态在不同时间步的 token MUST 能被 Transformer 区分

### Requirement: Transformer Encoder 融合与输出契约
CLS-token Transformer fusion MUST 使用包含多头自注意力和前馈网络的 Transformer Encoder 处理 CLS token 与模态 token。模型 MUST 通过 CLS 表示生成未来 beam prediction logits，并兼容现有 `ModelOutput` 适配逻辑。

#### Scenario: Transformer Encoder 处理 token 序列
- **WHEN** CLS token、token-type embedding 和 time embedding 已经加入输入序列
- **THEN** 模型 MUST 使用一层或多层 Transformer Encoder 执行多头自注意力和前馈变换
- **AND** 各模态信息 MUST 能在 Transformer Encoder 中通过 self-attention 交互

#### Scenario: 输出 future prediction slots
- **WHEN** batch size 为 `B`、配置 `num_pred` 为 `H`、beam 类别数为 `C`
- **THEN** 主 logits MUST 具有形状 `[B, H, C]`
- **AND** 该 logits MUST 能直接传入现有 `select_prediction_slots()`、loss 和 metric 流程

#### Scenario: 输出适配器解析
- **WHEN** CLS-token Transformer fusion forward 返回结果
- **THEN** `adapt_model_output()` MUST 能解析 `logits`、`input_features`、`output_features` 和 diagnostics
- **AND** `output_features` MUST 表示可用于 KD 或诊断的 fused CLS/horizon representation

### Requirement: 模态 mask 与 diagnostics
CLS-token Transformer fusion MUST 支持 `force_modality_mask`，并 MUST 输出足够的 diagnostics 以支持 G2D、模态子集评估和调试。

#### Scenario: force_modality_mask 排除模态 token
- **WHEN** forward 传入 `force_modality_mask` 屏蔽某个启用模态
- **THEN** 该模态对应的 token MUST 通过 attention padding mask 或等价机制从 Transformer 有效上下文中排除
- **AND** 被屏蔽模态 MUST 不通过 token-type embedding 或 time embedding 对 CLS 表示产生有效贡献

#### Scenario: 空可用模态被拒绝
- **WHEN** `force_modality_mask` 导致某个样本没有任何可用模态
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 系统 MUST 不静默使用全零 token 或退回 all-modal forward

#### Scenario: 输出 token diagnostics
- **WHEN** 模型 forward 完成
- **THEN** diagnostics MUST 包含 `token_features`、`modalities`、`effective_modality_mask` 和 `fusion_memory` 或等价 Transformer memory
- **AND** `token_features` MUST 能按 `[B, K, T, d_model]` 或等价可逆结构拆分到每个模态

#### Scenario: G2D feature diagnostics 兼容
- **WHEN** G2D distiller 或诊断工具读取 CLS-token Transformer fusion 输出
- **THEN** 系统 MUST 能从 diagnostics 中按模态拆分 feature
- **AND** 主 logits 契约 MUST 不因 diagnostics 输出而改变

