# cls-token-transformer-fusion Specification

## Purpose
定义 CLS-token Transformer fusion student 的配置、构建和兼容行为。
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
CLS-token Transformer fusion MUST 使用包含多头自注意力和前馈网络的 Transformer Encoder 处理 CLS token 与模态 token。模型 MUST 通过 CLS 表示生成未来 beam prediction logits，并兼容现有 `ModelOutput` 适配逻辑。`output_features` 若存在，MUST 用于诊断、auxiliary objective 或 downstream supervised/adaptation workflow，不得作为 KD 兼容要求。

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
- **AND** `output_features` MUST 表示可用于诊断、auxiliary objective 或 downstream supervised/adaptation workflow 的 fused CLS/horizon representation
- **AND** `output_features` MUST 不被要求服务 RKD 或其它 KD loss

### Requirement: 模态 mask 与 diagnostics
CLS-token Transformer fusion MUST 支持 `force_modality_mask`，并 MUST 输出足够的 diagnostics 以支持模态子集评估和调试。

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

### Requirement: CLS-token Transformer fusion 辅助头
CLS-token Transformer fusion MUST 支持可选遮挡检测头和位置估算头。辅助头启用时 MUST 从融合后的 CLS 表示或等价 horizon representation 生成输出；辅助头关闭时 MUST 不改变现有模型构建、forward 输入和 beam logits 输出。

#### Scenario: 构建辅助头
- **WHEN** 用户配置 `cls_token_transformer_fusion` 并启用 `auxiliary_heads.occlusion` 和 `auxiliary_heads.position`
- **THEN** 模型 MUST 创建遮挡检测头和位置估算头
- **AND** 模型 MUST 继续创建主 beam prediction head

#### Scenario: 辅助头关闭
- **WHEN** 用户未配置 auxiliary heads 或显式关闭 auxiliary heads
- **THEN** 模型 MUST 保持现有 beam-only output dict 兼容
- **AND** `adapt_model_output()` MUST 继续解析主 logits、input features、output features 和 diagnostics

### Requirement: 辅助输出 horizon 对齐
CLS-token Transformer fusion 的辅助输出 MUST 与主 beam prediction horizon 对齐。遮挡输出 MUST 是每个 future slot 的 logit，位置输出 MUST 是每个 future slot 的二维坐标。

#### Scenario: 五模态辅助输出形状
- **WHEN** 用户构建五模态 CLS-token Transformer fusion，batch size 为 `B`，`num_pred` 为 `H`
- **THEN** forward 返回的主 logits MUST 具有形状 `[B, H, num_classes]`
- **AND** forward 返回的 `occlusion_logits` MUST 具有形状 `[B, H]`
- **AND** forward 返回的 `position` MUST 具有形状 `[B, H, 2]`

#### Scenario: 任意模态子集辅助输出形状
- **WHEN** 用户构建任意合法非空模态子集并启用 auxiliary heads
- **THEN** 模型 MUST 只要求该模态子集的输入张量
- **AND** 辅助输出 shape MUST 只依赖 batch size 和 `num_pred`，不得依赖启用模态数量

### Requirement: 辅助头与模态 mask 兼容
CLS-token Transformer fusion 在使用 `force_modality_mask` 时 MUST 让辅助头基于同一个被 mask 后的 CLS 表示输出，确保 beam、遮挡和位置预测使用一致的有效模态上下文。

#### Scenario: 屏蔽模态后辅助输出仍可用
- **WHEN** forward 传入合法的 `force_modality_mask` 并启用 auxiliary heads
- **THEN** 模型 MUST 使用被 mask 后的 Transformer memory 生成 `occlusion_logits` 和 `position`
- **AND** 输出 diagnostics MUST 继续包含 `effective_modality_mask`

#### Scenario: 空模态 mask 仍被拒绝
- **WHEN** `force_modality_mask` 导致某个样本没有任何可用模态
- **THEN** 模型 MUST 抛出清晰错误
- **AND** 模型 MUST 不生成 beam、遮挡或位置输出

### Requirement: Auxiliary heads 可作为 primary objective
CLS-token Transformer fusion MUST 支持将 `occlusion_head` 和 `position_head` 作为 primary objective 输出使用。配置为 `occlusion` 或 `position` objective 时，模型 MUST 启用对应 head，并保持主 beam logits 输出兼容。

#### Scenario: occlusion primary output
- **WHEN** `experiment.objective` 为 `occlusion` 且模型类型为 `cls_token_transformer_fusion`
- **THEN** 模型配置 MUST 启用 `auxiliary_heads.occlusion`
- **AND** forward 输出 MUST 包含形状为 `[B, H]` 的 `occlusion_logits`

#### Scenario: position primary output
- **WHEN** `experiment.objective` 为 `position` 且模型类型为 `cls_token_transformer_fusion`
- **THEN** 模型配置 MUST 启用 `auxiliary_heads.position`
- **AND** forward 输出 MUST 包含形状为 `[B, H, 2]` 的 `position`

#### Scenario: multitask primary outputs
- **WHEN** `experiment.objective` 为 `multitask` 且模型类型为 `cls_token_transformer_fusion`
- **THEN** 模型配置 MUST 启用 beam、occlusion 和 position 所需输出
- **AND** forward 输出 MUST 同时提供 beam logits、`occlusion_logits` 和 `position`

### Requirement: Objective head 校验
配置校验 MUST 在训练开始前确认当前 objective 所需的模型 head 可用。模型不支持当前 objective 时，系统 MUST 拒绝配置并给出可执行的修复提示。

#### Scenario: occlusion head 未启用
- **WHEN** 配置设置 `experiment.objective: occlusion` 但 `model.student.auxiliary_heads.occlusion` 未启用
- **THEN** 系统 MUST 拒绝加载配置
- **AND** 错误信息 MUST 提示启用 `model.student.auxiliary_heads.occlusion=true`

#### Scenario: position head 未启用
- **WHEN** 配置设置 `experiment.objective: position` 但 `model.student.auxiliary_heads.position` 未启用
- **THEN** 系统 MUST 拒绝加载配置
- **AND** 错误信息 MUST 提示启用 `model.student.auxiliary_heads.position=true`

### Requirement: 默认 fusion no-KD 使用 CLS-token Transformer
推荐/default fusion no-KD 配置 MUST 使用 CLS-token Transformer fusion 作为混合方式。显式命名为 legacy 或 early-concat 的保留配置 MUST 保持其方法语义，不得被默认行为覆盖；已退役的 CRAF、MARF、G2D 或相关 ablation 配置 MUST 不再作为支持入口存在。

#### Scenario: 加载五模态默认 fusion no-KD
- **WHEN** 用户加载推荐的五模态 fusion no-KD 配置
- **THEN** 配置 MUST 启用 `["image", "radar", "gps", "lidar", "mmwave"]`
- **AND** `model.student.type` MUST 为 `cls_token_transformer_fusion`
- **AND** 配置 MUST 设置 CLS-token Transformer 所需的 `d_model`、`num_heads`、`num_layers` 或等价默认值

#### Scenario: 加载双模态默认 fusion no-KD
- **WHEN** 用户加载推荐的双模态 fusion no-KD 配置
- **THEN** 配置 MUST 使用 slug 表示的两个模态
- **AND** `model.student.type` MUST 为 `cls_token_transformer_fusion`
- **AND** dataset 字段 MUST 只启用该组合需要的模态数据

#### Scenario: 显式 early-concat baseline 不被覆盖
- **WHEN** 用户加载显式 early-concat、legacy fusion 或模块化 `early_concat_gru` 配置
- **THEN** 系统 MUST 保持该配置声明的模型类型和 representation core
- **AND** 系统 MUST 不将其静默改写为 `cls_token_transformer_fusion`

#### Scenario: 已退役方法不被默认配置保留
- **WHEN** 用户查找默认或高级 fusion no-KD 推荐入口
- **THEN** 项目 MUST 不再提供 CRAF、MARF、G2D 或其 ablation 配置作为推荐入口

### Requirement: CLS-token Transformer 配置复用 fusion 数据字段
CLS-token Transformer fusion 配置 MUST 复用现有 fusion 数据字段和模态启用语义。启用 GPS、LiDAR 或 mmWave 时，配置 MUST 使用与其它 fusion 配置一致的数据字段、归一化和输入准备逻辑。

#### Scenario: 启用 GPS
- **WHEN** CLS-token Transformer fusion 配置的 `modalities` 包含 `gps`
- **THEN** 配置 MUST 设置 `data.dataset.use_gps: true`
- **AND** 配置 MUST 设置 `gps_feature_mode: relative_polar`
- **AND** `gps_input_size` MUST 为 3

#### Scenario: 启用 LiDAR
- **WHEN** CLS-token Transformer fusion 配置的 `modalities` 包含 `lidar`
- **THEN** 配置 MUST 设置 `data.dataset.use_lidar: true`
- **AND** 配置 MUST 沿用 LiDAR BEV 默认字段、缓存和内存有界归一化语义
- **AND** 模型 `lidar_channels` MUST 与 LiDAR BEV 输入通道一致

#### Scenario: 启用 mmWave
- **WHEN** CLS-token Transformer fusion 配置的 `modalities` 包含 `mmwave`
- **THEN** 配置 MUST 设置 `data.dataset.use_mmwave: true`
- **AND** 配置 MUST 设置 `mmwave_normalize: true`
- **AND** `mmwave_input_size` MUST 为 64

### Requirement: CLS-token Transformer fusion 组件注册
项目 MUST 通过现有组件注册表暴露 CLS-token Transformer fusion 模型。新增模型 MUST 能通过 `MODELS` 注册表构建，并 MUST 复用现有 fusion 训练、验证和评估入口。

#### Scenario: 按名称构建 CLS-token Transformer fusion
- **WHEN** 配置指定 `type: cls_token_transformer_fusion`
- **THEN** 系统 MUST 通过 `MODELS` 注册表返回 CLS-token Transformer fusion 模型实例
- **AND** 构建参数 MUST 来自配置字段
- **AND** 模型 MUST 支持现有 fusion forward 输入键

#### Scenario: 注册错误可诊断
- **WHEN** 用户引用不存在或拼写错误的 CLS-token Transformer fusion 注册名
- **THEN** 系统 MUST 使用现有 registry 错误风格抛出异常
- **AND** 错误信息 MUST 包含请求名称和可用模型注册名

### Requirement: 默认组件导入包含 CLS-token Transformer fusion
默认组件导入流程 MUST 注册 CLS-token Transformer fusion 内置模型，同时保持 registry 本身轻量可导入。导入 `kd_sensing.registries` MUST 不急切导入 dataset、trainer、checkpoint 或重依赖运行模块。

#### Scenario: 构建流程导入默认组件
- **WHEN** 构建流程调用 `import_default_components()` 后再查询 `MODELS`
- **THEN** `MODELS` 注册表 MUST 包含 `cls_token_transformer_fusion`
- **AND** 系统 MUST 能通过配置构建该模型

#### Scenario: 轻量导入 registry
- **WHEN** 开发者仅执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不 eager import CLS-token Transformer fusion 模型依赖

#### Scenario: 内置组件列表可发现
- **WHEN** 开发者按扩展文档触发默认模型模块导入后查看 `MODELS.list()`
- **THEN** 输出 MUST 包含 `cls_token_transformer_fusion`
- **AND** 输出 MUST 继续包含现有 canonical fusion 模型注册名
