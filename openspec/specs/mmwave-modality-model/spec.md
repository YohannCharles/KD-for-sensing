# mmwave-modality-model Specification

## Purpose
定义 mmWave power vector 模型、feature extractor、scaler 和输入语义。
## Requirements
### Requirement: MmWaveFeatureExtractor 结构
系统 MUST 提供 `MmWaveFeatureExtractor`，用于从 mmWave 64 维 receive-power 特征序列中提取每个时隙的固定长度 embedding。该 feature extractor MUST 接收形状为 `(batch, sequence, 64)` 的 mmWave 张量，并输出 `(batch, sequence, feature_size)`。该类 MAY 通过 `kd_sensing.models.mmwave` 或 `kd_sensing.models` 窄导入暴露，但 MUST NOT 作为 current `MODELS` 注册名暴露。

#### Scenario: mmWave feature extractor 前向输出
- **WHEN** `MmWaveFeatureExtractor` 接收形状为 `(B, T, 64)` 的 mmWave 输入
- **THEN** 输出 MUST 为形状 `(B, T, feature_size)` 的特征张量
- **AND** 输出 feature 维 MUST 等于构造参数 `n_feature` 或 `feature_size`

#### Scenario: mmWave feature extractor 不作为完整模型注册
- **WHEN** 开发者查看 current `MODELS.list()`
- **THEN** 输出 MUST NOT 包含 `mmwave_feature_extractor`
- **AND** 需要配置构建 mmWave encoder 时 MUST 使用 `ENCODERS` 中的 `mmwave_mlp`

### Requirement: mmWave 模型参数校验
mmWave teacher 和 mmWave student MUST 校验 `gru_params` 和输入维度配置。`gru_params` MUST 包含 `[input_size, hidden_size, num_layers]`，且 `input_size` MUST 等于 `feature_size`；`mmwave_input_size` MUST 等于 dataset 输出的 64 维特征。

#### Scenario: 非法 gru_params 长度
- **WHEN** 用户用长度不是 3 的 `gru_params` 构建 `mmwave_teacher` 或 `mmwave_student`
- **THEN** 系统 MUST 抛出配置错误

#### Scenario: GRU 输入维度不匹配
- **WHEN** 用户配置的 `gru_params[0]` 不等于 `feature_size`
- **THEN** 系统 MUST 抛出包含实际维度和期望维度的错误

#### Scenario: mmWave 输入维度不匹配
- **WHEN** mmWave 模型收到最后一维不是 64 或不等于 `mmwave_input_size` 的输入张量
- **THEN** 系统 MUST 抛出包含实际输入维度和期望维度的错误

### Requirement: mmWave-only 输入准备
系统 MUST 提供 mmWave-only 输入准备路径，从 batch 中读取 `mmwave`，按现有预测窗口规则补齐未来占位时隙，并将结果传给 mmWave 模型。

#### Scenario: 准备 mmWave-only batch
- **WHEN** 训练、验证或评估流程处理 `experiment.task: mmwave`
- **THEN** 系统 MUST 使用 batch 中的 `mmwave` 构造 mmWave 输入
- **AND** 系统 MUST 不要求图像、雷达、GPS 或 LiDAR 输入参与模型 forward

#### Scenario: mmWave 预测窗口对齐
- **WHEN** `seq_length` 为 8 且 `num_pred` 为 3
- **THEN** mmWave-only 输入 MUST 包含最近 8 个 mmWave 历史时隙和 2 个未来 zero padding 时隙
- **AND** 验证和损失计算 MUST 使用最后 `num_pred` 个输出时隙与 `[t+1, t+2, t+3]` 标签对齐
- **AND** 输出时隙对齐 MUST 不包含历史窗口最后一个 beam

### Requirement: mmWave KD 入口已移除
mmWave-only 训练 MUST 不再支持 logits KD、RKD 或 distiller 运行时。旧 mmWave KD 配置路径 MUST 在配置解析阶段失败，并引导用户使用 `configs/mmwave/strong.yaml`、`configs/mmwave/lightweight.yaml` 或 `configs/mmwave/supervised.yaml`。

#### Scenario: mmWave logits KD 被拒绝
- **WHEN** 用户运行旧 mmWave-only logits KD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不构建 frozen mmWave teacher 或 distiller

#### Scenario: mmWave RKD 被拒绝
- **WHEN** 用户运行旧 mmWave-only RKD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不计算关系蒸馏损失

### Requirement: mmWave canonical 配置使用 modular_sequence
mmWave strong、lightweight 和 supervised canonical 配置 MUST 使用 `modular_sequence`、`mmwave_mlp` encoder、projector、`single_gru` representation core 和 `beam_head`，而不是旧 mmWave whole-model 注册名。

#### Scenario: 构建 mmWave strong/supervised 配置
- **WHEN** 用户加载 `configs/mmwave/strong.yaml` 或 `configs/mmwave/supervised.yaml`
- **THEN** 最终配置的 `model.primary.type` MUST 为 `modular_sequence`
- **AND** `model.primary.encoders.mmwave.type` MUST 为 `mmwave_mlp`
- **AND** mmWave-only task runtime MUST 继续只准备 mmWave 输入

#### Scenario: 构建 mmWave lightweight 配置
- **WHEN** 用户加载 `configs/mmwave/lightweight.yaml`
- **THEN** 系统 MUST 构建 `modular_sequence` mmWave-only 模型
- **AND** lightweight 差异 MUST 通过配置参数表达

### Requirement: mmWave legacy model names are removed
mmWave legacy whole-model 注册名和 feature extractor `MODELS` 注册名 MUST 被 removed guard 拒绝。

#### Scenario: 请求 mmWave legacy 注册名
- **WHEN** 用户请求 `mmwave_teacher`、`mmwave_student`、`mmwave_strong`、`mmwave_lightweight` 或 `mmwave_feature_extractor`
- **THEN** registry MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 建议使用 `modular_sequence + mmwave_mlp + single_gru`

### Requirement: mmWave 注册错误可诊断
mmWave 相关注册错误 MUST 使用现有注册表错误风格，并在未知名称、重复名称或缺失必需参数时提供清晰错误信息。

#### Scenario: 请求未知 mmWave 组件
- **WHEN** 配置中引用未注册的 mmWave 模型或预处理器名称
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含请求的组件名称、注册表名称和可用组件列表

#### Scenario: mmWave 构建参数缺失
- **WHEN** 配置中引用已注册 mmWave 组件但缺少必需构造参数
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含缺失字段或原始构建错误

### Requirement: mmWave 配置驱动实验
项目 MUST 支持通过配置文件启动 mmWave-only 训练和评估。mmWave-only 配置 MUST 使用 `experiment.task: mmwave`、当前 mmWave dataset contract 和 `model.primary`，并通过统一训练、验证、评估、loss、optimizer、scheduler、checkpoint 和指标流程运行。

#### Scenario: 使用配置启动 mmWave-only 训练
- **WHEN** 用户通过当前 CLI 传入 mmWave-only 训练配置
- **THEN** 系统 MUST 构建包含 mmWave 输入的 dataset、配置指定的 mmWave primary model、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 不要求图像、雷达、GPS、LiDAR、teacher checkpoint 或 distiller
- **AND** mmWave 输入 MUST 使用 `[B, T, 64]` 的 dB receive-power 特征序列

#### Scenario: 使用配置启动 mmWave-only 评估
- **WHEN** 用户通过当前 CLI 传入 mmWave-only 评估配置和 mmWave 模型权重
- **THEN** 系统 MUST 构建配置指定的 mmWave 模型并只使用 mmWave 输入完成评估
- **AND** 系统 MUST 保存 Top-K、DBA 和 loss 指标
- **AND** 评估流程 MUST 复用训练时保存的 mmWave scaler

### Requirement: mmWave fusion 配置驱动实验
项目 MUST 支持通过 fusion `modalities` 配置启用 mmWave。包含 mmWave 的 fusion 配置 MUST 复用统一 fusion 训练和评估流程，并 MUST 构建单个 fusion primary model。

#### Scenario: 使用配置启动五模态 fusion 训练
- **WHEN** 用户通过训练入口传入 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]` 的 fusion 配置
- **THEN** 系统 MUST 构建五个模态输入所需的 dataset 字段和 fusion primary model
- **AND** 系统 MUST 在 batch 准备阶段构造 image、radar、gps、lidar 和 mmWave 输入

#### Scenario: 使用配置启动 mmWave 参与的双模态 fusion 训练
- **WHEN** 用户通过训练入口传入包含 `mmwave` 的任意合法双模态 fusion 配置
- **THEN** 系统 MUST 只准备 `modalities` 中列出的模态输入
- **AND** 未启用的模态字段 MUST 不影响训练启动

### Requirement: mmWave 默认实验配置
项目 MUST 提供 mmWave-only strong、lightweight、supervised 配置和包含 mmWave 的 canonical fusion 配置。所有默认 mmWave primary 配置 MUST 使用 `mmwave_input_size: 64`、`mmwave_normalize: true` 和 `gru_params: [64, 64, 1]`。

#### Scenario: mmWave 默认配置可构建
- **WHEN** 开发者加载 `configs/mmwave/*.yaml`
- **THEN** 系统 MUST 能构建对应 dataset、model、loss、optimizer 和 scheduler
- **AND** primary 配置的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** primary 配置的 `mmwave_input_size` MUST 为 64

#### Scenario: mmWave KD 配置被拒绝
- **WHEN** 用户运行 `configs/mmwave/logits_kd.yaml` 或 `configs/mmwave/rkd.yaml`
- **THEN** 配置加载 MUST 失败
- **AND** 系统 MUST 不解析 teacher checkpoint
