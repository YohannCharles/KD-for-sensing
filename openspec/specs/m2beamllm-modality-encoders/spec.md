# m2beamllm-modality-encoders Specification

## Purpose
TBD - created by archiving change add-m2beamllm-encoders. Update Purpose after archive.
## Requirements
### Requirement: M2BeamLLM encoder 作为新增可选入口
系统 MUST 提供 M2BeamLLM 风格的 image、radar、GPS、LiDAR GRU 前编码器作为新增可选入口，并 MUST 不覆盖现有默认 `image_*`、`radar_*`、`gps_*`、`lidar_*` 和 `fusion_*` 注册名的行为。

#### Scenario: 默认模型不变
- **WHEN** 用户加载现有单模态或 fusion canonical 配置且没有显式启用 M2BeamLLM encoder
- **THEN** 系统 MUST 继续构建现有注册名对应的模型
- **AND** 模型的 GRU 前后行为 MUST 与本变更前保持兼容

#### Scenario: 显式启用 M2BeamLLM encoder
- **WHEN** 用户在配置中选择 M2BeamLLM encoder 注册名或 `encoder_profile: m2beamllm`
- **THEN** 系统 MUST 使用 M2BeamLLM 风格编码器替换启用模态的 GRU 前 feature extraction
- **AND** GRU 及之后模块 MUST 继续使用项目现有时序预测结构

### Requirement: M2BeamLLM encoder 输出契约
M2BeamLLM encoder MUST 对每个启用模态输出形状为 `[B, T, feature_size]` 的帧级特征，并 MUST 与现有 LayerNorm、GRU、classifier 和 KD distiller 输入契约兼容。

#### Scenario: 单模态 forward 输出
- **WHEN** 任一 M2BeamLLM 单模态模型接收合法输入
- **THEN** 模型 MUST 返回 `(pred, features, output_features)`
- **AND** `pred` 的形状 MUST 为 `[B, T, num_classes]`
- **AND** `features` 的形状 MUST 为 `[B, T, feature_size]`
- **AND** `output_features` 的 batch 和 sequence 维度 MUST 与输入一致

#### Scenario: KD 兼容
- **WHEN** M2BeamLLM teacher/student 用于 logits KD 或 RKD
- **THEN** distiller MUST 能接收其 logits、input features 和 output features
- **AND** 训练循环 MUST 不需要为 M2BeamLLM encoder 新增专用 KD 分支

### Requirement: Image M2BeamLLM encoder
系统 MUST 提供 image M2BeamLLM encoder。该 encoder MUST 将每个时隙图像处理为 224x224 RGB 输入，使用 ImageNet mean/std 标准化，通过去掉分类头的 ResNet-18 提取 512 维特征，并通过 FC/ReLU 投影到 `feature_size`。

#### Scenario: image encoder 输入输出
- **WHEN** image M2BeamLLM encoder 接收 `[B, T, C, H, W]` 图像输入
- **THEN** 系统 MUST 将输入适配为 RGB 224x224 图像
- **AND** 系统 MUST 按 ImageNet mean/std 标准化
- **AND** encoder 输出 MUST 为 `[B, T, feature_size]`

#### Scenario: image 单通道兼容
- **WHEN** 输入图像为单通道 motion mask 或灰度张量
- **THEN** 系统 MUST 通过显式配置将其转换或重复到 ResNet-18 期望的 RGB 通道
- **AND** 系统 MUST 不静默改变默认 image 模型的单通道处理方式

### Requirement: Radar M2BeamLLM encoder
系统 MUST 提供 radar M2BeamLLM encoder。该 encoder MUST 支持从 RA map 编码到 `feature_size`，并 SHOULD 在 raw radar tensor 可用时支持 Range FFT、DC removal、Angle FFT 生成 RA map 的路径；选择 raw FFT 路径时必须由显式配置启用。

#### Scenario: 使用现有 RA map 输入
- **WHEN** 配置选择 `radar_input_mode: ra_map`
- **THEN** radar M2BeamLLM encoder MUST 接收现有 batch 准备出的 `[B, T, C, H, W]` 雷达 map 输入
- **AND** encoder MUST 通过 CNN、pooling 和 FC/ReLU 投影输出 `[B, T, feature_size]`

#### Scenario: raw FFT 输入缺失
- **WHEN** 配置选择 `radar_input_mode: raw_fft` 但 batch 没有 raw radar tensor 字段
- **THEN** 系统 MUST 拒绝 forward 或 batch 准备
- **AND** 错误信息 MUST 指出 raw FFT 路径需要 raw radar 输入

#### Scenario: raw FFT 生成 RA map
- **WHEN** 配置选择 `radar_input_mode: raw_fft` 且 batch 提供 raw radar tensor
- **THEN** 系统 MUST 对每个时隙执行 Range/Angle FFT 相关处理以生成 RA map
- **AND** encoder MUST 只把生成的 RA map 输入后续 CNN 编码路径

### Requirement: LiDAR M2BeamLLM encoder
系统 MUST 提供 LiDAR M2BeamLLM encoder。该 encoder MUST 支持 M2BeamLLM 风格的单通道 256x256 histogram 输入，histogram 每个网格的点计数 MUST 裁剪到 5 并归一化到 `[0, 1]`，再通过改造 ResNet-18 和线性层输出 `feature_size`。

#### Scenario: LiDAR histogram 输入
- **WHEN** 配置选择 M2BeamLLM LiDAR histogram 路径
- **THEN** dataset 或预处理流程 MUST 产生 `[B, T, 1, 256, 256]` LiDAR histogram
- **AND** 点计数 MUST 裁剪到 5 并除以 5
- **AND** LiDAR encoder 输出 MUST 为 `[B, T, feature_size]`

#### Scenario: LiDAR BEV 输入显式适配
- **WHEN** 用户希望用现有 LiDAR BEV cache 作为 M2BeamLLM LiDAR encoder 输入
- **THEN** 配置 MUST 显式设置 `lidar_channels` 和输入尺寸适配策略
- **AND** 系统 MUST 不把 3 通道 BEV 静默当作论文的单通道 histogram

### Requirement: GPS M2BeamLLM encoder
系统 MUST 提供 GPS M2BeamLLM encoder。该 encoder MUST 使用训练集 fit 的 min-max 统计对二维 GPS 坐标归一化，并通过 `Linear(2,32) -> LayerNorm -> GELU -> Linear(32,64) -> LayerNorm -> GELU -> Linear(64,feature_size)` 或等价可配置 MLP 输出帧级特征。

#### Scenario: GPS min-max fit 与复用
- **WHEN** 构建训练 dataloader 且启用 M2BeamLLM GPS encoder
- **THEN** 系统 MUST 只在 train split 上 fit GPS min-max 统计
- **AND** 系统 MUST 将该统计作为 artifact 或 dataset 状态传递给 test split

#### Scenario: GPS encoder 输入输出
- **WHEN** GPS M2BeamLLM encoder 接收 `[B, T, 2]` 归一化前坐标输入
- **THEN** 系统 MUST 应用训练集 min-max 归一化
- **AND** encoder 输出 MUST 为 `[B, T, feature_size]`

#### Scenario: GPS 旧输入保持兼容
- **WHEN** 用户运行现有 GPS-Rel-Polar 配置
- **THEN** 系统 MUST 继续使用 `[B, T, 3]` GPS-Rel-Polar 输入和现有 GPS 模型
- **AND** 系统 MUST 不要求旧配置提供 M2BeamLLM GPS min-max artifact

### Requirement: Fusion 使用 M2BeamLLM encoder
fusion 模型 MUST 能在显式启用 M2BeamLLM encoder 时，对 image、radar、GPS、LiDAR 分支使用对应 M2BeamLLM encoder，并 MUST 保持 fusion GRU 及之后模块不变。

#### Scenario: fusion 部分模态启用 M2BeamLLM encoder
- **WHEN** fusion 配置的 `modalities` 为 `["image", "radar", "gps", "lidar", "mmwave"]` 且启用 M2BeamLLM encoder profile
- **THEN** image、radar、GPS、LiDAR 分支 MUST 使用 M2BeamLLM encoder
- **AND** mmWave 分支 MUST 继续使用现有 mmWave encoder
- **AND** fusion 后的 GRU、attention 和 classifier MUST 保持现有结构

#### Scenario: fusion 未启用 mmWave
- **WHEN** fusion 配置不包含 `mmwave`
- **THEN** M2BeamLLM encoder profile MUST 只应用于实际启用的非 mmWave 模态
- **AND** forward MUST 不要求 mmWave 输入

### Requirement: mmWave 排除规则
M2BeamLLM encoder profile MUST 不改变 mmWave-only 或 fusion mmWave 分支的输入处理、feature extractor、scaler 和配置默认值。

#### Scenario: mmWave-only 配置
- **WHEN** 用户运行 `experiment.task: mmwave`
- **THEN** 系统 MUST 继续构建现有 mmWave 模型
- **AND** 系统 MUST 不尝试构建 M2BeamLLM mmWave encoder

#### Scenario: fusion 中包含 mmWave
- **WHEN** fusion 配置同时包含 mmWave 和其它 M2BeamLLM encoder 模态
- **THEN** mmWave 分支 MUST 继续接收 `[B, T, 64]` receive-power 特征
- **AND** mmWave scaler 的 fit、保存和复用语义 MUST 保持不变

### Requirement: M2BeamLLM encoder 配置与测试
项目 MUST 提供可加载的 M2BeamLLM encoder 示例配置和自动化测试，覆盖模型注册、shape contract、默认配置不变、mmWave 排除和 GPS scaler 复用。

#### Scenario: 示例配置可加载
- **WHEN** 用户加载 M2BeamLLM encoder 单模态或 fusion 示例配置
- **THEN** 配置解析 MUST 成功
- **AND** 模型注册表 MUST 能构建对应模型

#### Scenario: GRU 后结构回归测试
- **WHEN** 测试构建旧模型和 M2BeamLLM encoder 变体
- **THEN** 测试 MUST 断言新变体的 GRU 参数来自配置
- **AND** 测试 MUST 断言新变体仍返回统一 forward 契约

