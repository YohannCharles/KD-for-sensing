## MODIFIED Requirements

### Requirement: Fusion 模态选择配置
Fusion teacher 和 fusion student MUST 支持通过 `modalities` 配置选择参与融合的模态。`modalities` MUST 是 `image`、`radar`、`gps`、`lidar`、`mmwave`、`csi` 的非空列表；默认值 MUST 保持既有 image+radar 行为。

#### Scenario: 默认 fusion 模态
- **WHEN** 用户构建 fusion 模型且未显式配置 `modalities`
- **THEN** 系统 MUST 使用 `["image", "radar"]`
- **AND** 系统 MUST 保持旧 image+radar 配置的模型输入和输出行为兼容

#### Scenario: 配置全部模态
- **WHEN** 用户配置 `modalities: ["image", "radar", "gps", "lidar", "mmwave", "csi"]`
- **THEN** fusion 模型 MUST 创建 image、radar、gps、lidar、mmWave 和 CSI 六个分支
- **AND** fusion projection 的输入维度 MUST 与六个分支输出拼接维度一致

#### Scenario: 配置任意双模态
- **WHEN** 用户配置 `modalities` 为 `["image", "csi"]`、`["radar", "csi"]`、`["mmwave", "csi"]` 或其它合法双模态组合
- **THEN** fusion 模型 MUST 只创建被启用模态的分支
- **AND** forward MUST 只要求被启用模态对应的输入张量

#### Scenario: 配置单模态 fusion
- **WHEN** 用户配置 `modalities` 为 `["image"]`、`["radar"]`、`["gps"]`、`["lidar"]`、`["mmwave"]` 或 `["csi"]`
- **THEN** fusion 模型 MUST 能构建并运行
- **AND** fusion projection MUST 只接收该单模态分支输出

### Requirement: Fusion 模态配置校验
系统 MUST 对 fusion `modalities` 做显式校验。空列表、重复模态或未知模态 MUST 在模型构建时抛出清晰错误。

#### Scenario: 空模态列表
- **WHEN** 用户配置 `modalities: []`
- **THEN** 系统 MUST 拒绝构建 fusion 模型
- **AND** 错误信息 MUST 指出至少需要一个模态

#### Scenario: 未知模态
- **WHEN** 用户配置 `modalities` 包含 `image`、`radar`、`gps`、`lidar`、`mmwave`、`csi` 之外的名称
- **THEN** 系统 MUST 拒绝构建 fusion 模型
- **AND** 错误信息 MUST 包含非法模态名称

#### Scenario: 重复模态
- **WHEN** 用户配置 `modalities` 包含重复项
- **THEN** 系统 MUST 拒绝构建 fusion 模型
- **AND** 错误信息 MUST 指出模态不能重复

## ADDED Requirements

### Requirement: Fusion 输入准备支持 CSI
训练、验证和评估流程在 `experiment.task: fusion` 下 MUST 能根据配置的 `modalities` 准备 CSI 输入。未启用 CSI 时，batch 准备和模型 forward MUST 不要求 `csi` 或 `csi_batch`。

#### Scenario: fusion 启用 CSI 和 GPS
- **WHEN** fusion 配置的 `modalities` 为 `["gps", "csi"]`
- **THEN** batch 准备 MUST 构造 `gps_batch` 和 `csi_batch`
- **AND** batch 准备 MUST 不要求 image、radar、LiDAR 或 mmWave 字段

#### Scenario: fusion 启用全部六模态
- **WHEN** fusion 配置的 `modalities` 为 `["image", "radar", "gps", "lidar", "mmwave", "csi"]`
- **THEN** batch 准备 MUST 构造六个模态输入
- **AND** 六个输入的 batch 和 sequence 维度 MUST 对齐

### Requirement: Modular fusion 使用 CSI encoder 输出
`modular_sequence` fusion 模型 MUST 能将 CSI encoder 的 `[B, T, D]` 输出与其它模态 encoder 输出对齐，并通过既有 projector 和 representation core 处理。

#### Scenario: modular_sequence 融合 CSI 与 mmWave
- **WHEN** 配置 `modalities: ["mmwave", "csi"]`
- **THEN** 模型 MUST 分别调用 mmWave encoder 和 CSI encoder
- **AND** 两个 projected feature MUST 在 batch、time 和 `d_model` 维度上兼容
