## REMOVED Requirements

### Requirement: M2BeamLLM encoder 作为新增可选入口
**Reason**: M2BeamLLM 风格编码器实验效果不佳，继续提供该可选入口会增加维护成本并误导后续实验。
**Migration**: 使用现有默认 `image_*`、`radar_*`、`gps_*`、`lidar_*` 和 `fusion_*` 注册名；不要在配置中引用 `m2beamllm_*` 或 `encoder_profile: m2beamllm`。

#### Scenario: M2BeamLLM 注册名退役
- **WHEN** 用户配置 `m2beamllm_image_teacher`、`m2beamllm_radar_teacher`、`m2beamllm_gps_teacher`、`m2beamllm_lidar_teacher` 或对应 student 注册名
- **THEN** 系统 MUST 不再构建这些模型
- **AND** 注册表错误 MUST 指出该名称不可用

### Requirement: M2BeamLLM encoder 输出契约
**Reason**: 该输出契约只服务于已退役的 M2BeamLLM encoder 模型。
**Migration**: 继续使用现有标准模型的 `(pred, features, output_features)` 输出契约。

#### Scenario: 标准模型输出契约保留
- **WHEN** 用户运行标准单模态或 fusion 模型
- **THEN** 系统 MUST 继续返回现有 `(pred, features, output_features)` 契约
- **AND** 训练循环 MUST 不需要 M2BeamLLM 专用 KD 分支

### Requirement: Image M2BeamLLM encoder
**Reason**: image M2BeamLLM encoder 效果不佳且依赖额外 ResNet-18 适配逻辑。
**Migration**: 使用标准 image teacher/student 模型和现有 image feature extractor。

#### Scenario: image M2BeamLLM encoder 不可用
- **WHEN** 用户尝试导入或构建 image M2BeamLLM encoder
- **THEN** 系统 MUST 不提供该 encoder 类或注册入口

### Requirement: Radar M2BeamLLM encoder
**Reason**: radar M2BeamLLM encoder 及 raw FFT/RA map 适配路径不再作为正式实验入口维护。
**Migration**: 使用标准 radar teacher/student 模型和现有 radar feature extractor。

#### Scenario: radar M2BeamLLM encoder 不可用
- **WHEN** 用户尝试通过 M2BeamLLM radar 注册名或配置构建模型
- **THEN** 系统 MUST 不再构建该模型

### Requirement: LiDAR M2BeamLLM encoder
**Reason**: LiDAR M2BeamLLM histogram 编码路径与退役 encoder 绑定，继续保留会扩大数据预处理分支。
**Migration**: 使用标准 LiDAR BEV 路径和 `lidar_teacher/student` 注册名。

#### Scenario: LiDAR M2BeamLLM encoder 不可用
- **WHEN** 用户尝试使用 M2BeamLLM LiDAR encoder 或专用示例配置
- **THEN** 系统 MUST 不再提供对应模型入口

### Requirement: GPS M2BeamLLM encoder
**Reason**: GPS M2BeamLLM min-max MLP 路径与退役 encoder 绑定，继续保留会增加 scaler 语义维护成本。
**Migration**: 使用现有 GPS-Rel-Polar 输入和标准 GPS 模型。

#### Scenario: GPS M2BeamLLM encoder 不可用
- **WHEN** 用户尝试使用 M2BeamLLM GPS encoder 或 `gps_feature_mode: m2beamllm_minmax`
- **THEN** 系统 MUST 不再把该路径作为支持的正式 M2BeamLLM encoder 行为

### Requirement: Fusion 使用 M2BeamLLM encoder
**Reason**: fusion M2BeamLLM profile 会继续暴露已退役 encoder，并增加 fusion 分支复杂度。
**Migration**: 使用标准 `fusion_teacher/student`，不要设置 `encoder_profile: m2beamllm`。

#### Scenario: fusion M2BeamLLM profile 退役
- **WHEN** 用户在 fusion 配置中设置 `encoder_profile: m2beamllm`
- **THEN** 系统 MUST 不再切换 image、radar、GPS 或 LiDAR 分支到 M2BeamLLM encoder
- **AND** 系统 MUST 以可诊断错误拒绝该 profile 或因参数不受支持而失败

### Requirement: mmWave 排除规则
**Reason**: M2BeamLLM profile 已退役，不再需要为其定义 mmWave 排除规则。
**Migration**: mmWave-only 与 fusion mmWave 分支继续使用现有 mmWave 输入、feature extractor、scaler 和默认配置。

#### Scenario: mmWave 行为不变
- **WHEN** 用户运行 mmWave-only 或包含 mmWave 的标准 fusion 配置
- **THEN** 系统 MUST 继续使用现有 mmWave 分支
- **AND** 系统 MUST 不尝试构建 M2BeamLLM mmWave encoder

### Requirement: M2BeamLLM encoder 配置与测试
**Reason**: M2BeamLLM encoder 示例配置和正向测试只服务于已退役能力。
**Migration**: 删除 M2BeamLLM 示例配置和正向测试；保留默认模型构建回归测试，并可增加退役入口失败测试。

#### Scenario: 示例配置删除
- **WHEN** 用户查看项目内置配置
- **THEN** 系统 MUST 不再提供 `configs/m2beamllm/` 示例配置

#### Scenario: 退役入口测试
- **WHEN** 自动化测试覆盖模型注册和 fusion 构建
- **THEN** 测试 MUST 验证默认模型仍可构建
- **AND** 测试 MUST 不再要求 M2BeamLLM encoder 正向构建成功
