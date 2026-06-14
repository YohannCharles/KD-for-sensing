## ADDED Requirements

### Requirement: Scenario 23 descriptor for AMR-Net-gps-image
DeepSense6G 场景选择 MUST 支持 AMR-Net-gps-image 复现所需的 Scenario 23 描述符。描述符 MUST 包含 `scene_id: 23`、`scene_slug: scene23`、别名、默认数据根目录、legacy 数据根目录、默认 train/test CSV 名和 scene-scoped 输出目录规则。

#### Scenario: 通过整数选择 Scenario 23
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 23`
- **THEN** 系统 MUST 将场景解析为 Scenario 23
- **AND** 默认数据根目录 MUST 指向 `dataset/DeepSense6G/scenario23`
- **AND** runtime metadata MUST 记录 `scene_id: 23` 和 `scene_slug: scene23`

#### Scenario: 通过别名选择 Scenario 23
- **WHEN** 用户设置 `data.dataset.scene` 为 `scene23` 或 `scenario23`
- **THEN** 系统 MUST 解析到 Scenario 23
- **AND** 配置中的大小写差异 MUST 不影响解析结果

#### Scenario: Scenario 23 默认 CSV 名可审计
- **WHEN** 用户选择 Scenario 23 且未显式设置 train/test CSV 名
- **THEN** 系统 MUST 使用 Scenario 23 描述符声明的默认 CSV 名
- **AND** source audit 或 runtime metadata MUST 记录这些 CSV 名来自 AMR-Net-gps-image 复现协议

### Requirement: Scenario 23 output isolation
Scenario 23 的训练、评估、诊断和 paper reproduction 输出 MUST 与其它 DeepSense6G 场景隔离。默认输出目录 MUST 使用 `outputs/scene23/` 或 paper-specific analysis root，不得写入其它 scene 目录。

#### Scenario: 单场景输出隔离
- **WHEN** AMR-Net-gps-image runner 在 Scenario 23 上训练或评估
- **THEN** 默认 run output MUST 位于 `outputs/scene23/<run_name>/` 或 `outputs/analysis/ieee_11282996_gps_image/<run_id>/`
- **AND** report MUST 记录 scene slug 和 output root

#### Scenario: 显式 data_root 不改写
- **WHEN** 用户为 Scenario 23 显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 使用用户提供的路径构建 dataset
- **AND** runtime metadata MUST 仍记录规范场景为 Scenario 23
