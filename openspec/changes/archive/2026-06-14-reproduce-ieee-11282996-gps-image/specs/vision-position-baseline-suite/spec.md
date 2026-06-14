## ADDED Requirements

### Requirement: AMR-Net-gps-image preset
Vision-Position baseline suite MUST 提供 AMR-Net-gps-image paper preset。该 preset MUST 使用 DeepSense6G dataset、Scenario 23、image 与 GPS 输入、paper-audited target source、paper-audited GPS feature mode 和 64-beam 或 source-audit 声明的 beam classifier label space。

#### Scenario: paper preset 可加载
- **WHEN** 用户加载 AMR-Net-gps-image paper preset
- **THEN** 配置加载 MUST 成功
- **AND** 启用模态 MUST 标准化为 `["image", "gps"]`
- **AND** dataset scene MUST 解析为 Scenario 23 或 source audit 声明的论文场景

#### Scenario: paper preset 输出 logits
- **WHEN** paper preset 模型接收 synthetic image 与 GPS batch
- **THEN** forward MUST 成功
- **AND** 输出 logits MUST 能被现有 training/evaluation runtime 解释为 beam classifier 输出

### Requirement: AMR-Net-gps-image model metadata
AMR-Net-gps-image preset 的每个 model group MUST 输出可审计 metadata。metadata MUST 包含 model name、model group、enabled modalities、image encoder、GPS feature mode、GPS normalizer provenance、fusion type、num beams、target source、metric profile、claim status 和 whether paper-reported row。

#### Scenario: metadata 区分 paper row 和 local control
- **WHEN** runner 构建 Image+GPS fusion model group
- **THEN** metadata MUST 记录该 group 是否来自论文报告行
- **AND** 未被论文报告的辅助对照 MUST 标记为 `local_control`

#### Scenario: metadata 标记不使用 LiDAR
- **WHEN** AMR-Net-gps-image preset 完成构建、训练或评估
- **THEN** metadata MUST 记录 `enabled_modalities: ["image", "gps"]`
- **AND** metadata MUST 记录 `uses_lidar: false`

### Requirement: AMR-Net-gps-image smoke and regression tests
Vision-Position baseline suite MUST 提供不依赖真实 DeepSense6G 数据的 AMR-Net-gps-image smoke 测试。测试 MUST 覆盖 config loading、Scenario 23 descriptor、LiDAR 禁用、synthetic forward、Top-k metric aggregation 和 report schema。

#### Scenario: synthetic smoke 不读取真实数据
- **WHEN** 测试运行 AMR-Net-gps-image mock 或 synthetic smoke
- **THEN** 测试 MUST 不读取真实 `dataset/` 文件
- **AND** 产物 MUST 标记 `mock_data: true`

#### Scenario: LiDAR override 被测试拒绝
- **WHEN** 测试对 AMR-Net-gps-image preset 注入 `lidar` modality 或 `use_lidar: true`
- **THEN** 配置或 runner MUST 抛出清晰错误
- **AND** 错误信息 MUST 指向 GPS+Image-only 复现边界
