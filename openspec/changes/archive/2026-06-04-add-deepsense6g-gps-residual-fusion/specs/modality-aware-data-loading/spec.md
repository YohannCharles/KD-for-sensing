## ADDED Requirements

### Requirement: DeepSense6G residual modality discovery
Residual manifest builder MUST automatically discover DeepSense6G optional modality resources and precomputed features without requiring every modality to exist.

#### Scenario: 自动发现预计算 feature
- **WHEN** manifest builder 扫描 DeepSense6G resources
- **THEN** 系统 MUST 优先识别 `.npy`、`.npz`、`.pt`、`.csv` 和 `.parquet` 形式的 precomputed feature
- **AND** 系统 MUST 将可用 feature path 写入 manifest 对应列

#### Scenario: 自动发现 sensor path
- **WHEN** image、LiDAR 或 radar path 在原始 CSV 或场景目录中可发现
- **THEN** 系统 MUST 将对应 path 写入 manifest
- **AND** path 不可用时对应 manifest 列 MUST 为空或标记不可用

#### Scenario: 缺失模态不阻断 GPS baseline
- **WHEN** 某 optional modality 在全部或部分场景缺失
- **THEN** manifest builder MUST 继续完成
- **AND** residual training MUST 仍能运行 `gps_prior_only` 与 `gps_context_only_residual`
- **AND** skipped modality ablation MUST 在 summary 中记录 `skipped_reason`

### Requirement: Residual dataset 按启用模态读取
Residual fusion Dataset/DataLoader MUST 根据 manifest 与 ablation 启用模态读取数据，不得读取未启用或不可用的 optional modality。

#### Scenario: GPS context only 不读取 sensor 文件
- **WHEN** ablation 为 `gps_context_only_residual`
- **THEN** Dataset MUST 只读取 GPS context、prior logits/stats 和标签所需字段
- **AND** Dataset MUST 不读取 image、LiDAR 或 radar 文件

#### Scenario: array modality shape 校验
- **WHEN** LiDAR 或 radar array feature 被启用
- **THEN** Dataset MUST 校验 array shape 是否可被选定 encoder 处理
- **AND** shape 不一致且无法使用预处理 feature 时 MUST 报告清晰错误或跳过该 ablation
