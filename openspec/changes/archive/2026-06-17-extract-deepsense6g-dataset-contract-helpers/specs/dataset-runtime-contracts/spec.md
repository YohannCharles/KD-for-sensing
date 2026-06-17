## ADDED Requirements

### Requirement: DeepSense6G dataset contract helper 拆分
DeepSense6G runtime dataset SHALL 将配置 normalization、column validation、target source normalization、GPS feature contract 和 cache path resolution 等低风险契约逻辑拆分到轻量 helper 模块。`DeepSense6GDataset` MUST 继续保持现有 flat sample、target schema、metadata 和资源读取语义兼容。

#### Scenario: helper 不读取真实资源
- **WHEN** 测试或代码调用 DeepSense6G contract helper 解析 GPS feature mode、beam target source、required columns 或 cache path
- **THEN** helper MUST 不读取 image、LiDAR、CSI、GPS 文件或 beam label 文件
- **AND** helper MUST 不导入训练循环、模型 registry 或 heavy runtime module

#### Scenario: dataset 输出保持兼容
- **WHEN** 使用相同 synthetic dataframe、配置和 mock resource paths 构建 DeepSense6G dataset
- **THEN** helper 拆分前后的 sample keys、target fields、sample id、split metadata 和 enabled modality behavior MUST 保持兼容
- **AND** target labels MUST 不因 helper 拆分而改变

### Requirement: DeepSense6G GPS 和 target source contract 可测试
DeepSense6G GPS feature mode、scene calibration、GPS angle offset、GPS BEV XY source 和 beam target source MUST 有集中 helper 和 focused tests。错误信息 MUST 指向具体字段和支持值。

#### Scenario: unsupported GPS feature mode 被拒绝
- **WHEN** 配置声明未知 `gps_feature_mode`
- **THEN** helper MUST raise 清晰错误
- **AND** 错误信息 MUST 列出支持的 GPS feature mode

#### Scenario: current target source 保持 Table III 语义
- **WHEN** BeamBench-fair 或 Table III 风格配置声明 `beam_target_source=current`
- **THEN** helper MUST 保持 current beam target 语义
- **AND** `num_pred`、`seq_len` 和 target path 选择规则 MUST 与现有实现兼容
