## ADDED Requirements

### Requirement: Dataset layout audit
Dataset layout descriptor MUST 支持 dataset audit 读取 canonical root、legacy root、scene/condition alias 和 required subdirectories。Audit MUST 使用该 descriptor，而不是在多个脚本中重复硬编码路径。

#### Scenario: audit 解析 canonical layout
- **WHEN** audit 检查 DeepSense6G Scenario 31 或 MMW sunny condition
- **THEN** 它 MUST 通过 dataset layout descriptor 解析 canonical data root 和 expected subdirectories
- **AND** 报告 MUST 标记路径存在、缺失或显式 override

#### Scenario: legacy layout 标记为兼容输入
- **WHEN** audit 发现 legacy `dataset/scenario31` 或其它旧兼容根
- **THEN** 报告 MUST 标记为 legacy-compatible input
- **AND** audit MUST 不自动迁移、复制、删除或重命名该目录

#### Scenario: cache 与数据目录区分
- **WHEN** audit 发现 image/LiDAR/radar/cache-like artifacts
- **THEN** 报告 MUST 区分 raw data、prepared data、runtime cache 和 output artifact
- **AND** 可再生成 cache SHOULD 建议位于 `outputs/cache/`，但 audit MUST 不移动文件
