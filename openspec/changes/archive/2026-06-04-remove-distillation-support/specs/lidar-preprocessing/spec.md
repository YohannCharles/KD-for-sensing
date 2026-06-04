## ADDED Requirements

### Requirement: LiDAR 预处理示例不依赖 KD 命名
LiDAR preprocessing、cache 和 normalization 文档或示例配置 MUST 使用 supervised/strong/lightweight 命名引用训练入口，不得推荐 `*_no_kd`、`logits_kd` 或 `rkd` 路径。

#### Scenario: LiDAR preprocessing 后续训练提示
- **WHEN** 文档或 CLI 输出提示用户运行 LiDAR 训练
- **THEN** 推荐路径 MUST 使用 LiDAR supervised、strong 或 lightweight 配置
- **AND** 推荐路径 MUST 不包含 KD token

