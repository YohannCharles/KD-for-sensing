## MODIFIED Requirements

### Requirement: 归档 metadata 与归一化工件关联
归档 checkpoint MUST 具备可机器读取的 metadata，用于记录源运行目录、配置 slug、模态、KD 模式、epoch、验证 Top-1 accuracy、源 checkpoint 路径、split 信息和训练归一化工件路径。启用 GPS、LiDAR 或 mmWave 归一化时，metadata MUST 能让评估入口复用训练时的 scaler 或 normalizer/stats。

#### Scenario: 写入归档 sidecar
- **WHEN** 系统将 checkpoint 复制到归档目录
- **THEN** 系统 MUST 写入同名或可关联的 JSON sidecar metadata
- **AND** metadata MUST 记录验证 Top-1 accuracy、源 `run_dir`、源 checkpoint、配置 slug 和启用模态

#### Scenario: 评估复用归一化工件
- **WHEN** 用户评估一个 registry checkpoint 且 metadata 记录了 GPS scaler、LiDAR normalizer/stats 或 mmWave scaler 路径
- **THEN** 评估入口 MUST 加载 metadata 中的归一化工件
- **AND** 评估入口 MUST 不为了重新 fit 归一化状态而扫描训练 split

#### Scenario: 评估缺少 mmWave scaler 工件
- **WHEN** 用户评估启用 mmWave 归一化的 checkpoint 且 metadata 没有记录可用 mmWave scaler 路径
- **THEN** 评估入口 MUST 抛出清晰错误
- **AND** 错误信息 MUST 提示提供 mmWave scaler 或使用带 metadata 的训练 checkpoint
