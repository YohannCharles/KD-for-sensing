## ADDED Requirements

### Requirement: 实验输出记录 split 协议
训练和评估流程 MUST 在运行产物中记录足够的 split 协议信息，用于判断不同实验是否使用同一数据协议并可横向比较。记录 MUST 包含实际 CSV 路径、样本数和 `balanced_seq` split metadata 路径或核心字段。

#### Scenario: 训练输出包含 split metadata 引用
- **WHEN** 训练入口构建 train/test dataset
- **THEN** `final_config.yaml`、`train_log.json` 或等价运行产物 MUST 记录 split metadata 路径或核心字段
- **AND** 记录 MUST 包含 split 策略、seed、train/test `seq_index` 数量和 train/test 样本数

#### Scenario: 评估输出包含 split 协议
- **WHEN** 评估入口构建 test dataset
- **THEN** 评估报告 MUST 记录实际使用的 test CSV 和可用的 split 协议信息
- **AND** 当当前 CSV 缺少 `balanced_seq` split metadata 时，系统 MUST 给出清晰错误或显式警告，避免把未知 split 协议误当成新协议结果

#### Scenario: 跨模态 split 可比较
- **WHEN** 用户使用同一组 train/test CSV 运行 image、radar、GPS、LiDAR、mmWave 或 fusion 实验
- **THEN** 各运行产物中的 split 协议信息 MUST 能显示它们使用相同 CSV 和相同 split metadata
- **AND** 如果 CSV 路径或 split metadata 不同，用户 MUST 能从运行产物中看出这些结果不应直接作为同一 split 协议比较
