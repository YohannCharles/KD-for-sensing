## ADDED Requirements

### Requirement: 序列 CSV 使用 balanced_seq split 协议
DeepSense6G 序列 CSV 生成流程 MUST 使用单一的 `balanced_seq` train/test split 协议。split MUST 以完整 `seq_index` 为最小单位，MUST 不把同一 `seq_index` 的滑动窗口同时分配到 train 和 test，且 MUST 保持每个窗口仍只在单个 `seq_index` 内生成。

#### Scenario: split 可复现
- **WHEN** 用户使用相同原始 CSV、`training_set_pct`、`split_seed` 和 seq 数量控制配置
- **THEN** 序列生成流程 MUST 在重复运行时产生相同的 train/test `seq_index` 集合
- **AND** 系统 MUST 允许不同 `split_seed` 产生不同集合

#### Scenario: 标签分布感知选择 test seq
- **WHEN** 用户运行序列 CSV 预处理
- **THEN** 序列生成流程 MUST 基于生成后的窗口标签统计选择完整 test seq
- **AND** test 窗口数量 MUST 尽量接近配置的目标测试比例或显式 test seq 数
- **AND** test label 分布 MUST 尽量接近全量窗口 label 分布

#### Scenario: 小 seq 数场景的最少验证 seq
- **WHEN** 用户配置 `min_test_sequences` 且可用 `seq_index` 数量足以满足该约束
- **THEN** 序列生成流程 MUST 至少选择该数量的 test seq
- **AND** 如果该约束与显式 `test_sequence_count` 冲突，系统 MUST 抛出清晰错误或按文档定义的优先级处理

### Requirement: 序列 split metadata 可追踪
序列 CSV 预处理 MUST 为生成的 train/test split 记录可机器读取的 metadata。metadata MUST 足以解释当前 split 的策略、seed、seq 分配、窗口数和主要 label 分布。

#### Scenario: 写出 split metadata
- **WHEN** 用户运行序列 CSV 预处理
- **THEN** 系统 MUST 写出 split metadata sidecar
- **AND** metadata MUST 包含 `split_protocol: balanced_seq`、`split_seed`、`training_set_pct`、train/test `seq_index` 列表、train/test 窗口数和输出 CSV 路径

#### Scenario: 记录标签分布摘要
- **WHEN** 序列 CSV 中包含 beam 标签路径
- **THEN** split metadata MUST 记录 train/test 的 label 分布摘要
- **AND** 摘要 MUST 至少覆盖当前时隙标签或所有训练目标时隙中的一种明确口径

#### Scenario: 新统一 split 必须有 metadata
- **WHEN** 用户使用新预处理配置生成默认统一 split CSV
- **THEN** train/test CSV 旁 MUST 存在 split metadata sidecar
- **AND** metadata 中的 train/test 窗口数 MUST 与输出 CSV 行数一致
