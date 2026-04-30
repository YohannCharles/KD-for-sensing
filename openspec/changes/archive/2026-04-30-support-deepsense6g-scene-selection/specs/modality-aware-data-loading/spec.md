## ADDED Requirements

### Requirement: DeepSense6G 场景感知数据构建
数据构建流程 MUST 根据 DeepSense6G 场景选择解析 dataset 类型、数据根目录和 split CSV。现有 `scenario9` 配置 MUST 继续可构建，并 MUST 等价于选择 Scenario 9。

#### Scenario: scenario9 兼容旧配置
- **WHEN** 用户运行包含 `data.dataset.type: scenario9` 的旧配置
- **THEN** 数据构建流程 MUST 将该配置视为 DeepSense6G Scenario 9
- **AND** dataset 返回字段、启用模态推导和标签张量 shape MUST 保持兼容

#### Scenario: 通用 deepsense6g 类型选择 Scenario 32
- **WHEN** 用户运行 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 32`
- **THEN** 数据构建流程 MUST 构建 Scenario 32 对应的 DeepSense6G dataset
- **AND** 启用模态推导 MUST 继续按 `experiment.task` 或 fusion `modalities` 生效

#### Scenario: split metadata 记录场景
- **WHEN** 训练或评估构建 train/test dataset
- **THEN** split metadata MUST 记录每个 split 的 `scene_id`、`scene_slug`、CSV 路径和样本数
- **AND** 这些字段 MUST 出现在最终配置、运行日志或测试报告中

#### Scenario: 场景不影响模态按需读取
- **WHEN** 用户在 Scenario 32 上运行 mmWave-only 或 GPS+mmWave fusion 配置
- **THEN** dataset MUST 只读取启用模态所需文件和 beam label 文件
- **AND** 未启用模态的缺失文件不得阻止该任务运行

