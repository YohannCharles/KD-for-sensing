## MODIFIED Requirements

### Requirement: DeepSense6G 场景感知数据构建
数据构建流程 MUST 根据 DeepSense6G 场景选择解析数据根目录和 split CSV。canonical 配置 MUST 使用 `data.dataset.type: deepsense6g`。旧 `scenario9`、`scenario31` 和 `scenario32` dataset type MUST 不再可构建。

#### Scenario: 旧 scenario9 配置被拒绝
- **WHEN** 用户运行包含 `the scene-9 dataset-type spelling` 的旧配置
- **THEN** 数据构建流程 MUST 拒绝该配置
- **AND** 错误信息 MUST 说明应使用 `data.dataset.type: deepsense6g` 和 `data.dataset.scene: 9`

#### Scenario: 通用 deepsense6g 类型默认选择 Scenario 31
- **WHEN** 用户运行 `data.dataset.type: deepsense6g` 且未显式设置 `data.dataset.scene`
- **THEN** 数据构建流程 MUST 构建 Scenario 31 对应的 DeepSense6G dataset
- **AND** 启用模态推导 MUST 继续按 `experiment.task` 或 fusion `modalities` 生效

#### Scenario: 显式选择 Scenario 32
- **WHEN** 用户运行 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 32`
- **THEN** 数据构建流程 MUST 构建 Scenario 32 对应的 DeepSense6G dataset
- **AND** 启用模态推导 MUST 继续按 `experiment.task` 或 fusion `modalities` 生效

#### Scenario: split metadata 记录场景
- **WHEN** 训练或评估构建 train/test dataset
- **THEN** split metadata MUST 记录每个 split 的 `scene_id`、`scene_slug`、CSV 路径和样本数
- **AND** 这些字段 MUST 出现在最终配置、运行日志或测试报告中

#### Scenario: 场景不影响模态按需读取
- **WHEN** 用户在任一受支持 DeepSense6G 场景上运行 mmWave-only 或 GPS+mmWave fusion 配置
- **THEN** dataset MUST 只读取启用模态所需文件和 beam label 文件
- **AND** 未启用模态的缺失文件不得阻止该任务运行
