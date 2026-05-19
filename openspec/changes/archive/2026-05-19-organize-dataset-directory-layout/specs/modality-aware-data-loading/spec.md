## MODIFIED Requirements

### Requirement: DeepSense6G 场景感知数据构建
数据构建流程 MUST 根据 DeepSense6G 场景选择和 dataset layout descriptor 解析数据根目录和 split CSV。canonical 配置 MUST 使用 `data.dataset.type: deepsense6g`。旧 `scenario9`、`scenario31` 和 `scenario32` dataset type MUST 不再可构建。

#### Scenario: 旧 scenario9 配置被拒绝
- **WHEN** 用户运行包含 `the scene-9 dataset-type spelling` 的旧配置
- **THEN** 数据构建流程 MUST 拒绝该配置
- **AND** 错误信息 MUST 说明应使用 `data.dataset.type: deepsense6g` 和 `data.dataset.scene: 9`

#### Scenario: 通用 deepsense6g 类型默认选择 Scenario 31
- **WHEN** 用户运行 `data.dataset.type: deepsense6g` 且未显式设置 `data.dataset.scene`
- **THEN** 数据构建流程 MUST 构建 Scenario 31 对应的 DeepSense6G dataset
- **AND** 数据根目录 MUST 默认为 `dataset/DeepSense6G/scenario31`
- **AND** 启用模态推导 MUST 继续按 `experiment.task` 或 fusion `modalities` 生效

#### Scenario: 显式选择 Scenario 32
- **WHEN** 用户运行 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 32`
- **THEN** 数据构建流程 MUST 构建 Scenario 32 对应的 DeepSense6G dataset
- **AND** 数据根目录 MUST 默认为 `dataset/DeepSense6G/scenario32`
- **AND** 启用模态推导 MUST 继续按 `experiment.task` 或 fusion `modalities` 生效

#### Scenario: split metadata 记录场景
- **WHEN** 训练或评估构建 train/test dataset
- **THEN** split metadata MUST 记录每个 split 的 `scene_id`、`scene_slug`、CSV 路径和样本数
- **AND** 这些字段 MUST 出现在最终配置、运行日志或测试报告中

#### Scenario: 场景不影响模态按需读取
- **WHEN** 用户在任一受支持 DeepSense6G 场景上运行 mmWave-only 或 GPS+mmWave fusion 配置
- **THEN** dataset MUST 只读取启用模态所需文件和 beam label 文件
- **AND** 未启用模态的缺失文件不得阻止该任务运行

## ADDED Requirements

### Requirement: DeepSense6G CSV 相对路径基准
DeepSense6G dataset MUST 继续以解析后的 scene root 作为 CSV 内相对文件路径的基准目录。将场景目录移动到 `dataset/DeepSense6G/scenario*` 后，CSV 内现有相对路径格式 MUST 不需要增加 `DeepSense6G` 前缀。

#### Scenario: 读取新规范目录下的相对路径
- **WHEN** dataset 的 `data_root` 为 `dataset/DeepSense6G/scenario31` 且 CSV 内某个 radar 路径为 `/unit1/radar_data_RA/sample.npy`
- **THEN** 文件读取 MUST 解析到 `dataset/DeepSense6G/scenario31/unit1/radar_data_RA/sample.npy`
- **AND** 系统 MUST 不把该路径解析到 `dataset/unit1/radar_data_RA/sample.npy`

#### Scenario: 读取显式旧目录下的相对路径
- **WHEN** dataset 的 `data_root` 被显式设置为 `dataset/scenario31` 且 CSV 内某个 mmWave 路径为 `/unit1/pwr/sample.txt`
- **THEN** 文件读取 MUST 解析到 `dataset/scenario31/unit1/pwr/sample.txt`
- **AND** 系统 MUST 不要求用户修改 CSV 内相对路径

### Requirement: DeepSense6G 预处理路径重定向
DeepSense6G 序列 CSV 预处理 MUST 使用 dataset layout descriptor 解析 scene root。命令行或配置中的场景覆盖 MUST 同时更新 `preprocessing.data_root` 和默认 `preprocessing.csv_path` 到目标场景的规范目录，除非用户显式提供自定义绝对路径。

#### Scenario: 预处理默认 Scenario 31 路径
- **WHEN** 用户运行默认 DeepSense6G sequence CSV 预处理配置
- **THEN** `preprocessing.data_root` MUST 指向 `dataset/DeepSense6G/scenario31`
- **AND** `preprocessing.csv_path` MUST 指向 `dataset/DeepSense6G/scenario31/scenario31_RA.csv`

#### Scenario: 预处理场景覆盖到 Scenario 9
- **WHEN** 用户在 sequence CSV 预处理中覆盖 `data.dataset.scene: 9`
- **THEN** `preprocessing.data_root` MUST 更新为 `dataset/DeepSense6G/scenario9`
- **AND** 默认 `preprocessing.csv_path` MUST 更新为 `dataset/DeepSense6G/scenario9/scenario9_RA.csv`
