## MODIFIED Requirements

### Requirement: Scenario 9 按模态选择加载样本
Scenario 9 dataset MUST 根据训练或评估配置中的启用模态加载样本字段。未启用模态的文件 MUST 不被读取，未启用模态的输入字段 MUST 不出现在样本字典中，且未启用模态的路径列或文件缺失不得阻止当前任务运行。dataset MUST 始终加载 beam 历史标签和 future beam 目标标签。

#### Scenario: GPS-only 不读取 image 或 radar 文件
- **WHEN** 用户运行 `experiment.task: gps` 的训练或评估配置
- **THEN** dataset MUST 只读取 GPS、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image motion mask 或 radar map 加载逻辑
- **AND** 返回样本 MUST 不包含 `image`、`radar_ra` 或 `radar_da`

#### Scenario: LiDAR-only 不读取 image 或 radar 文件
- **WHEN** 用户运行 `experiment.task: lidar` 的训练或评估配置
- **THEN** dataset MUST 只读取 LiDAR、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image motion mask、radar map 或 GPS 加载逻辑
- **AND** 返回样本 MUST 不包含 `image`、`radar_ra`、`radar_da` 或 `gps`

#### Scenario: mmWave-only 不读取其它输入模态文件
- **WHEN** 用户运行 `experiment.task: mmwave` 的训练或评估配置
- **THEN** dataset MUST 只读取 mmWave、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image motion mask、radar map、GPS 或 LiDAR 加载逻辑
- **AND** 返回样本 MUST 包含 `mmwave`
- **AND** 返回样本 MUST 不包含 `image`、`radar_ra`、`radar_da`、`gps` 或 `lidar`

#### Scenario: radar-only 只读取 radar 输入
- **WHEN** 用户运行 `experiment.task: radar` 的训练或评估配置
- **THEN** dataset MUST 只读取 radar、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、GPS、LiDAR 或 mmWave 加载逻辑
- **AND** 返回样本 MUST 包含 `radar_ra` 和 `radar_da`

#### Scenario: image-only 只读取 image 输入
- **WHEN** 用户运行 `experiment.task: image` 的训练或评估配置
- **THEN** dataset MUST 只读取 image、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 radar、GPS、LiDAR 或 mmWave 加载逻辑
- **AND** 返回样本 MUST 包含 `image`

#### Scenario: fusion 按 modalities 读取输入
- **WHEN** 用户运行 `experiment.task: fusion` 且配置 `modalities: ["radar", "mmwave"]`
- **THEN** dataset MUST 只读取 radar、mmWave、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、GPS 或 LiDAR 加载逻辑
- **AND** 返回样本 MUST 只包含启用模态对应输入字段和标签字段

### Requirement: 启用模态推导
数据构建流程 MUST 从 `experiment.task`、fusion teacher/student `modalities` 和显式 dataset 开关推导有序启用模态，并将该选择传递给 dataset、训练 batch 准备和评估 batch 准备。默认 fusion 模态 MUST 保持既有 `["image", "radar"]` 行为。

#### Scenario: 单模态任务推导
- **WHEN** 配置的 `experiment.task` 是 `image`、`radar`、`gps`、`lidar` 或 `mmwave`
- **THEN** 数据构建流程 MUST 将启用模态推导为对应单模态
- **AND** 显式启用的 GPS、LiDAR 或 mmWave dataset 开关 MUST 与任务模态保持一致或被清晰拒绝

#### Scenario: fusion teacher/student 模态一致
- **WHEN** fusion KD 配置同时定义 teacher 和 student `modalities`
- **THEN** 数据构建流程 MUST 使用 teacher 与 student 的并集作为 dataset 启用模态
- **AND** 如果 teacher 与 student 模态不一致且配置未声明受支持跨模态蒸馏，系统 MUST 抛出清晰错误

#### Scenario: 未配置 fusion modalities
- **WHEN** fusion 配置没有显式设置 teacher 或 student `modalities`
- **THEN** 数据构建流程 MUST 使用 `["image", "radar"]`
- **AND** dataset MUST 保持旧 image+radar fusion 的样本字段兼容

#### Scenario: mmWave dataset 开关冲突
- **WHEN** 配置设置 `data.dataset.use_mmwave: true` 但 `experiment.task` 或 fusion `modalities` 未启用 `mmwave`
- **THEN** 系统 MUST 拒绝构建 dataset
- **AND** 错误信息 MUST 指出 `use_mmwave` 与启用模态冲突
