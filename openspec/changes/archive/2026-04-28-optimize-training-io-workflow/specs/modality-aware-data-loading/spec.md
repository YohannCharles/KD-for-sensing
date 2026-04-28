## ADDED Requirements

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

#### Scenario: radar-only 只读取 radar 输入
- **WHEN** 用户运行 `experiment.task: radar` 的训练或评估配置
- **THEN** dataset MUST 只读取 radar、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、GPS 或 LiDAR 加载逻辑
- **AND** 返回样本 MUST 包含 `radar_ra` 和 `radar_da`

#### Scenario: image-only 只读取 image 输入
- **WHEN** 用户运行 `experiment.task: image` 的训练或评估配置
- **THEN** dataset MUST 只读取 image、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 radar、GPS 或 LiDAR 加载逻辑
- **AND** 返回样本 MUST 包含 `image`

#### Scenario: fusion 按 modalities 读取输入
- **WHEN** 用户运行 `experiment.task: fusion` 且配置 `modalities: ["radar", "gps"]`
- **THEN** dataset MUST 只读取 radar、GPS、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image 或 LiDAR 加载逻辑
- **AND** 返回样本 MUST 只包含启用模态对应输入字段和标签字段

### Requirement: 启用模态推导
数据构建流程 MUST 从 `experiment.task`、fusion teacher/student `modalities` 和显式 dataset 开关推导有序启用模态，并将该选择传递给 dataset、训练 batch 准备和评估 batch 准备。默认 fusion 模态 MUST 保持既有 `["image", "radar"]` 行为。

#### Scenario: 单模态任务推导
- **WHEN** 配置的 `experiment.task` 是 `image`、`radar`、`gps` 或 `lidar`
- **THEN** 数据构建流程 MUST 将启用模态推导为对应单模态
- **AND** 显式启用的 GPS 或 LiDAR dataset 开关 MUST 与任务模态保持一致或被清晰拒绝

#### Scenario: fusion teacher/student 模态一致
- **WHEN** fusion KD 配置同时定义 teacher 和 student `modalities`
- **THEN** 数据构建流程 MUST 使用 teacher 与 student 的并集作为 dataset 启用模态
- **AND** 如果 teacher 与 student 模态不一致且配置未声明受支持跨模态蒸馏，系统 MUST 抛出清晰错误

#### Scenario: 未配置 fusion modalities
- **WHEN** fusion 配置没有显式设置 teacher 或 student `modalities`
- **THEN** 数据构建流程 MUST 使用 `["image", "radar"]`
- **AND** dataset MUST 保持旧 image+radar fusion 的样本字段兼容

### Requirement: 标签张量维度稳定
Scenario 9 dataset MUST 返回稳定维度的 `input_beam` 和 `target_beam`。单样本 `target_beam` MUST 保持形状 `[num_pred]`，batch 后 MUST 保持形状 `[batch_size, num_pred]`，包括 `num_pred=1` 的情况。

#### Scenario: num_pred 为 1
- **WHEN** dataset 配置 `num_pred: 1` 且读取一个样本
- **THEN** 返回样本的 `target_beam` MUST 是一维张量且长度为 1
- **AND** DataLoader batch 的 `target_beam` MUST 是二维张量且第二维长度为 1
- **AND** `prepare_labels` MUST 能正常拼接历史 beam 和目标 beam

#### Scenario: num_pred 大于 1
- **WHEN** dataset 配置 `num_pred: 3` 且读取一个 batch
- **THEN** batch 的 `target_beam` MUST 保持 `[batch_size, 3]`
- **AND** 训练、验证和评估指标 MUST 继续按 `num_pred + 1` 个时隙计算

### Requirement: DataLoader 运行参数可配置
训练和评估入口 MUST 支持通过配置控制 DataLoader 运行参数，包括 batch size、`num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor` 和 `drop_last`。当 `num_workers=0` 时，系统 MUST 不传入仅适用于多 worker 的参数。

#### Scenario: 多 worker DataLoader
- **WHEN** 配置设置 `data.dataloader.num_workers: 4`、`persistent_workers: true`、`pin_memory: true` 和 `prefetch_factor: 2`
- **THEN** 训练和评估 DataLoader MUST 使用这些参数
- **AND** train loader MUST 按配置决定是否 `drop_last`

#### Scenario: 单进程 DataLoader
- **WHEN** 配置设置 `data.dataloader.num_workers: 0`
- **THEN** 系统 MUST 不向 DataLoader 传入 `persistent_workers` 或 `prefetch_factor`
- **AND** DataLoader MUST 能在 CPU-only smoke test 中正常迭代

#### Scenario: 评估复用 loader 参数解析
- **WHEN** 用户运行评估入口并配置 DataLoader 参数
- **THEN** 评估入口 MUST 使用与训练入口一致的参数解析逻辑
- **AND** 评估入口 MUST 保持 `shuffle: false`
