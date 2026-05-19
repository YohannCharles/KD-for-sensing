## MODIFIED Requirements

### Requirement: 启用模态推导
数据构建流程 MUST 从 `experiment.task`、fusion teacher/student `modalities` 和显式 dataset 开关推导有序启用模态，并将该选择传递给 dataset、训练 batch 准备和评估 batch 准备。默认 fusion 模态 MUST 保持既有 `["image", "radar"]` 行为。

#### Scenario: 单模态任务推导
- **WHEN** 配置的 `experiment.task` 是 `image`、`radar`、`gps`、`lidar`、`mmwave` 或 `csi`
- **THEN** 数据构建流程 MUST 将启用模态推导为对应单模态
- **AND** 显式启用的 GPS、LiDAR、mmWave 或 CSI dataset 开关 MUST 与任务模态保持一致或被清晰拒绝

#### Scenario: fusion teacher/student 模态一致
- **WHEN** fusion KD 配置同时定义 teacher 和 student `modalities`
- **THEN** 数据构建流程 MUST 使用 teacher 与 student 的并集作为 dataset 启用模态
- **AND** 如果 teacher 与 student 模态不一致且配置未声明受支持跨模态蒸馏，系统 MUST 抛出清晰错误

#### Scenario: 未配置 fusion modalities
- **WHEN** fusion 配置没有显式设置 teacher 或 student `modalities`
- **THEN** 数据构建流程 MUST 使用 `["image", "radar"]`
- **AND** dataset MUST 保持旧 image+radar fusion 的样本字段兼容

#### Scenario: CSI dataset 开关冲突
- **WHEN** 配置设置 `data.dataset.use_csi: true` 但 `experiment.task` 或 fusion `modalities` 未启用 `csi`
- **THEN** 系统 MUST 拒绝构建 dataset
- **AND** 错误信息 MUST 指出 `use_csi` 与启用模态冲突

## ADDED Requirements

### Requirement: CSI 按模态选择加载样本
DeepSense6G/MMW dataset MUST 根据启用模态决定是否加载 CSI。未启用 CSI 时，CSI 路径列或文件缺失不得阻止当前任务运行；启用 CSI 时，dataset MUST 返回 `csi` 字段并保持其它未启用模态不读取。

#### Scenario: CSI-only 不读取其它输入模态文件
- **WHEN** 用户运行 `experiment.task: csi` 的训练或评估配置
- **THEN** dataset MUST 只读取 CSI、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、radar map、GPS、LiDAR 或 mmWave 加载逻辑
- **AND** 返回样本 MUST 包含 `csi`

#### Scenario: fusion 按 modalities 读取 CSI
- **WHEN** 用户运行 `experiment.task: fusion` 且配置 `modalities: ["gps", "csi"]`
- **THEN** dataset MUST 只读取 GPS、CSI、`input_beam` 和 `target_beam` 所需文件
- **AND** 返回样本 MUST 只包含启用模态对应输入字段和标签字段

### Requirement: CSI normalizer artifact 复用
数据构建流程 MUST 将训练集 CSI RMS normalizer 从 train dataset 传递给 test dataset，并允许训练/评估 metadata 记录该统计。

#### Scenario: dataloader 复用 CSI RMS
- **WHEN** `build_dataloaders` 构建启用 CSI 的 train 和 test dataset
- **THEN** train dataset MUST 先准备 CSI RMS normalizer
- **AND** test dataset MUST 接收同一个 CSI RMS normalizer 或等价数值
