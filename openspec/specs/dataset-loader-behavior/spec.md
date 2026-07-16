# dataset-loader-behavior Specification

## Purpose

定义 MMW 四模态训练和评估所需的 dataset 与 DataLoader 行为，保证 retained workflow 使用统一 prepared sequence、batch 语义和可复现加载参数。

## Requirements

### Requirement: MMW 四模态样本契约

dataset MUST 只为 T2/baseline 提供 image、radar、gps、lidar 的 prepared sequence、beam target、时间元数据和可用性 mask。

#### Scenario: 构建 retained recipe 数据集

- **WHEN** T2、S1、AMBER-Full 或 RMBP-MM 构建 train、validation 或 test dataset
- **THEN** batch MUST 使用统一四模态顺序与 MMW label contract
- **AND** 不得要求已退役数据族或物理/通道 payload

### Requirement: DataLoader 运行参数可配置

训练和评估 MUST 从 current recipe 解析 batch size、worker、pin memory、drop-last 与可用的 persistent-worker 参数；`num_workers=0` 时不得传递仅限多 worker 的参数。

#### Scenario: CPU smoke

- **WHEN** 配置 `num_workers: 0`
- **THEN** DataLoader MUST 能在无本地数据的 synthetic smoke 中迭代
- **AND** 不得传递 `persistent_workers` 或 `prefetch_factor`

### Requirement: MMW loader 不保留历史数据兼容路径

MMW loader MUST 只消费 current prepared sequence 与统一四模态字段。系统 MUST 不提供 data cache、descriptor、protocol-split compatibility 或 route-specific GPS/yaw fallback；all-weather evaluation MUST 使用同一 GPS 表示。

#### Scenario: 构建 all-weather batch

- **WHEN** retained workflow 构建任一 domain 的训练或评估 batch
- **THEN** GPS 与 temporal metadata MUST 来自 current prepared fields
- **AND** runtime MUST 不根据旧 route、RSU-local 坐标或 yaw 选择替代输入
