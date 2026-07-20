# dataset-loader-behavior Specification

## Purpose

定义 MMW 与受限 DeepSense6G T2 四模态训练和评估所需的 dataset 与 DataLoader 行为，保证 retained workflow 使用统一 batch 语义和可复现加载参数。
## Requirements
### Requirement: 双数据集四模态样本契约

dataset MUST 只为 current T2/baseline workflow 提供 image、radar、gps、lidar 的 prepared sequence、beam target、时间元数据和可用性 mask。MMW 与 DeepSense6G MUST 输出相同的四模态 batch 字段；两者的标签解析 MUST 由各自 dataset owner 保持显式。

#### Scenario: 构建 current recipe 数据集

- **WHEN** MMW T2、S1、AMBER-Full、RMBP-MM 或 DeepSense6G T2 构建 train、validation 或 test dataset
- **THEN** batch MUST 使用统一四模态顺序与各自明确的 label contract
- **AND** 不得要求已退役数据族或物理/通道 payload

### Requirement: DataLoader 运行参数可配置

训练和评估 MUST 从 current recipe 解析 batch size、worker、pin memory、drop-last 与可用的 persistent-worker 参数；`num_workers=0` 时不得传递仅限多 worker 的参数。

#### Scenario: CPU smoke

- **WHEN** 配置 `num_workers: 0`
- **THEN** DataLoader MUST 能在无本地数据的 synthetic smoke 中迭代
- **AND** 不得传递 `persistent_workers` 或 `prefetch_factor`

### Requirement: loader 不保留历史数据兼容路径

MMW loader MUST 只消费 current prepared sequence 与统一四模态字段。DeepSense6G loader MUST 只消费受限 Scene31–34 的标准 CSV 和 future beam 文件。系统 MUST 不提供 data cache、descriptor、protocol-split compatibility、route-specific GPS/yaw fallback 或历史输入 payload。

#### Scenario: 构建任一 current batch

- **WHEN** retained workflow 构建任一 MMW domain 或 DeepSense6G scene 的训练或评估 batch
- **THEN** GPS 与 temporal metadata MUST 来自当前字段
- **AND** runtime MUST 不根据旧 route、RSU-local 坐标、yaw、CSI 或 input beam 选择替代输入

### Requirement: 四模态数据输入必须在读取前通过完整 schema 校验
MMW current loader 和 launcher preflight MUST 共同校验连续的 camera/radar/gps/bs_gps/lidar/future label 列、0..63 label、所需直接及派生资源和受 data root 限制的相对路径。缺失 label 不得转换为类别零。

#### Scenario: 缺少 BS GPS 或 future label
- **WHEN** MMW CSV 缺少任一 required BS GPS 或 future label 字段
- **THEN** preflight 和 loader MUST 在训练开始前拒绝该 CSV
- **AND** 不得报告 ready 或生成可训练 split

#### Scenario: 路径越出 data root
- **WHEN** CSV path 为绝对路径、包含 `..` 或通过 symlink 越出 data root
- **THEN** loader MUST 拒绝该路径
- **AND** 错误 MUST 标识对应字段

### Requirement: 数据读取的随机性与不可变标签必须可复现
DeepSense6G future-beam label MUST 在数据集构造期完成 finite/size 校验并缓存硬标签。启用 LiDAR augmentation 时，随机数 MUST 从确定性的 worker/epoch/sample identity 派生；无法对预计算 BEV 提供等价增强时 MUST 明确拒绝。

#### Scenario: 重复读取相同 DeepSense sample
- **WHEN** 同一 dataset instance 多次访问同一 DeepSense sample
- **THEN** future-beam source file MUST 不被重复解析
- **AND** 返回的 64-class argmax label MUST 保持一致
