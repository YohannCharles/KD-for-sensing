## MODIFIED Requirements

### Requirement: 双数据集四模态样本契约

dataset MUST 只为 current T2/baseline workflow 提供 image、radar、gps、lidar 的 prepared sequence、beam target、时间元数据和可用性 mask。MMW 与 DeepSense6G MUST 输出相同的四模态 batch 字段；两者的标签解析 MUST 由各自 dataset owner 保持显式。

#### Scenario: 构建 current recipe 数据集

- **WHEN** MMW T2、S1、AMBER-Full、RMBP-MM 或 DeepSense6G T2 构建 train、validation 或 test dataset
- **THEN** batch MUST 使用统一四模态顺序与各自明确的 label contract
- **AND** 不得要求已退役数据族或物理/通道 payload

### Requirement: loader 不保留历史数据兼容路径

MMW loader MUST 只消费 current prepared sequence 与统一四模态字段。DeepSense6G loader MUST 只消费受限 Scene31–34 的标准 CSV 和 future beam 文件。系统 MUST 不提供 data cache、descriptor、protocol-split compatibility、route-specific GPS/yaw fallback 或历史输入 payload。

#### Scenario: 构建任一 current batch

- **WHEN** retained workflow 构建任一 MMW domain 或 DeepSense6G scene 的训练或评估 batch
- **THEN** GPS 与 temporal metadata MUST 来自当前字段
- **AND** runtime MUST 不根据旧 route、RSU-local 坐标、yaw、CSI 或 input beam 选择替代输入
