## MODIFIED Requirements

### Requirement: MMW loader 不保留历史数据兼容路径

MMW loader MUST 只消费 current prepared sequence 与统一四模态字段。data cache、descriptor、protocol-split compatibility 以及 route-specific GPS/yaw fallback MUST 从 T2/baseline runtime 删除；all-weather evaluation MUST 使用同一 GPS 表示。

#### Scenario: 构建 all-weather batch

- **WHEN** retained workflow 构建任一 domain 的训练或评估 batch
- **THEN** GPS 与 temporal metadata MUST 来自 current prepared fields
- **AND** runtime MUST 不根据旧 route、RSU-local 坐标或 yaw 选择替代输入
