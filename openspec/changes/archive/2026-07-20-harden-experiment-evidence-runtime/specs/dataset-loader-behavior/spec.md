## ADDED Requirements

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
