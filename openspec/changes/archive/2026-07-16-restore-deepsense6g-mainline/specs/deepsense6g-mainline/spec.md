## ADDED Requirements

### Requirement: DeepSense6G 仅提供受限四模态主线

系统 MUST 仅接受 `data.dataset.type: deepsense6g` 的整数 `scene` 值 31、32、33 或 34。dataset MUST 从 `dataset/DeepSense6G/scenario{scene}` 或显式 `data_root` 的 CSV 构建 image、radar RA/DA、GPS 和 LiDAR 时间序列，并输出现有四模态 batch 契约要求的字段、稳定 sample ID 与 dataset/scene metadata。dataset MUST 拒绝缺失的必需编号列、资源路径或不支持场景，且 MUST 不接受场景别名。

#### Scenario: 构建 Scene31 四模态样本

- **WHEN** 配置选择 `deepsense6g`、`scene: 31` 和有效的标准 train CSV
- **THEN** dataset MUST 产出 image、radar_ra、radar_da、gps、lidar 和 target_beam
- **AND** 输出 MUST 可直接由共享 batch preparation 消费

#### Scenario: 请求非主线场景

- **WHEN** 配置使用 Scene9、Scene23、字符串场景名或任何其他值
- **THEN** 配置校验或 dataset 构建 MUST 失败
- **AND** runtime MUST 不回退到历史场景映射

### Requirement: future beam 必须归约为 64 类硬标签

每个 DeepSense6G 样本的预测时间步 MUST 从对应 `future_beamN` 文件读取有限的 64 维功率向量，并以 `argmax` 生成 `target_beam`。系统 MUST 拒绝长度错误、非有限或缺失的 future beam 文件，且 MUST 不输出软标签、input beam、CSI 或原始毫米波 payload。

#### Scenario: 读取有效 future beam

- **WHEN** future beam 文件包含一个有限的 64 维功率向量
- **THEN** target_beam MUST 等于该向量最大值的索引
- **AND** target_beam MUST 是供现有 64 类 beam loss 使用的整数标签

#### Scenario: 读取无效 future beam

- **WHEN** future beam 文件缺失、包含非有限值或维度不是 64
- **THEN** 取样 MUST 抛出说明数据路径和标签契约的错误
- **AND** runtime MUST 不以零、软标签或其他历史目标替代该标签

### Requirement: DeepSense6G split 与 GPS 标准化显式

DeepSense6G MUST 仅将 train、test 和显式配置的 validation CSV 作为 split 来源；validation 不得隐式复用 test CSV。训练 split 拟合的当前 GPS scaler MUST 被注入 validation/test dataset，且所有 split MUST 使用相同的相对极坐标 GPS 表示。

#### Scenario: 构建训练和测试 split

- **WHEN** recipe 提供 train 与 test CSV，且没有 val CSV
- **THEN** factory MUST 构建 train 和 test dataset
- **AND** validation dataset MUST 保持未定义而不是复用 test 数据

#### Scenario: 使用训练 GPS scaler 评估

- **WHEN** training dataset 已拟合 GPS scaler 并构建 DeepSense6G test dataset
- **THEN** test dataset MUST 使用训练 scaler 变换 GPS 特征
- **AND** test dataset MUST 不重新拟合 scaler
