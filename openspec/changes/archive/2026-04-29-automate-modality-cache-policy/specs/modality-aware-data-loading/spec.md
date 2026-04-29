## ADDED Requirements

### Requirement: 自动 cache policy 下的模态感知 cache 访问
Scenario 9 dataset MUST 在自动 cache policy 下保持按模态访问数据。启用 image 时 MAY 使用 image motion mask cache；启用 LiDAR 时 MAY 使用 LiDAR BEV cache；未启用对应模态时 MUST 完全跳过对应 cache 访问。

#### Scenario: image-only 自动使用 image cache
- **WHEN** 用户运行 image-only 配置且 `data.cache.policy: auto`
- **THEN** dataset MUST 对 image motion mask 启用 cache 读取
- **AND** cache miss 时 dataset MUST 生成并写入缺失的 image motion mask cache
- **AND** 返回样本字段、shape 和 dtype MUST 与未启用 cache 时一致

#### Scenario: LiDAR fusion 自动使用 LiDAR cache
- **WHEN** 用户运行包含 LiDAR 的 fusion 配置且 `data.cache.policy: auto`
- **THEN** dataset MUST 对 LiDAR BEV 启用 cache 读取
- **AND** cache miss 时 dataset MUST 生成并写入缺失的 LiDAR BEV cache
- **AND** 返回样本字段、shape 和 dtype MUST 与未启用 cache 时一致

#### Scenario: 非相关模态不触发 cache 初始化
- **WHEN** 用户运行不包含 image 或 LiDAR 的单模态或 fusion 配置
- **THEN** dataset 初始化 MUST 不创建 image 或 LiDAR cache 目录
- **AND** dataset 取样 MUST 不调用 image 或 LiDAR cache path 解析逻辑
