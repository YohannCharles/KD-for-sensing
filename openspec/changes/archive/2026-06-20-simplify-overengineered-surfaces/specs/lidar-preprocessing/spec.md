## ADDED Requirements

### Requirement: 未接入 LiDAR pillar encoder 原型不属于当前支持面
当前 LiDAR preprocessing support surface MUST 以点云读取、BEV 伪图像构造、cache、normalization、质量摘要和启用 LiDAR 的 dataset flat sample 为准。未注册、未配置、未被训练/评估/诊断入口消费的 pillar encoder 或 spatial encoder 原型 MUST 不作为当前必须保留的 LiDAR 能力。

#### Scenario: 删除未接入 pillar encoder
- **WHEN** `lidar_pillar_encoder` 或等价原型没有 registry、config、dataset、trainer、CLI、README/docs 或 current OpenSpec 消费
- **THEN** 本 change MAY 删除该源码模块
- **AND** LiDAR BEV 构造、cache 预热、normalization 和输入质量摘要 MUST 保持可用

#### Scenario: 新增 pillar 能力必须重新走 OpenSpec
- **WHEN** 后续需要 pillar encoder、pillar scatter 或 LiDAR spatial encoder 作为当前训练能力
- **THEN** 项目 MUST 通过新的 OpenSpec change 声明模型注册、配置入口、dataset contract、forward metadata 和 focused tests
- **AND** 不得通过恢复未接入原型文件把该能力静默加入 current surface
