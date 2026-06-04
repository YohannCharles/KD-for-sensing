## MODIFIED Requirements

### Requirement: 单模态 snapshot baseline 配置矩阵
项目 MUST 为 `image`、`radar`、`gps`、`lidar` 和 `mmwave` 提供可加载的 snapshot next-frame supervised 单模态配置入口。每个入口 MUST 复用对应模态的现有 loader、encoder、normalization、loss 和指标契约。

#### Scenario: 加载 image snapshot 配置
- **WHEN** 用户加载 `configs/image/snapshot_next_frame_supervised.yaml`
- **THEN** 最终配置 MUST 设置 `experiment.task: image`
- **AND** 最终配置 MUST 使用 image-only 当前帧模型
- **AND** 最终配置 MUST 不包含 `distillation.type`

#### Scenario: 加载所有单模态 snapshot 配置
- **WHEN** 开发者加载 `configs/<modality>/snapshot_next_frame_supervised.yaml`
- **THEN** `<modality>` 为 `image`、`radar`、`gps`、`lidar` 或 `mmwave` 时配置 MUST 可加载
- **AND** 每个配置 MUST 只要求对应单模态输入字段和标签字段

### Requirement: 多模态 snapshot fusion baseline 配置矩阵
项目 MUST 为合法多模态组合提供 snapshot next-frame supervised fusion 配置入口，至少 MUST 支持五模态 `image_radar_gps_lidar_mmwave`。这些配置 MUST 复用现有 fusion 输入路由和固定模态顺序。

#### Scenario: 加载五模态 snapshot fusion 配置
- **WHEN** 用户加载 `configs/fusion/image_radar_gps_lidar_mmwave_snapshot_next_frame_supervised.yaml`
- **THEN** 最终配置 MUST 设置 `experiment.task: fusion`
- **AND** 最终配置 MUST 构建无时序 snapshot fusion 模型
- **AND** 最终配置 MUST 不包含 distillation 配置块

