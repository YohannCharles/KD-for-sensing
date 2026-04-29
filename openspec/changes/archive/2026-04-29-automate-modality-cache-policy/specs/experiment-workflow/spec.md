## ADDED Requirements

### Requirement: 实验入口自动解析 cache policy
训练、评估和 profile 入口 MUST 在构建 dataset 前解析 cache policy，并将解析后的实际 cache 读写开关传递给 dataset。解析过程 MUST 使用配置中的启用模态，不得要求用户为每个单模态或 fusion 组合手动设置低层 cache 读写字段。

#### Scenario: 单模态训练自动解析
- **WHEN** 用户运行 `configs/image/teacher_no_kd.yaml` 且未手动设置 image cache 低层开关
- **THEN** 训练入口 MUST 根据 cache policy 自动决定 `image_motion_use_cache` 和 `image_motion_write_cache`
- **AND** 用户 MUST 能通过命令行覆盖这些低层开关

#### Scenario: 任意 fusion 组合自动解析
- **WHEN** 用户运行任意 fusion 配置并声明 `modalities`
- **THEN** 训练入口 MUST 只为该组合包含的 image 或 LiDAR 模态解析 cache 行为
- **AND** 不包含 image 或 LiDAR 的组合 MUST 不需要相关 cache 参数才能启动

#### Scenario: profile 使用相同 policy
- **WHEN** 用户运行训练 I/O profile 入口
- **THEN** profile MUST 使用与训练入口一致的 cache policy 解析逻辑
- **AND** profile 输出 MUST 记录实际 cache policy 和 cache 目录
