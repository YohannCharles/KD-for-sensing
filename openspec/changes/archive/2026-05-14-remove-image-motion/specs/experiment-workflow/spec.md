## MODIFIED Requirements

### Requirement: 实验入口自动解析 cache policy
训练、评估和 profile 入口 MUST 在构建 dataset 前解析 cache policy，并将解析后的实际 cache 读写开关传递给 dataset。解析过程 MUST 使用配置中的启用模态，不得要求用户为每个单模态或 fusion 组合手动设置低层 cache 读写字段。系统 MUST 不再解析或传递 `image_motion_*` 低层开关。

#### Scenario: 单模态 image 训练不解析 image motion cache
- **WHEN** 用户运行 image-only 训练配置
- **THEN** 训练入口 MUST 使用 RGB/ImageNet image 输入构建 dataset
- **AND** 训练入口 MUST 不生成 `image_motion_use_cache` 或 `image_motion_write_cache`
- **AND** 用户 MUST 不能通过命令行恢复这些已删除低层开关

#### Scenario: 任意 fusion 组合自动解析
- **WHEN** 用户运行任意 fusion 配置并声明 `modalities`
- **THEN** 训练入口 MUST 只为该组合包含的受支持 cache 模态解析 cache 行为
- **AND** 不包含 LiDAR 的组合 MUST 不需要相关 cache 参数才能启动
- **AND** 包含 image 的组合 MUST 不需要且不得接受 image motion cache 参数

#### Scenario: profile 使用相同 policy
- **WHEN** 用户运行训练 I/O profile 入口
- **THEN** profile MUST 使用与训练入口一致的 cache policy 解析逻辑
- **AND** profile 输出 MUST 记录实际 cache policy 和受支持 cache 目录
- **AND** profile 输出 MUST 不记录 image motion cache 目录或读写开关
