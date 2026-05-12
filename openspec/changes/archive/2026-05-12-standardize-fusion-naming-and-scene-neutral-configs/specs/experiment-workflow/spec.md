## ADDED Requirements

### Requirement: Fusion 实验配置命名保持场景中立
推荐的 fusion 实验配置文件名、`experiment.name` 和 `output.run_name` MUST 不硬编码 `scene32_` 前缀。场景选择 MUST 通过 dataset 场景字段、命令行覆盖、输出根目录或 checkpoint metadata 表达，而不是混入方法 slug。

#### Scenario: MARF 主配置不包含 scene32 前缀
- **WHEN** 开发者加载推荐 MARF 主实验配置
- **THEN** 配置路径、`experiment.name` 和 `output.run_name` MUST 不包含 `scene32_`
- **AND** 配置 MAY 继续默认选择 Scene 32 数据集字段

#### Scenario: CRAF/MARF ablation 配置不包含 scene32 前缀
- **WHEN** 开发者加载推荐 CRAF 或 MARF ablation 配置
- **THEN** 配置文件名、`experiment.name` 和 `output.run_name` MUST 使用场景中立方法名
- **AND** 用户 MUST 能通过 dataset 场景覆盖在其它场景复用该方法配置

#### Scenario: 场景信息保留在数据和产物 metadata
- **WHEN** 训练或评估使用场景中立配置运行
- **THEN** dataset 配置和运行 metadata MUST 仍记录实际 scene / scene_id / scene_slug
- **AND** checkpoint registry MUST 能继续按场景目录或 metadata 区分产物
