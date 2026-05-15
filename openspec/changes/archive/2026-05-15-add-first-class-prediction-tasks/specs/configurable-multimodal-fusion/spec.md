## ADDED Requirements

### Requirement: Fusion objective 配置矩阵
系统 MUST 为 fusion 实验提供 objective-aware 配置入口，使同一模态集合能够分别运行 `beam`、`occlusion`、`position` 和 `multitask` 预测目标。配置命名 MUST 同时表达模态集合和预测目标。

#### Scenario: 五模态 objective 配置
- **WHEN** 用户查看 recommended 五模态 fusion 配置
- **THEN** 系统 MUST 提供或虚拟解析 beam、occlusion、position 和 multitask 四类 objective 入口
- **AND** 每个入口 MUST 使用相同的五模态集合 `[image, radar, gps, lidar, mmwave]`

#### Scenario: 配置名表达 objective
- **WHEN** 用户使用 objective-aware fusion 配置
- **THEN** 配置名或 virtual config stem MUST 包含 canonical 模态 slug 和 objective 名称
- **AND** 配置中的 `experiment.objective` MUST 与名称中的 objective 一致

#### Scenario: 旧配置兼容
- **WHEN** 用户继续使用既有 `configs/fusion/all_modalities_no_kd.yaml`
- **THEN** 系统 MUST 将该配置视为 `experiment.objective: beam`
- **AND** 系统 MUST 不要求用户修改旧运行命令

### Requirement: 模态失衡 objective 子集
fusion 配置系统 MUST 支持为模态失衡研究生成强模态、弱模态、单模态和全模态 objective 对照实验。每个 objective 配置 MUST 使用同一套 target 生成语义和同一套 metric 名称。

#### Scenario: strong-only occlusion 配置
- **WHEN** 用户请求 strong-only 模态集合的 occlusion fusion 配置
- **THEN** 系统 MUST 能生成只包含 strong modalities 的 fusion 配置
- **AND** 配置 MUST 设置 `experiment.objective: occlusion`

#### Scenario: weak-only position 配置
- **WHEN** 用户请求 weak-only 模态集合的 position fusion 配置
- **THEN** 系统 MUST 能生成只包含 weak modalities 的 fusion 配置
- **AND** 配置 MUST 设置 `experiment.objective: position`

#### Scenario: objective 间可比性
- **WHEN** 用户比较同一模态集合下的 beam、occlusion、position 和 multitask 结果
- **THEN** 系统 MUST 保持数据 split、target horizon、模态顺序和模型 backbone 默认配置一致
